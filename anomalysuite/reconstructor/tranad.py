"""TranAD and PyTorch Lightning wrapper."""

import math

import lightning as L
import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.optim.lr_scheduler import StepLR

N_WINDOWS = 10


class PositionalEncoding(nn.Module):
    """Positional Encoding layer."""

    pe: torch.Tensor

    def __init__(self, d_model: int, dropout: float = 0.1, max_len: int = 5000) -> None:
        """Init model.

        Args:
            d_model: Dimension of model, usually d_model = 2 * (the number of data column).
            dropout: Dropout ratio. Defaults to 0.1.
            max_len: Window length. Defaults to 5000.

        """
        super().__init__()
        self.dropout: nn.Dropout = nn.Dropout(p=dropout)

        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model).float() * (-math.log(10000.0) / d_model))
        pe += torch.sin(position * div_term)
        pe += torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)
        self.register_buffer("pe", pe)

    def forward(self, x: torch.Tensor, pos: int = 0) -> torch.Tensor:
        """Forward pass.

        Args:
            x: Input.
            pos (int, optional): Starting position. Defaults to 0.

        Returns:
            torch.Tensor: X with the positional encodings.

        """
        x = x + self.pe[pos : pos + x.size(0), :]
        x = self.dropout(x)
        return x


class TranAD(L.LightningModule):
    """LightningModule for training TranAD."""

    def __init__(self, sequence_length: int, number_of_features: int) -> None:
        """Initialize model.

        TranAD model from https://github.com/imperial-qore/TranAD.

        Args:
            sequence_length: window size of rolling windowed data.
            number_of_features: number of features, normally number of columns.

        """
        super().__init__()
        self.number_of_features = number_of_features
        self.sequence_length = sequence_length
        self.doubled_dimension = 2 * number_of_features
        self.pos_encoder = PositionalEncoding(self.doubled_dimension, 0.1, self.sequence_length)
        encoder_layers = nn.TransformerEncoderLayer(
            d_model=self.doubled_dimension,
            nhead=number_of_features,
            dim_feedforward=16,
            dropout=0.1,
        )
        self.transformer_encoder = nn.TransformerEncoder(encoder_layers, 1)
        decoder_layers1 = nn.TransformerDecoderLayer(
            d_model=self.doubled_dimension,
            nhead=number_of_features,
            dim_feedforward=16,
            dropout=0.1,
        )
        self.transformer_decoder_without_context = nn.TransformerDecoder(decoder_layers1, 1)
        decoder_layers2 = nn.TransformerDecoderLayer(
            d_model=self.doubled_dimension,
            nhead=number_of_features,
            dim_feedforward=16,
            dropout=0.1,
        )
        self.transformer_decoder_with_context = nn.TransformerDecoder(decoder_layers2, 1)
        self.fcn = nn.Sequential(
            nn.Linear(self.doubled_dimension, number_of_features),
            nn.Sigmoid(),
        )

    def forward(self, src: torch.Tensor, tgt: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Forward pass.

        Args:
            src: Source.
            tgt: Target.

        Returns:
            Prediction values.

        """
        # Phase 1 - Without anomaly scores
        c = torch.zeros_like(src)
        x1 = self.fcn(self.transformer_decoder_without_context(*self.encode(src, c, tgt)))
        # Phase 2 - With anomaly scores
        c = (x1 - src) ** 2
        x2 = self.fcn(self.transformer_decoder_with_context(*self.encode(src, c, tgt)))
        return x1, x2

    def encode(self, src: torch.Tensor, c: torch.Tensor, tgt: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Encode tensor.

        Args:
            src: Source.
            c: Context.
            tgt: Target.

        Returns:
            tuple: Encoded tensor tuple.

        """
        src = torch.cat((src, c), dim=2)
        src = src * math.sqrt(self.number_of_features)
        src = self.pos_encoder(src)
        memory = self.transformer_encoder(src)
        tgt = tgt.repeat(1, 1, 2)
        return tgt, memory

    def training_step(self, batch: torch.Tensor) -> torch.Tensor:
        """Train using dataloader or batch data.

        Args:
            batch: X input

        Returns:
            loss for backpropagation.

        """
        loss = torch.tensor([])
        d, _ = batch
        assert d.shape == (d.shape[0], self.sequence_length, self.number_of_features), (
            f"입력 텐서 {d.shape}가 예상한 텐서 ({d.shape[0]},{self.sequence_length},{self.number_of_features})과 다릅니다."
        )
        epoch = self.current_epoch + 1
        local_bs = d.shape[0]
        window = d.permute(1, 0, 2)
        elem = window[-1, :, :].view(1, local_bs, self.number_of_features)
        z = self(window, elem)

        loss = (1 / epoch) * nn.functional.mse_loss(z[0], elem, reduction="mean") + (
            1 - 1 / epoch
        ) * nn.functional.mse_loss(z[1], elem, reduction="mean")
        z = z[1]

        self.log("train_loss", loss, on_step=True, on_epoch=True, prog_bar=True, logger=True)

        return loss

    def configure_optimizers(
        self,
    ) -> tuple[list[torch.optim.Optimizer], list[torch.optim.lr_scheduler.LRScheduler]]:
        """Set Optimizers.

        Returns:
            List: optimizers with scheduler.

        """
        optimizer = AdamW(self.parameters(), lr=0.0001, weight_decay=1e-5)
        scheduler = StepLR(optimizer, step_size=5, gamma=0.9)
        return [optimizer], [scheduler]
