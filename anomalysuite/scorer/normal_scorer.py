"""Normal scorer for scoring residuals."""

import lightning as L
import torch
import torch.nn as nn
from torch.optim import Adam


class NormalScorer(L.LightningModule):
    """Scorer for residuals using normal distributions."""

    def __init__(self, number_of_features: int) -> None:
        """Initialize model.

        Args:
            number_of_features: The number of parameter sets to calculate the residuals for each features.
                                If the data has 3 features, 3 mu and 3 sigma is required.

        """
        super().__init__()
        self.mu = nn.Parameter(torch.randn(number_of_features))  # (number_of_features,)
        self.log_sigma = nn.Parameter(torch.zeros(number_of_features))  # (number_of_features,)
        self.dist = torch.distributions.Normal

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass.

        Args:
            x: residuals, expects X - output from reconstructor.
                shape: (batch_size, sequence_length, number_of_features).

        Returns:
            scalar negative log likelihood.

        """
        dist = self.dist(loc=self.mu, scale=torch.exp(self.log_sigma))  # type: ignore
        negative_log_likelihood = dist.log_prob(x)  # type: ignore
        return negative_log_likelihood.sum(dim=1).mean()  # type: ignore

    def training_step(self, batch: torch.Tensor) -> torch.Tensor:
        """Trainig step.

        Args:
            batch: input training data.

        Returns:
            log likelihood to minimize.

        """
        x_batch = batch[0]
        neg_log_likelihood = self(x_batch)
        log_likelihood = -neg_log_likelihood
        return log_likelihood  # type: ignore

    def calculate_score_by_cdf(self, residual: torch.Tensor) -> torch.Tensor:
        """Return cdf based score from 0~1, which is 1 is normal.

        Args:
            residual:
                residual between x and xhat.
                shape: (batch_size, sequence_length, number_of_features).

        Returns:
            anomaly score for each point.
            shape: same as input residual.

        """
        dist = self.dist(loc=self.mu, scale=torch.exp(self.log_sigma))  # type: ignore
        cdf = dist.cdf(residual)  # type: ignore
        return 2 * torch.min(cdf, 1 - cdf)  # type: ignore

    def configure_optimizers(self) -> list[torch.optim.Optimizer]:
        """Set Optimizers.

        Returns:
            optimizer.

        """
        optimizer = Adam(self.parameters())
        return [optimizer]
