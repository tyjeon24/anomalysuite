"""test reconstruction models."""

import lightning as L
from torch.utils.data.dataloader import DataLoader
from torch.utils.data.dataset import TensorDataset

from anomalysuite.reconstructor.tranad import TranAD


def test_tranad(hyperparameters: dict[str, int], dataloader: DataLoader[TensorDataset]) -> None:
    """Test tranad training is working correctly.

    Args:
        hyperparameters: hyperparametrs from fixture.
        dataloader: dataloader from fixture.

    """
    model = TranAD(
        sequence_length=hyperparameters["sequence_length"], number_of_features=hyperparameters["number_of_features"]
    )
    trainer = L.Trainer(fast_dev_run=True)
    trainer.fit(model, dataloader)
    assert trainer.state.finished
