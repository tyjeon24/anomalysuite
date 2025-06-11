"""pytest fixtures."""

import pytest
import torch
from torch.utils.data.dataloader import DataLoader
from torch.utils.data.dataset import TensorDataset


@pytest.fixture(scope="session")
def hyperparameters() -> dict[str, int]:
    """Hyperparameters for model init and dataloader setup.

    Yields:
        dictionary with hyperparameters

    """
    hyperparameters = {"number_of_data": 100, "sequence_length": 10, "number_of_features": 2, "batch_size": 3}
    return hyperparameters


@pytest.fixture(scope="session")
def dataloader(hyperparameters: dict[str, int]) -> DataLoader[tuple[torch.Tensor, ...]]:
    """Return dataloader containing random values.

    Args:
        hyperparameters: hyperparameter fixtrue

    Yields:
        dataloader.

    """
    data = torch.rand(
        hyperparameters["number_of_data"], hyperparameters["sequence_length"], hyperparameters["number_of_features"]
    )
    dataset = TensorDataset(data, data)
    dataloader = DataLoader(dataset, hyperparameters["batch_size"])
    return dataloader
