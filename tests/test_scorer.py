"""test scorer modules."""

import lightning as L
import torch
from einops import rearrange
from hypothesis import assume, given, settings
from torch.utils.data import DataLoader, TensorDataset

from anomalysuite.scorer.normal_scorer import NormalScorer
from tests.strategies import batch_size, number_of_data, number_of_features, sequence_length


def dataloader_from_normal_distribution(
    mu: torch.Tensor, sigma: torch.Tensor, number_of_data: int, sequence_length: int, batch_size: int
) -> DataLoader[tuple[torch.Tensor, ...]]:
    """Return dataloader for model.

    Args:
        mu: normal distribution parameter.
        sigma: normal distribution parameter.
        number_of_data: number_of_data.
        sequence_length: sequence_length.
        batch_size: batch_size.

    Returns:
        torch dataloader.

    """
    assert mu.shape == sigma.shape, "mu and sigma must be shaped identically"
    dist = torch.distributions.Normal(loc=mu, scale=sigma)  # type: ignore

    X = dist.sample((number_of_data,))  # type: ignore
    X_rolled = rearrange(
        X.unfold(0, sequence_length, 1),
        "length number_of_featurs sequence_length -> length sequence_length number_of_featurs",
    )
    dataset = TensorDataset(X_rolled)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    return dataloader


@settings(deadline=None, max_examples=20)
@given(
    number_of_features=number_of_features,
    number_of_data=number_of_data,
    sequence_length=sequence_length,
    batch_size=batch_size,
)
def test_normal_scorer(number_of_features: int, number_of_data: int, sequence_length: int, batch_size: int) -> None:
    """Test the training sequnce of normal scorer.

    Args:
        number_of_features: number of features for data.
        number_of_data: number of data.
        sequence_length: sequence length for rolling data.
        batch_size: batch size.

    """
    assume(number_of_data > sequence_length)  # Ensure data is more than sequence_length.

    mu, sigma = torch.rand(number_of_features), torch.rand(number_of_features)
    dataloader = dataloader_from_normal_distribution(
        mu=mu,
        sigma=sigma,
        number_of_data=number_of_data,
        sequence_length=sequence_length,
        batch_size=batch_size,
    )

    model = NormalScorer(number_of_features)

    trainer = L.Trainer(fast_dev_run=True, enable_model_summary=False)
    trainer.fit(model, dataloader)
    assert trainer.state.finished
