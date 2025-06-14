"""test reconstruction models."""

import lightning as L
import torch
from hypothesis import given, settings
from torch.utils.data.dataloader import DataLoader
from torch.utils.data.dataset import TensorDataset

from anomalysuite.reconstructor.tranad import TranAD
from tests.strategies import batch_size, number_of_data, number_of_features, sequence_length


@settings(deadline=None, max_examples=20)
@given(
    number_of_data=number_of_data,
    number_of_features=number_of_features,
    sequence_length=sequence_length,
    batch_size=batch_size,
)
def test_tranad(number_of_data: int, number_of_features: int, sequence_length: int, batch_size: int) -> None:
    """Test tranad training is working correctly.

    Args:
        number_of_data: number of data to generate.
        number_of_features: number of features length for model and data.
        sequence_length: sequence length for model and data.
        batch_size: dataloader batch size.

    """
    data = torch.rand(number_of_data, sequence_length, number_of_features)
    dataset = TensorDataset(data, data)
    dataloader = DataLoader(dataset, batch_size)

    model = TranAD(sequence_length=sequence_length, number_of_features=number_of_features)

    trainer = L.Trainer(fast_dev_run=True, enable_model_summary=False)
    trainer.fit(model, dataloader)
    assert trainer.state.finished
