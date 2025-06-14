"""test reconstruction models."""

import lightning as L
from torch.utils.data.dataloader import DataLoader
from torch.utils.data.dataset import TensorDataset
import torch
from anomalysuite.reconstructor.tranad import TranAD
from hypothesis import given, strategies as st
from hypothesis import settings


@settings(deadline=None)
@given(
    number_of_data=st.integers(min_value=1, max_value=10000),
    number_of_features=st.integers(min_value=1, max_value=50),
    sequence_length=st.integers(min_value=1, max_value=200),
    batch_size=st.integers(min_value=1, max_value=256),
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
