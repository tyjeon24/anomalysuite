"""tensor related processors."""

import lightning as L
import pandas as pd
import torch
from einops import rearrange
from torch.utils.data.dataloader import DataLoader
from torch.utils.data.dataset import TensorDataset


def expand_rolled_array(arr: torch.Tensor) -> torch.Tensor:
    """Expand 2d rolled array to 2d unrolled array.

    input shape = (row, col)
    output shape = (row - col + 1, col)

    Args:
        arr: 2d torch tensor.

    Returns:
        torch.Tensor

    Examples:
    >>> expand_rolled_array(
        torch.tensor(
            [
                [1, 2, 3],
                [2, 3, 4],
                [3, 4, 5],
            ]
        )
    )

    tensor([[1., 2., 3., nan, nan],
            [nan, 2., 3., 4., nan],
            [nan, nan, 3., 4., 5.]])

    >>> expand_rolled_array(
        torch.tensor(
            [
                [10, 20],
                [20, 30],
                [30, 40],
                [40, 50],
            ]
        )
    )

    tensor([[10., 20., nan, nan, nan],
            [nan, 20., 30., nan, nan],
            [nan, nan, 30., 40., nan],
            [nan, nan, nan, 40., 50.]])

    """
    assert arr.ndim == 2, "array must be shaped 2d."
    sequence_length = arr.shape[1]

    orig_len = sequence_length + arr.shape[0] - 1
    result = torch.full((arr.shape[0], orig_len), torch.nan)  # np.nan으로 가득 찬 배열 생성
    for i in range(arr.shape[0]):
        result[i, i : i + sequence_length] = arr[i]
    return result


def nan_to_insufficient_columns(arr: torch.Tensor) -> torch.Tensor:
    """Make nan masked 2d tensor where the number of non-na values are not enough.

    It is required for post-processing step of reconstruction based AI model.
    The reconstruction based AI models take rolled tensor as input and returns the same shape.
    However, the end user wants the original 2d input, not 3d so 3d tensor must be aggregated.
    In this procedure, rolled tensor will lose some of the input data because the data on tails are missing.
    For example, [1,2,3] can be [[1,2], [2,3]] then 1 and 3 is cannot be used,
    because they do not have enough count to aggreagte. 2 has two values so it can be aggregated.

    input shape = (row, col)
    output shape = (row, col)

    1. count the non-na values
    2. replace all the values which have insufficient non-na count with torch.nan.

    Args:
        arr: 2d torch tensor.

    Returns:
        torch.Tensor.

    Examples:
        >>> nan_to_insufficient_columns(
            torch.tensor(
                [
                    [1.0, 2.0, 3.0, torch.nan, torch.nan],
                    [torch.nan, 2.0, 3.0, 4.0, torch.nan],
                    [torch.nan, torch.nan, 3.0, 4.0, 5.0],
                ]
            )
        )

        tensor([[nan, nan, 3., nan, nan],
                [nan, nan, 3., nan, nan],
                [nan, nan, 3., nan, nan]])

        >>> nan_to_insufficient_columns(
            torch.tensor(
                [
                    [10.0, 20.0, torch.nan, torch.nan, torch.nan],
                    [torch.nan, 20.0, 30.0, torch.nan, torch.nan],
                    [torch.nan, torch.nan, 30.0, 40.0, torch.nan],
                    [torch.nan, torch.nan, torch.nan, 40.0, 50.0],
                ]
            )
        )

        tensor([[nan, 20., nan, nan, nan],
                [nan, 20., 30., nan, nan],
                [nan, nan, 30., 40., nan],
                [nan, nan, nan, 40., nan]])

    """
    non_na_count = (~torch.isnan(arr)).sum(dim=0)
    mask = non_na_count == max(non_na_count)
    mask_broadcast = mask.repeat(arr.size(0), 1)
    result = torch.full_like(arr, torch.nan)
    result[mask_broadcast] = arr[mask_broadcast]
    return result


def predict_dataframe(model: L.LightningModule, df_data: pd.DataFrame) -> pd.DataFrame:
    """Predict pandas dataframe and return pandas dataframe that has same shape.

    Args:
        model: lightning module.
        df_data: pandas dataframe.

    Raises:
        ValueError: when dataframe columns are not equal to model.number_of_features.

    Returns:
        pandas dataframe.

    """
    if df_data.shape[1] != model.number_of_features:
        raise ValueError("the number_of_features of the model must be equal to the columns of data.")

    sequence_length: int = model.__dict__.get("sequence_length", -1)
    if sequence_length == -1:
        raise KeyError("model must contain sequence_length:int variable.")

    tensor_raw = torch.tensor(df_data.to_numpy(), dtype=torch.float)
    tensor_rolled = tensor_raw.unfold(0, sequence_length, 1)
    x = rearrange(
        tensor_rolled, "length number_of_features sequence_length -> length sequence_length number_of_features"
    )
    predict_dataloader = DataLoader(TensorDataset(x, x), batch_size=int(1e10))
    pred = L.Trainer().predict(model, predict_dataloader)
    if not pred or not isinstance(pred, list):
        raise ValueError("prediction failed.")
    else:
        xhat: torch.Tensor = pred[0]  # type: ignore

    df_hat = pd.DataFrame(index=df_data.index)
    for column_index, column_name in enumerate(df_data.columns):
        column = xhat[:, :, column_index]
        df_hat[column_name] = nan_to_insufficient_columns(expand_rolled_array(column)).nanmean(dim=0)

    return df_hat
