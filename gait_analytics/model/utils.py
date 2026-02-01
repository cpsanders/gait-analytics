import polars as pl
import numpy as np
from numpy.typing import NDArray
from sklearn.preprocessing import StandardScaler  # type: ignore[import-untyped]

def train_val_split_data(
    df: pl.DataFrame,
    feature_columns: list[str],
    target_column: str,
    window_size: int,
    train_size: float = 0.8,
):    
    split_idx = int(len(df) * train_size)
    train_df = df.head(split_idx)
    val_df = df.tail(-split_idx - window_size) # create a gap between the train and validation data of window_size rows
    
    feature_scaler = StandardScaler()
    target_scaler = StandardScaler()
    
    train_features = feature_scaler.fit_transform(train_df.select(feature_columns).to_numpy())
    train_targets = target_scaler.fit_transform(train_df.select(target_column).to_numpy().reshape(-1, 1))
    
    # Apply scaling to the target set
    val_features = feature_scaler.transform(val_df.select(feature_columns).to_numpy())
    val_targets = target_scaler.transform(val_df.select(target_column).to_numpy().reshape(-1, 1))
    
    return (train_features, train_targets), (val_features, val_targets), feature_scaler, target_scaler


def create_cnn_windows(
    feature_array: NDArray[np.floating], 
    target_array: NDArray[np.floating],
    feature_columns: list[str], 
    window_size: int = 100, 
    stride: int = 10
):
    """
    Converts a flat Polars DataFrame into 3D Windows for 1D-CNNs.
    Output Shape: (N_Samples, Num_Features, Window_Size)

    Args:
        feature_array: Numpy array of feature data
        target_arrauy: Numpy array of target data
        feature_columns: List of column names to use as inputs
        target: The name of the target column (speed)
        window_size: Number of time-steps per sequence (e.g., 100 for 2s @ 50Hz)
        stride: How many rows to move the window (1 = heavily overlapping)
    """
    num_samples = (len(feature_array) - window_size) // stride + 1
    
    # Use strides to create (Samples, Window_Size, Features)
    X = np.lib.stride_tricks.as_strided(
        feature_array,
        shape=(num_samples, window_size, len(feature_columns)),
        strides=(
            feature_array.strides[0] * stride, 
            feature_array.strides[0], 
            feature_array.strides[1]
        )
    )

    # TRANSPOSE: Swap axes to go from (Samples, Window Size, Features) --> (Samples, Features, Window_Size)
    # This matches PyTorch Conv1d requirements: (N, C, L)
    X = X.transpose(0, 2, 1)

    # Calculate X first, then just grab the indices that match the end of each window
    indices = np.arange(window_size - 1, window_size - 1 + num_samples * stride, stride)
    y = target_array[indices]

    return X, y
