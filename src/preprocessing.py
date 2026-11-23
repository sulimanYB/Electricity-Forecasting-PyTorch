import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler


def scale_data(
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
    test_df: pd.DataFrame,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, StandardScaler]:
    """
    Standardise electricity consumption using training statistics only.

    Parameters
    ----------
    train_df, val_df, test_df : pd.DataFrame
        Chronologically split datasets containing a 'consumption' column.

    Returns
    -------
    train_scaled, val_scaled, test_scaled : np.ndarray
        Scaled arrays with shape (N, 1).
    scaler : StandardScaler
        Fitted scaler for transforming future data and
        converting predictions back to original units.
    """

    # Initialise the scaler
    scaler = StandardScaler()

    # Fit using training data only to prevent data leakage
    train_scaled = scaler.fit_transform(train_df[["consumption"]])

    # Apply the same training statistics to validation and test data
    val_scaled = scaler.transform(val_df[["consumption"]])
    test_scaled = scaler.transform(test_df[["consumption"]])

    return train_scaled, val_scaled, test_scaled, scaler

def create_sequences(
    data: np.ndarray,
    seq_length: int = 96,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Convert a time series into input sequences and next-step targets.

    Parameters
    ----------
    data : np.ndarray
        Scaled time series with shape (N, 1).
    seq_length : int
        Number of historical observations per input sequence.

    Returns
    -------
    X : np.ndarray
        Input sequences with shape (N - seq_length, seq_length, 1).
    y : np.ndarray
        Next-step targets with shape (N - seq_length, 1).
    """

    X, y = [], []

    # Sliding window: use the previous 96 observations
    # to predict the next electricity consumption value.
    for i in range(len(data) - seq_length):
        X.append(data[i:i + seq_length])
        y.append(data[i + seq_length])

    # Convert to float32 for compatibility with PyTorch
    X = np.array(X, dtype=np.float32)
    y = np.array(y, dtype=np.float32)

    return X, y

def prepare_sequences(
    train_scaled: np.ndarray,
    val_scaled: np.ndarray,
    test_scaled: np.ndarray,
    seq_length: int = 96,
) -> tuple[
    tuple[np.ndarray, np.ndarray],
    tuple[np.ndarray, np.ndarray],
    tuple[np.ndarray, np.ndarray],
]:
    """
    Prepare training, validation and test sequences.

    Includes the previous 96 observations as historical context
    for the validation and test periods.

    Returns
    -------
    train_data, val_data, test_data : tuple
        Each contains (X, y), where:
        X has shape (num_sequences, seq_length, 1)
        y has shape (num_sequences, 1)
    """

    # Training: sequences entirely within the training period
    X_train, y_train = create_sequences(train_scaled, seq_length)

    # Validation: include the last 96 training observations
    val_with_context = np.concatenate(
        [train_scaled[-seq_length:], val_scaled],
        axis=0,
    )
    X_val, y_val = create_sequences(val_with_context, seq_length)

    # Test: include the last 96 validation observations
    test_with_context = np.concatenate(
        [val_scaled[-seq_length:], test_scaled],
        axis=0,
    )
    X_test, y_test = create_sequences(test_with_context, seq_length)

    return (X_train, y_train), (X_val, y_val), (X_test, y_test)