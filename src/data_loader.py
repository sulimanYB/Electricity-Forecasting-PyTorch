import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset


def create_dataloaders(
    train_data: tuple[np.ndarray, np.ndarray],
    val_data: tuple[np.ndarray, np.ndarray],
    test_data: tuple[np.ndarray, np.ndarray],
    batch_size: int = 32,
) -> tuple[DataLoader, DataLoader, DataLoader]:
    """
    Convert NumPy sequences into PyTorch DataLoaders.

    Parameters
    ----------
    train_data, val_data, test_data : tuple
        Each contains (X, y), where:
        X shape: (num_sequences, seq_length, input_size)
        y shape: (num_sequences, 1)

    batch_size : int
        Number of sequences per batch.

    Returns
    -------
    train_loader, val_loader, test_loader : DataLoader
        DataLoaders ready for RNN, LSTM and GRU training.
    """

    def make_dataset(data):
        """Convert NumPy inputs and targets into a TensorDataset."""

        X, y = data

        # Preserve tensor dimensions and ensure float32 data type
        X_tensor = torch.as_tensor(X, dtype=torch.float32)
        y_tensor = torch.as_tensor(y, dtype=torch.float32)

        return TensorDataset(X_tensor, y_tensor)

    # Create datasets
    train_dataset = make_dataset(train_data)
    val_dataset = make_dataset(val_data)
    test_dataset = make_dataset(test_data)

    # Shuffle training sequences, but preserve temporal order
    # inside each individual sequence.
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
    )

    # Keep validation and test sequences in chronological order
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
    )

    return train_loader, val_loader, test_loader

from pathlib import Path


def load_prepared_data(
    tensor_path: Path,
    batch_size: int = 32,
) -> tuple[DataLoader, DataLoader, DataLoader]:
    """
    Load saved PyTorch tensors and create DataLoaders.

    Parameters
    ----------
    tensor_path : Path
        Path to the saved electricity_sequences.pt file.
    batch_size : int
        Number of sequences per mini-batch.

    Returns
    -------
    train_loader, val_loader, test_loader : DataLoader
        DataLoaders for model training and evaluation.
    """

    # Load the prepared tensors from disk.
    # weights_only=True avoids unrestricted pickle loading.
    data = torch.load(tensor_path, map_location="cpu", weights_only=True)

    # Extract inputs and targets
    train_data = (data["X_train"], data["y_train"])
    val_data = (data["X_val"], data["y_val"])
    test_data = (data["X_test"], data["y_test"])

    # Reuse our existing DataLoader function
    return create_dataloaders(
        train_data,
        val_data,
        test_data,
        batch_size=batch_size,
    )