import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.preprocessing import StandardScaler


def generate_predictions(
    model: nn.Module,
    data_loader: DataLoader,
    device: torch.device,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Generate predictions and collect actual target values.

    Parameters
    ----------
    model : nn.Module
        Trained RNN, LSTM or GRU model.
    data_loader : DataLoader
        Validation or test DataLoader.
    device : torch.device
        Device used for inference.

    Returns
    -------
    predictions : np.ndarray
        Model predictions with shape (N, 1).
    targets : np.ndarray
        Actual target values with shape (N, 1).
    """

    # Set the model to evaluation mode
    model.eval()

    predictions = []
    targets = []

    # Disable gradient tracking during inference
    with torch.no_grad():

        for X_batch, y_batch in data_loader:

            # Input: (batch_size, 96, 1)
            X_batch = X_batch.to(device)

            # Output: (batch_size, 1)
            y_pred = model(X_batch)

            # Move predictions and targets to CPU
            predictions.append(y_pred.cpu().numpy())
            targets.append(y_batch.cpu().numpy())

    # Combine predictions from all mini-batches
    # (num_batches, variable_batch_size, 1) -> (N, 1)
    predictions = np.concatenate(predictions, axis=0)
    targets = np.concatenate(targets, axis=0)

    return predictions, targets

def calculate_metrics(
    predictions: np.ndarray,
    targets: np.ndarray,
    scaler: StandardScaler,
) -> dict[str, float]:
    """
    Calculate forecasting metrics in original consumption units.

    Parameters
    ----------
    predictions : np.ndarray
        Scaled model predictions, shape (N, 1).
    targets : np.ndarray
        Scaled actual values, shape (N, 1).
    scaler : StandardScaler
        Scaler fitted on the training dataset only.

    Returns
    -------
    dict[str, float]
        MAE, MSE and RMSE in original consumption units.
    """

    # Convert scaled values back to original consumption units
    predictions_original = scaler.inverse_transform(predictions)
    targets_original = scaler.inverse_transform(targets)

    # Calculate evaluation metrics
    mae = mean_absolute_error(targets_original, predictions_original)
    mse = mean_squared_error(targets_original, predictions_original)
    rmse = np.sqrt(mse)

    return {
        "MAE": float(mae),
        "MSE": float(mse),
        "RMSE": float(rmse),
    }