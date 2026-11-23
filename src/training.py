import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from pathlib import Path


def train_one_epoch(
    model: nn.Module,
    train_loader: DataLoader,
    criterion: nn.Module,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
) -> float:
    """
    Train a PyTorch model for one epoch.

    Returns
    -------
    float
        Average training loss across all samples.
    """

    # Enable training mode
    model.train()

    total_loss = 0.0
    total_samples = 0

    for X_batch, y_batch in train_loader:

        # X_batch: (batch_size, 96, 1)
        # y_batch: (batch_size, 1)
        X_batch = X_batch.to(device)
        y_batch = y_batch.to(device)

        # Clear gradients from the previous batch
        optimizer.zero_grad()

        # Forward pass
        # predictions: (batch_size, 1)
        predictions = model(X_batch)

        # Calculate Mean Squared Error loss
        loss = criterion(predictions, y_batch)

        # Backpropagation Through Time (BPTT)
        loss.backward()

        # Update model parameters
        optimizer.step()

        # Accumulate sample-weighted loss
        batch_size = X_batch.size(0)
        total_loss += loss.item() * batch_size
        total_samples += batch_size

    return total_loss / total_samples

def validate_one_epoch(
    model: nn.Module,
    val_loader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
) -> float:
    """
    Evaluate the model on the validation dataset.

    Returns
    -------
    float
        Average validation loss across all samples.
    """

    # Switch to evaluation mode
    model.eval()

    total_loss = 0.0
    total_samples = 0

    # Disable gradient tracking during validation
    with torch.no_grad():

        for X_batch, y_batch in val_loader:

            # X_batch: (batch_size, 96, 1)
            # y_batch: (batch_size, 1)
            X_batch = X_batch.to(device)
            y_batch = y_batch.to(device)

            # Forward pass
            # predictions: (batch_size, 1)
            predictions = model(X_batch)

            # Calculate validation loss
            loss = criterion(predictions, y_batch)

            # Accumulate sample-weighted loss
            batch_size = X_batch.size(0)
            total_loss += loss.item() * batch_size
            total_samples += batch_size

    return total_loss / total_samples

def train_model(
    model: nn.Module,
    train_loader: DataLoader,
    val_loader: DataLoader,
    criterion: nn.Module,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    num_epochs: int = 20,
    checkpoint_path: Path | None = None,
    patience: int = 5,
) -> dict[str, list[float]]:
    """
    Train and validate a model with checkpointing and early stopping.

    The best model is selected using the lowest validation loss.
    Training stops after 'patience' consecutive epochs without
    validation improvement.
    """

    history = {
        "train_loss": [],
        "val_loss": [],
    }

    best_val_loss = float("inf")
    epochs_without_improvement = 0

    model.to(device)

    for epoch in range(num_epochs):

        # Train for one epoch
        train_loss = train_one_epoch(
            model, train_loader, criterion, optimizer, device
        )

        # Evaluate on validation data
        val_loss = validate_one_epoch(
            model, val_loader, criterion, device
        )

        # Record losses for plotting
        history["train_loss"].append(train_loss)
        history["val_loss"].append(val_loss)

        print(
            f"Epoch [{epoch + 1}/{num_epochs}] "
            f"Train Loss: {train_loss:.6f} | "
            f"Val Loss: {val_loss:.6f}"
        )

        # Check whether validation performance improved
        if val_loss < best_val_loss:

            best_val_loss = val_loss
            epochs_without_improvement = 0

            # Save the best-performing model
            if checkpoint_path is not None:
                checkpoint_path = Path(checkpoint_path)
                checkpoint_path.parent.mkdir(
                    parents=True, exist_ok=True
                )

                torch.save(
                    model.state_dict(),
                    checkpoint_path
                )

                print("  Best model saved.")

        else:
            epochs_without_improvement += 1

        # Stop if validation loss has not improved
        if epochs_without_improvement >= patience:
            print(
                f"Early stopping at epoch {epoch + 1}. "
                f"Best validation loss: {best_val_loss:.6f}"
            )
            break

    return history