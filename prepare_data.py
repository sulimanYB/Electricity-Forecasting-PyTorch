from pathlib import Path

import joblib
import torch

from src.data_processing import (
    load_and_combine_data,
    validate_data,
    split_data,
)
from src.preprocessing import scale_data, prepare_sequences


# Project configuration
PROJECT_DIR = Path(__file__).resolve().parent
RAW_DATA_DIR = PROJECT_DIR / "data" / "raw"
PROCESSED_DIR = PROJECT_DIR / "data" / "processed"
TENSOR_DIR = PROJECT_DIR / "data" / "tensors"

SEQ_LENGTH = 96
BATCH_SIZE = 32


def main():
    """Prepare and save electricity consumption datasets."""

    # 1. Load and validate the original data
    df = load_and_combine_data(RAW_DATA_DIR)
    validate_data(df)

    # 2. Save the reconstructed chronological dataset
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    df.to_csv(PROCESSED_DIR / "electricity_full.csv", index=False)

    # 3. Split chronologically
    train_df, val_df, test_df = split_data(df)

    # 4. Scale using training statistics only
    train_scaled, val_scaled, test_scaled, scaler = scale_data(
        train_df, val_df, test_df
    )

    # 5. Create sequences, preserving historical context
    train_data, val_data, test_data = prepare_sequences(
        train_scaled, val_scaled, test_scaled, SEQ_LENGTH
    )

    # 6. Convert NumPy arrays into PyTorch tensors
    X_train, y_train = train_data
    X_val, y_val = val_data
    X_test, y_test = test_data

    # 7. Save the prepared tensors
    TENSOR_DIR.mkdir(parents=True, exist_ok=True)

    torch.save(
        {
            "X_train": torch.from_numpy(X_train),
            "y_train": torch.from_numpy(y_train),
            "X_val": torch.from_numpy(X_val),
            "y_val": torch.from_numpy(y_val),
            "X_test": torch.from_numpy(X_test),
            "y_test": torch.from_numpy(y_test),
            "seq_length": SEQ_LENGTH,
            "batch_size": BATCH_SIZE,
        },
        TENSOR_DIR / "electricity_sequences.pt",
    )

    # 8. Save the fitted scaler
    joblib.dump(
        scaler,
        TENSOR_DIR / "consumption_scaler.pkl",
    )

    print("\nData preparation completed successfully.")
    print(f"Training sequences:   {len(X_train):,}")
    print(f"Validation sequences: {len(X_val):,}")
    print(f"Test sequences:       {len(X_test):,}")


if __name__ == "__main__":
    main()