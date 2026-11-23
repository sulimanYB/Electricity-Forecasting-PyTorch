from pathlib import Path
import pandas as pd


def load_and_combine_data(raw_data_dir: Path) -> pd.DataFrame:
    """
    Load and combine the original electricity consumption datasets.

    Parameters
    ----------
    raw_data_dir : Path
        Directory containing the original training and test CSV files.

    Returns
    -------
    pd.DataFrame
        Combined dataset sorted chronologically, containing
        'timestamp' and 'consumption' columns.
    """

    # Load the original datasets
    train_df = pd.read_csv(raw_data_dir / "electricity_train.csv")
    test_df = pd.read_csv(raw_data_dir / "electricity_test.csv")

    # Combine both datasets into a single DataFrame
    df = pd.concat([train_df, test_df], ignore_index=True)

    # Convert timestamps to datetime objects
    df["timestamp"] = pd.to_datetime(df["timestamp"])

    # Sort chronologically and reset the index
    df = df.sort_values("timestamp").reset_index(drop=True)

    return df


def validate_data(df: pd.DataFrame) -> None:
    """
    Validate the electricity consumption dataset.

    Checks:
    - Required columns are present.
    - No missing values.
    - No duplicate timestamps.
    - Timestamps are chronologically ordered.
    - Consecutive observations are exactly 15 minutes apart.

    Raises
    ------
    ValueError
        If any validation check fails.
    """

    # Check required columns
    required_columns = {"timestamp", "consumption"}

    if not required_columns.issubset(df.columns):
        raise ValueError("Dataset is missing required columns.")

    # Check missing values
    if df[["timestamp", "consumption"]].isnull().any().any():
        raise ValueError("Dataset contains missing values.")

    # Check duplicate timestamps
    if df["timestamp"].duplicated().any():
        raise ValueError("Dataset contains duplicate timestamps.")

    # Check chronological order
    if not df["timestamp"].is_monotonic_increasing:
        raise ValueError("Timestamps are not chronologically ordered.")

    # Check 15-minute sampling intervals
    time_diffs = df["timestamp"].diff().dropna()

    if not (time_diffs == pd.Timedelta(minutes=15)).all():
        raise ValueError("Dataset contains irregular time intervals.")

    print("Data validation passed successfully.")


def split_data(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Split the electricity consumption dataset chronologically.

    Training:   2011–2012
    Validation: 2013
    Test:       2014

    Parameters
    ----------
    df : pd.DataFrame
        Combined and chronologically sorted dataset.

    Returns
    -------
    tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]
        Training, validation and test DataFrames.
    """

    # Training period: before 2013
    train_df = df[df["timestamp"] < "2013-01-01"].copy()

    # Validation period: 2013
    val_df = df[
        (df["timestamp"] >= "2013-01-01")
        & (df["timestamp"] < "2014-01-01")
    ].copy()

    # Test period: 2014 onwards
    test_df = df[df["timestamp"] >= "2014-01-01"].copy()

    return train_df, val_df, test_df