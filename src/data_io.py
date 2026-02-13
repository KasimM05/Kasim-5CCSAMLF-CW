from pathlib import Path

import pandas as pd

from src.config import TARGET_COL, TEST_FILE, TRAIN_FILE


def resolve_data_path(filename: str) -> Path:
    candidates = [Path("data") / filename, Path(filename)]
    for p in candidates:
        if p.exists():
            return p
    raise FileNotFoundError(f"Could not find {filename}. Checked: {candidates}")


def load_train_test() -> tuple[pd.DataFrame, pd.Series, pd.DataFrame]:
    train_df = pd.read_csv(resolve_data_path(TRAIN_FILE))
    test_df = pd.read_csv(resolve_data_path(TEST_FILE))

    if TARGET_COL not in train_df.columns:
        raise ValueError(f"Training data must contain target column: {TARGET_COL}")

    X = train_df.drop(columns=[TARGET_COL])
    y = train_df[TARGET_COL]

    missing_in_test = [c for c in X.columns if c not in test_df.columns]
    if missing_in_test:
        raise ValueError(f"Test data is missing required columns: {missing_in_test}")

    X_test = test_df[X.columns].copy()
    return X, y, X_test
