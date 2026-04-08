"""
datasets.py - Load and describe the 5 evaluation datasets.
"""

import pandas as pd
import numpy as np
from pathlib import Path

DATA_DIR = Path(__file__).parent / "data"


def load_titanic() -> tuple:
    df = pd.read_csv(DATA_DIR / "titanic.csv")
    meta = {
        "name": "titanic",
        "description": "Titanic passenger survival dataset. Predict survival based on passenger attributes.",
        "columns": list(df.columns),
        "dtypes": {col: str(df[col].dtype) for col in df.columns},
        "shape": df.shape,
    }
    return df, meta


def load_iris() -> tuple:
    iris_path = DATA_DIR / "iris.csv"
    if not iris_path.exists():
        from sklearn.datasets import load_iris as _load_iris
        data = _load_iris(as_frame=True)
        df = data.frame
        df.to_csv(iris_path, index=False)
    else:
        df = pd.read_csv(iris_path)
    meta = {
        "name": "iris",
        "description": "Iris flower species classification dataset with sepal/petal measurements.",
        "columns": list(df.columns),
        "dtypes": {col: str(df[col].dtype) for col in df.columns},
        "shape": df.shape,
    }
    return df, meta


def load_sleep_health() -> tuple:
    df = pd.read_csv(DATA_DIR / "sleep_health.csv")
    meta = {
        "name": "sleep_health",
        "description": "Sleep health and lifestyle dataset tracking sleep duration, quality, physical activity, and health metrics.",
        "columns": list(df.columns),
        "dtypes": {col: str(df[col].dtype) for col in df.columns},
        "shape": df.shape,
    }
    return df, meta


def load_customer_segments() -> tuple:
    df = pd.read_csv(DATA_DIR / "customer_segments.csv")
    meta = {
        "name": "customer_segments",
        "description": "Customer segmentation dataset with demographic and spending behavior data.",
        "columns": list(df.columns),
        "dtypes": {col: str(df[col].dtype) for col in df.columns},
        "shape": df.shape,
    }
    return df, meta


def load_credit_fraud() -> tuple:
    credit_path = DATA_DIR / "credit_fraud.csv"
    if not credit_path.exists():
        np.random.seed(42)
        n = 1000
        df = pd.DataFrame({
            "amount": np.random.exponential(100, n).round(2),
            "is_fraud": np.random.binomial(1, 0.1, n),
            "transaction_type": np.random.choice(["purchase", "withdrawal", "transfer", "payment"], n),
            "hour_of_day": np.random.randint(0, 24, n),
            "merchant_category": np.random.choice(["retail", "food", "travel", "online", "atm"], n),
            "card_present": np.random.binomial(1, 0.7, n),
        })
        df.to_csv(credit_path, index=False)
    else:
        df = pd.read_csv(credit_path)
    meta = {
        "name": "credit_fraud",
        "description": "Synthetic credit card fraud detection dataset with transaction features.",
        "columns": list(df.columns),
        "dtypes": {col: str(df[col].dtype) for col in df.columns},
        "shape": df.shape,
    }
    return df, meta


DATASET_LOADERS = {
    "titanic": load_titanic,
    "iris": load_iris,
    "sleep_health": load_sleep_health,
    "customer_segments": load_customer_segments,
    "credit_fraud": load_credit_fraud,
}


def load_dataset(name: str) -> tuple:
    if name not in DATASET_LOADERS:
        raise ValueError(f"Unknown dataset: {name}. Available: {list(DATASET_LOADERS.keys())}")
    return DATASET_LOADERS[name]()
