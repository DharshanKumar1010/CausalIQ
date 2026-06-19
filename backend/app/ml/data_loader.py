"""Data loading utilities for CausalIQ — real Criteo data and synthetic fallback."""

import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler


def load_criteo_data(
    filepath: str = 'data/processed/criteo_sample.csv',
    sample_size: int | None = None,
    seed: int = 42,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, list[str]]:
    """Load and preprocess the Criteo Uplift v2.1 dataset from a local CSV.

    Args:
        filepath: Path to the processed Criteo CSV file.
        sample_size: If set, randomly sample this many rows from the full file.
        seed: Random seed for reproducible sampling.

    Returns:
        Tuple of (X, treatment, y, feature_cols) where:
            X: Normalized float32 feature matrix of shape (n_samples, 12).
            treatment: Binary treatment indicator array of shape (n_samples,).
            y: Binary conversion outcome array of shape (n_samples,).
            feature_cols: List of the 12 feature column name strings.
    """
    feature_cols = ['f0', 'f1', 'f2', 'f3', 'f4', 'f5', 'f6', 'f7', 'f8', 'f9', 'f10', 'f11']

    df = pd.read_csv(filepath)

    if sample_size is not None:
        df = df.sample(n=sample_size, random_state=seed)

    X = df[feature_cols].values.astype(np.float32)
    treatment = df['treatment'].values.astype(int)
    y = df['conversion'].values.astype(int)

    scaler = MinMaxScaler()
    X = scaler.fit_transform(X).astype(np.float32)

    print(f'Loaded Criteo dataset: {X.shape}')
    print(f'Treatment ratio: {treatment.mean():.1%}')
    print(f'Conversion rate: {y.mean():.3%}')
    print(f'Treated: {treatment.sum():,} | Control: {(1 - treatment).sum():,}')

    return X, treatment, y, feature_cols


def generate_synthetic_campaign_data(
    n_samples: int = 10000, seed: int = 42
) -> tuple[np.ndarray, np.ndarray, np.ndarray, list[str], np.ndarray]:
    """Generate synthetic marketing campaign data with known heterogeneous treatment effects.

    Kept for fast local testing without requiring the Criteo dataset.

    Args:
        n_samples: Number of customer samples to generate.
        seed: Random seed for reproducibility.

    Returns:
        Tuple of (X, treatment, y, feature_names, true_uplift) where:
            X: Normalized feature matrix of shape (n_samples, 5).
            treatment: Binary treatment assignment array of shape (n_samples,).
            y: Binary outcome array of shape (n_samples,).
            feature_names: List of feature name strings.
            true_uplift: Ground-truth individual treatment effect array of shape (n_samples,).
    """
    np.random.seed(seed)

    age = np.random.randint(18, 81, size=n_samples).astype(float)
    income = np.random.uniform(20000, 200000, size=n_samples)
    engagement = np.random.uniform(0, 1, size=n_samples)
    recency = np.random.uniform(0, 365, size=n_samples)
    frequency = np.random.uniform(0, 100, size=n_samples)

    raw_features = np.column_stack([age, income, engagement, recency, frequency])
    scaler = MinMaxScaler()
    X = scaler.fit_transform(raw_features)

    feature_names = ['age', 'income', 'engagement', 'recency', 'frequency']

    true_uplift = (X[:, 0] * 0.1 + X[:, 2] * 0.15 + X[:, 1] * 0.05) - 0.05
    true_uplift = np.clip(true_uplift, -0.10, 0.30)

    treatment = np.random.binomial(1, 0.5, n_samples)

    p_outcome = np.where(treatment == 1, 0.15 + true_uplift, 0.15)
    p_outcome = np.clip(p_outcome, 0, 1)
    y = np.random.binomial(1, p_outcome)

    print(f"✓ X shape: {X.shape}")
    print(f"✓ Treatment split: {treatment.sum()} treated, {(1 - treatment).sum()} control")
    print(f"✓ Control conversion: {y[treatment == 0].mean():.1%}")
    print(f"✓ Treatment conversion: {y[treatment == 1].mean():.1%}")
    print(f"✓ Mean true uplift: {true_uplift.mean():.3f}")

    return X, treatment, y, feature_names, true_uplift
