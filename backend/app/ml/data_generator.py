"""Synthetic campaign data generator for CausalIQ causal inference experiments."""

import numpy as np
from sklearn.preprocessing import MinMaxScaler


def generate_synthetic_campaign_data(
    n_samples: int = 10000, seed: int = 42
) -> tuple[np.ndarray, np.ndarray, np.ndarray, list[str], np.ndarray]:
    """Generate synthetic marketing campaign data with known heterogeneous treatment effects.

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

    p_base = 0.15
    p_outcome = np.where(treatment == 1, p_base + true_uplift, p_base)
    p_outcome = np.clip(p_outcome, 0, 1)
    y = np.random.binomial(1, p_outcome)

    print(f"✓ X shape: {X.shape}")
    print(f"✓ Treatment split: {treatment.sum()} treated, {(1 - treatment).sum()} control")
    print(f"✓ Control conversion: {y[treatment == 0].mean():.1%}")
    print(f"✓ Treatment conversion: {y[treatment == 1].mean():.1%}")
    print(f"✓ Mean true uplift: {true_uplift.mean():.3f}")

    return X, treatment, y, feature_names, true_uplift
