"""Causal inference uplift models for CausalIQ marketing campaign optimization."""

import numpy as np
from xgboost import XGBRegressor
from sklearn.model_selection import train_test_split

from app.ml.data_loader import generate_synthetic_campaign_data, load_criteo_data


class TLearnerModel:
    """T-Learner (Two-Model) uplift estimator.

    Trains separate outcome models for the control and treatment groups,
    then estimates individual treatment effects as the difference in predictions.
    """

    def __init__(self, base_model: XGBRegressor | None = None) -> None:
        """Initialize the T-Learner with independent XGBoost models.

        Args:
            base_model: Unused; models are always instantiated fresh to avoid
                shared state between control and treatment learners.
        """
        self.control_model = XGBRegressor(random_state=42, n_estimators=100, verbosity=0)
        self.treatment_model = XGBRegressor(random_state=42, n_estimators=100, verbosity=0)
        self.is_fitted = False

    def fit(self, X: np.ndarray, treatment: np.ndarray, y: np.ndarray) -> None:
        """Fit separate models on control and treatment subgroups.

        Args:
            X: Feature matrix of shape (n_samples, n_features).
            treatment: Binary treatment indicator array of shape (n_samples,).
            y: Binary outcome array of shape (n_samples,).
        """
        mask_control = treatment == 0
        mask_treatment = treatment == 1
        self.control_model.fit(X[mask_control], y[mask_control])
        self.treatment_model.fit(X[mask_treatment], y[mask_treatment])
        self.is_fitted = True
        print(
            f"✓ T-Learner fitted | Control: {mask_control.sum()} | Treatment: {mask_treatment.sum()}"
        )

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Predict individual treatment effects (uplift) for each sample.

        Args:
            X: Feature matrix of shape (n_samples, n_features).

        Returns:
            Uplift estimates of shape (n_samples,).
        """
        pred_control = self.control_model.predict(X)
        pred_treatment = self.treatment_model.predict(X)
        return pred_treatment - pred_control

    def get_feature_importance(self) -> dict[str, list[float]]:
        """Return feature importances from both sub-models.

        Returns:
            Dict with 'control' and 'treatment' keys mapping to importance lists.
        """
        return {
            'control': self.control_model.feature_importances_.tolist(),
            'treatment': self.treatment_model.feature_importances_.tolist(),
        }


class SLearnerModel:
    """S-Learner (Single-Model) uplift estimator.

    Trains one outcome model on all data with treatment as an additional feature,
    then estimates uplift by predicting counterfactuals for each individual.
    """

    def __init__(self, base_model: XGBRegressor | None = None) -> None:
        """Initialize the S-Learner with a single XGBoost model.

        Args:
            base_model: Unused; the model is always instantiated fresh.
        """
        self.model = XGBRegressor(random_state=42, n_estimators=100, verbosity=0)
        self.is_fitted = False

    def fit(self, X: np.ndarray, treatment: np.ndarray, y: np.ndarray) -> None:
        """Fit the model on the full dataset with treatment appended as a feature.

        Args:
            X: Feature matrix of shape (n_samples, n_features).
            treatment: Binary treatment indicator array of shape (n_samples,).
            y: Binary outcome array of shape (n_samples,).
        """
        treatment_col = treatment.reshape(-1, 1)
        X_aug = np.hstack([X, treatment_col])
        self.model.fit(X_aug, y)
        self.is_fitted = True
        print(f"✓ S-Learner fitted on {len(X)} samples")

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Predict individual treatment effects by differencing counterfactual outcomes.

        Args:
            X: Feature matrix of shape (n_samples, n_features).

        Returns:
            Uplift estimates of shape (n_samples,).
        """
        n = len(X)
        X_control = np.hstack([X, np.zeros((n, 1))])
        X_treatment = np.hstack([X, np.ones((n, 1))])
        pred_control = self.model.predict(X_control)
        pred_treatment = self.model.predict(X_treatment)
        return pred_treatment - pred_control

    def get_feature_importance(self) -> list[float]:
        """Return feature importances from the single model.

        Returns:
            List of feature importances (treatment column is the last entry).
        """
        return self.model.feature_importances_.tolist()


def compare_models_on_synthetic_data(
    n_samples: int = 10000, train_split: float = 0.7, seed: int = 42
) -> dict:
    """Train and compare T-Learner and S-Learner on synthetic campaign data.

    Args:
        n_samples: Total number of synthetic customer samples to generate.
        train_split: Fraction of data used for training (remainder is test).
        seed: Random seed for data generation and train/test split.

    Returns:
        Dict containing train/test arrays, model instances, uplift predictions,
        and summary statistics for both models.
    """
    X, treatment, y, feature_names, true_uplift = generate_synthetic_campaign_data(
        n_samples, seed
    )

    indices = np.arange(n_samples)
    train_idx, test_idx = train_test_split(
        indices, test_size=1 - train_split, random_state=seed
    )
    X_train, X_test = X[train_idx], X[test_idx]
    treatment_train, treatment_test = treatment[train_idx], treatment[test_idx]
    y_train, y_test = y[train_idx], y[test_idx]
    true_uplift_test = true_uplift[test_idx]

    t_model = TLearnerModel()
    t_model.fit(X_train, treatment_train, y_train)

    s_model = SLearnerModel()
    s_model.fit(X_train, treatment_train, y_train)

    t_predictions = t_model.predict(X_test)
    s_predictions = s_model.predict(X_test)

    correlation = np.corrcoef(t_predictions, s_predictions)[0, 1]

    print("=" * 55)
    print("  CAUSALIQ Week 1: Synthetic Data & Baseline Models")
    print("=" * 55)
    print(f"\nData Generation:")
    print(f"  ✓ Generated {n_samples:,} customers")
    print(f"  ✓ Train: {len(X_train):,} | Test: {len(X_test):,}")
    print(f"  ✓ Control conversion (train): {y_train[treatment_train == 0].mean():.1%}")
    print(f"  ✓ Treatment conversion (train): {y_train[treatment_train == 1].mean():.1%}")
    print(f"\nT-Learner:")
    print(f"  ✓ Mean uplift (test): {t_predictions.mean():.4f}")
    print(f"  ✓ Uplift range: [{t_predictions.min():.4f}, {t_predictions.max():.4f}]")
    print(f"\nS-Learner:")
    print(f"  ✓ Mean uplift (test): {s_predictions.mean():.4f}")
    print(f"  ✓ Uplift range: [{s_predictions.min():.4f}, {s_predictions.max():.4f}]")
    print(f"\nModel Comparison:")
    print(f"  T-Learner mean uplift: {t_predictions.mean():.4f}")
    print(f"  S-Learner mean uplift: {s_predictions.mean():.4f}")
    print(f"  Prediction correlation: {correlation:.4f}")
    print("=" * 55)

    return {
        'X_train': X_train,
        'X_test': X_test,
        'treatment_train': treatment_train,
        'treatment_test': treatment_test,
        'y_train': y_train,
        'y_test': y_test,
        'feature_names': feature_names,
        'true_uplift_test': true_uplift_test,
        't_learner_predictions': t_predictions,
        's_learner_predictions': s_predictions,
        't_learner_mean_uplift': float(t_predictions.mean()),
        's_learner_mean_uplift': float(s_predictions.mean()),
        'correlation_t_s': float(correlation),
        't_model': t_model,
        's_model': s_model,
    }


def compare_models_on_criteo(
    filepath: str = 'data/processed/criteo_sample.csv',
    train_split: float = 0.7,
    seed: int = 42,
) -> dict:
    """Train and compare T-Learner and S-Learner on the real Criteo Uplift dataset.

    Args:
        filepath: Path to the processed Criteo CSV file.
        train_split: Fraction of data used for training (remainder is test).
        seed: Random seed for data loading and train/test split.

    Returns:
        Dict containing train/test arrays, model instances, uplift predictions,
        and summary statistics for both models evaluated on Criteo data.
    """
    X, treatment, y, feature_names = load_criteo_data(filepath, seed=seed)

    indices = np.arange(len(X))
    train_idx, test_idx = train_test_split(
        indices, test_size=1 - train_split, random_state=seed
    )
    X_train, X_test = X[train_idx], X[test_idx]
    treatment_train, treatment_test = treatment[train_idx], treatment[test_idx]
    y_train, y_test = y[train_idx], y[test_idx]

    t_model = TLearnerModel()
    t_model.fit(X_train, treatment_train, y_train)

    s_model = SLearnerModel()
    s_model.fit(X_train, treatment_train, y_train)

    t_predictions = t_model.predict(X_test)
    s_predictions = s_model.predict(X_test)

    correlation = np.corrcoef(t_predictions, s_predictions)[0, 1]

    print("=" * 60)
    print("  CAUSALIQ: Criteo Ad-Tech Dataset Results")
    print("=" * 60)
    print(f"Dataset: Criteo Uplift v2.1 (real ad-tech RCT data)")
    print(f"Sample: 500,000 rows | Features: 12 anonymized ad features")
    print(f"Train: {len(X_train):,} | Test: {len(X_test):,}")
    print(f"Treatment ratio: {treatment_train.mean():.1%}")
    print(f"Conversion rate: {y_train.mean():.3%}")
    print(f"\nT-Learner:")
    print(f"  Mean uplift (test): {t_predictions.mean():.6f}")
    print(f"  Uplift range: [{t_predictions.min():.6f}, {t_predictions.max():.6f}]")
    print(f"\nS-Learner:")
    print(f"  Mean uplift (test): {s_predictions.mean():.6f}")
    print(f"  Uplift range: [{s_predictions.min():.6f}, {s_predictions.max():.6f}]")
    print(f"\nModel Comparison:")
    print(f"  T-Learner mean uplift: {t_predictions.mean():.6f}")
    print(f"  S-Learner mean uplift: {s_predictions.mean():.6f}")
    print(f"  Prediction correlation: {correlation:.4f}")
    print("=" * 60)

    return {
        'X_train': X_train,
        'X_test': X_test,
        'treatment_train': treatment_train,
        'treatment_test': treatment_test,
        'y_train': y_train,
        'y_test': y_test,
        'feature_names': feature_names,
        't_learner_predictions': t_predictions,
        's_learner_predictions': s_predictions,
        't_learner_mean_uplift': float(t_predictions.mean()),
        's_learner_mean_uplift': float(s_predictions.mean()),
        'correlation_t_s': float(correlation),
        't_model': t_model,
        's_model': s_model,
    }


class XLearnerModel:
    """X-Learner (Cross-Model) uplift estimator.

    Fits outcome models on each group, computes residual pseudo-outcomes
    (cross-group imputations), then trains effect models on those residuals.
    Better than T-Learner when treatment/control group sizes are imbalanced.
    """

    def __init__(self) -> None:
        """Initialize the X-Learner with four independent XGBoost models."""
        self.control_outcome_model = XGBRegressor(random_state=42, n_estimators=100, verbosity=0)
        self.treatment_outcome_model = XGBRegressor(random_state=42, n_estimators=100, verbosity=0)
        self.control_effect_model = XGBRegressor(random_state=42, n_estimators=100, verbosity=0)
        self.treatment_effect_model = XGBRegressor(random_state=42, n_estimators=100, verbosity=0)
        self.is_fitted = False

    def fit(self, X: np.ndarray, treatment: np.ndarray, y: np.ndarray) -> None:
        """Fit outcome models then cross-impute pseudo-treatment effects.

        Args:
            X: Feature matrix of shape (n_samples, n_features).
            treatment: Binary treatment indicator array of shape (n_samples,).
            y: Binary outcome array of shape (n_samples,).
        """
        mask_c = treatment == 0
        mask_t = treatment == 1

        self.control_outcome_model.fit(X[mask_c], y[mask_c])
        self.treatment_outcome_model.fit(X[mask_t], y[mask_t])

        tau_t = y[mask_t] - self.control_outcome_model.predict(X[mask_t])
        tau_c = self.treatment_outcome_model.predict(X[mask_c]) - y[mask_c]

        self.treatment_effect_model.fit(X[mask_t], tau_t)
        self.control_effect_model.fit(X[mask_c], tau_c)

        self.is_fitted = True
        print(f"✓ X-Learner fitted | Control: {mask_c.sum()} | Treatment: {mask_t.sum()}")

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Predict uplift as a propensity-weighted blend of the two effect models.

        Args:
            X: Feature matrix of shape (n_samples, n_features).

        Returns:
            Uplift estimates of shape (n_samples,).
        """
        g = 0.5
        tau_t = self.treatment_effect_model.predict(X)
        tau_c = self.control_effect_model.predict(X)
        return g * tau_c + (1 - g) * tau_t

    def get_feature_importance(self) -> dict[str, list[float]]:
        """Return feature importances from both effect models.

        Returns:
            Dict with 'control_effect' and 'treatment_effect' keys.
        """
        return {
            'control_effect': self.control_effect_model.feature_importances_.tolist(),
            'treatment_effect': self.treatment_effect_model.feature_importances_.tolist(),
        }


class DRLearnerModel:
    """DR-Learner (Doubly-Robust) uplift estimator.

    Constructs doubly-robust pseudo-outcomes using separate outcome models and
    a fixed propensity score, then regresses those pseudo-outcomes on features.
    Consistent if either the outcome model or propensity model is correctly specified.
    """

    def __init__(self) -> None:
        """Initialize the DR-Learner with outcome models and a fixed propensity."""
        self.outcome_model_c = XGBRegressor(random_state=42, n_estimators=100, verbosity=0)
        self.outcome_model_t = XGBRegressor(random_state=42, n_estimators=100, verbosity=0)
        self.effect_model = XGBRegressor(random_state=42, n_estimators=100, verbosity=0)
        self.propensity = 0.5
        self.is_fitted = False

    def fit(self, X: np.ndarray, treatment: np.ndarray, y: np.ndarray) -> None:
        """Build DR pseudo-outcomes and fit the final effect model.

        Args:
            X: Feature matrix of shape (n_samples, n_features).
            treatment: Binary treatment indicator array of shape (n_samples,).
            y: Binary outcome array of shape (n_samples,).
        """
        mask_c = treatment == 0
        mask_t = treatment == 1

        self.outcome_model_c.fit(X[mask_c], y[mask_c])
        self.outcome_model_t.fit(X[mask_t], y[mask_t])

        mu0 = self.outcome_model_c.predict(X)
        mu1 = self.outcome_model_t.predict(X)
        p = self.propensity

        dr_outcome = (
            mu1 - mu0
            + (treatment * (y - mu1)) / p
            - ((1 - treatment) * (y - mu0)) / (1 - p)
        )

        self.effect_model.fit(X, dr_outcome)
        self.is_fitted = True
        print(f"✓ DR-Learner fitted on {len(X)} samples")

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Predict uplift using the fitted effect model.

        Args:
            X: Feature matrix of shape (n_samples, n_features).

        Returns:
            Uplift estimates of shape (n_samples,).
        """
        return self.effect_model.predict(X)

    def get_feature_importance(self) -> list[float]:
        """Return feature importances from the effect model.

        Returns:
            List of feature importances.
        """
        return self.effect_model.feature_importances_.tolist()


class UpliftEnsemble:
    """Ensemble of all four meta-learner uplift models.

    Holds T-, S-, X-, and DR-Learner instances and provides joint
    training, per-model prediction, and simple mean-ensemble prediction.
    """

    def __init__(self) -> None:
        """Initialize the ensemble with one instance of each meta-learner."""
        self.models: dict[str, TLearnerModel | SLearnerModel | XLearnerModel | DRLearnerModel] = {
            't_learner': TLearnerModel(),
            's_learner': SLearnerModel(),
            'x_learner': XLearnerModel(),
            'dr_learner': DRLearnerModel(),
        }
        self.is_fitted = False

    def fit_all(self, X: np.ndarray, treatment: np.ndarray, y: np.ndarray) -> None:
        """Fit every model in the ensemble sequentially.

        Args:
            X: Feature matrix of shape (n_samples, n_features).
            treatment: Binary treatment indicator array of shape (n_samples,).
            y: Binary outcome array of shape (n_samples,).
        """
        for name, model in self.models.items():
            print(f"Training {name}...")
            model.fit(X, treatment, y)
        self.is_fitted = True
        print("✓ All 4 models fitted")

    def predict_all(self, X: np.ndarray) -> dict[str, np.ndarray]:
        """Return per-model uplift predictions.

        Args:
            X: Feature matrix of shape (n_samples, n_features).

        Returns:
            Dict mapping model name to uplift array of shape (n_samples,).
        """
        return {name: model.predict(X) for name, model in self.models.items()}

    def ensemble_predict(self, X: np.ndarray) -> np.ndarray:
        """Return the element-wise mean uplift across all four models.

        Args:
            X: Feature matrix of shape (n_samples, n_features).

        Returns:
            Ensemble uplift estimates of shape (n_samples,).
        """
        all_preds = self.predict_all(X)
        return np.mean(list(all_preds.values()), axis=0)


def compare_all_models_on_criteo(
    filepath: str = 'data/processed/criteo_sample.csv',
    train_split: float = 0.7,
    seed: int = 42,
) -> dict:
    """Train all four meta-learners and the ensemble on the Criteo dataset.

    Args:
        filepath: Path to the processed Criteo CSV file.
        train_split: Fraction of data used for training (remainder is test).
        seed: Random seed for data loading and train/test split.

    Returns:
        Dict with per-model predictions, ensemble predictions, test arrays,
        and the fitted UpliftEnsemble instance.
    """
    X, treatment, y, feature_names = load_criteo_data(filepath, seed=seed)

    indices = np.arange(len(X))
    train_idx, test_idx = train_test_split(
        indices, test_size=1 - train_split, random_state=seed
    )
    X_train, X_test = X[train_idx], X[test_idx]
    treatment_train, treatment_test = treatment[train_idx], treatment[test_idx]
    y_train, y_test = y[train_idx], y[test_idx]

    ensemble = UpliftEnsemble()
    ensemble.fit_all(X_train, treatment_train, y_train)

    predictions = ensemble.predict_all(X_test)
    ensemble_preds = ensemble.ensemble_predict(X_test)

    print("=" * 60)
    print("  CAUSALIQ Week 2: All 4 Meta-Learners on Criteo")
    print("=" * 60)
    print(f"Train: {len(X_train):,} | Test: {len(X_test):,}")
    for name, preds in predictions.items():
        print(f"\n{name.upper()}:")
        print(f"  Mean uplift: {preds.mean():.6f}")
        print(f"  Std: {preds.std():.6f}")
        print(f"  Range: [{preds.min():.6f}, {preds.max():.6f}]")
    print(f"\nENSEMBLE (mean of 4 models):")
    print(f"  Mean uplift: {ensemble_preds.mean():.6f}")
    print(f"  Std: {ensemble_preds.std():.6f}")
    print("=" * 60)

    return {
        'predictions': predictions,
        'ensemble_predictions': ensemble_preds,
        'X_test': X_test,
        'treatment_test': treatment_test,
        'y_test': y_test,
        'ensemble': ensemble,
    }
