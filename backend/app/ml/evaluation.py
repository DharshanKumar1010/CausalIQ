"""Week 3 evaluation suite for CausalIQ — AUUC, Qini, uplift curves, persuadable identification."""

import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def compute_uplift_curve(
    y: np.ndarray,
    treatment: np.ndarray,
    uplift_scores: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Compute cumulative uplift curve data points.

    Sorts customers by predicted uplift score descending, then at each
    percentile step measures the actual observed uplift within that top-k group.

    Args:
        y: Binary outcome array of shape (n_samples,).
        treatment: Binary treatment indicator array of shape (n_samples,).
        uplift_scores: Predicted uplift scores of shape (n_samples,).

    Returns:
        Tuple of (x_pct, cumulative_gains) where x_pct is 0–100 in 100 steps
        and cumulative_gains is the uplift gain at each percentile.
    """
    sorted_idx = np.argsort(uplift_scores)[::-1]
    y_sorted = y[sorted_idx]
    treatment_sorted = treatment[sorted_idx]
    n = len(y)

    x_pct = np.linspace(0, 100, 101)
    cumulative_gains = np.zeros(101)

    for i, pct in enumerate(x_pct):
        k = max(1, int(n * pct / 100))
        treated_in_group = treatment_sorted[:k] == 1
        control_in_group = treatment_sorted[:k] == 0

        if treated_in_group.sum() == 0 or control_in_group.sum() == 0:
            uplift = 0.0
        else:
            uplift = (
                y_sorted[:k][treated_in_group].mean()
                - y_sorted[:k][control_in_group].mean()
            )

        cumulative_gains[i] = uplift * (k / n)

    cumulative_gains[0] = 0.0
    return x_pct, cumulative_gains


def compute_auuc(
    y: np.ndarray,
    treatment: np.ndarray,
    uplift_scores: np.ndarray,
) -> float:
    """Compute Area Under the Uplift Curve (AUUC).

    Measures how much better the model's targeting is compared to random
    targeting. Higher is better; good models score > 0.0001 on Criteo data.

    Args:
        y: Binary outcome array of shape (n_samples,).
        treatment: Binary treatment indicator array of shape (n_samples,).
        uplift_scores: Predicted uplift scores of shape (n_samples,).

    Returns:
        AUUC as a float.
    """
    x_pct, cumulative_gains = compute_uplift_curve(y, treatment, uplift_scores)
    random_baseline = np.linspace(0, cumulative_gains[-1], len(cumulative_gains))
    auuc = np.trapezoid(cumulative_gains - random_baseline, x_pct / 100)
    return float(auuc)


def compute_qini_coefficient(
    y: np.ndarray,
    treatment: np.ndarray,
    uplift_scores: np.ndarray,
) -> float:
    """Compute the Qini coefficient.

    Ratio of the model's incremental area over random to the maximum possible
    area. Range: -1 to 1. Good models score > 0.1.

    Args:
        y: Binary outcome array of shape (n_samples,).
        treatment: Binary treatment indicator array of shape (n_samples,).
        uplift_scores: Predicted uplift scores of shape (n_samples,).

    Returns:
        Qini coefficient as a float.
    """
    x_pct, cumulative_gains = compute_uplift_curve(y, treatment, uplift_scores)
    random_baseline = np.linspace(0, cumulative_gains[-1], len(cumulative_gains))
    model_area = np.trapezoid(cumulative_gains, x_pct / 100)
    random_area = np.trapezoid(random_baseline, x_pct / 100)
    max_possible = cumulative_gains.max() * 0.5
    if max_possible == 0:
        return 0.0
    qini = (model_area - random_area) / (abs(max_possible) + 1e-10)
    return float(qini)


def identify_persuadables(
    X: np.ndarray,
    uplift_scores: np.ndarray,
    feature_names: list[str],
    top_pct: float = 10,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, pd.DataFrame]:
    """Identify top persuadable customers by predicted uplift score.

    Persuadables are customers with the highest predicted treatment effect —
    those most likely to convert if treated but not without treatment.

    Args:
        X: Normalized feature matrix of shape (n_samples, n_features).
        uplift_scores: Predicted uplift scores of shape (n_samples,).
        feature_names: List of feature name strings matching columns of X.
        top_pct: Percentage of the population to label as persuadable.

    Returns:
        Tuple of (persuadable_idx, persuadable_scores, persuadable_X, profile) where:
            persuadable_idx: Integer indices into the original arrays.
            persuadable_scores: Uplift scores for identified persuadables.
            persuadable_X: Feature rows for identified persuadables.
            profile: DataFrame comparing persuadable vs population feature means.
    """
    n_persuadables = int(len(X) * top_pct / 100)
    sorted_idx = np.argsort(uplift_scores)[::-1]
    persuadable_idx = sorted_idx[:n_persuadables]
    persuadable_scores = uplift_scores[persuadable_idx]
    persuadable_X = X[persuadable_idx]

    profile = pd.DataFrame({
        'feature': feature_names,
        'persuadable_mean': persuadable_X.mean(axis=0),
        'population_mean': X.mean(axis=0),
        'difference': persuadable_X.mean(axis=0) - X.mean(axis=0),
    })

    print(f"✓ Identified {n_persuadables:,} persuadables ({top_pct}% of {len(X):,})")
    print(f"  Mean uplift score: {persuadable_scores.mean():.6f}")
    print(f"  Min uplift score: {persuadable_scores.min():.6f}")
    print(profile.to_string(index=False))

    return persuadable_idx, persuadable_scores, persuadable_X, profile


def evaluate_all_models(
    predictions_dict: dict[str, np.ndarray],
    y_test: np.ndarray,
    treatment_test: np.ndarray,
    X_test: np.ndarray,
    feature_names: list[str],
) -> dict:
    """Run full AUUC, Qini, and persuadable evaluation across all models.

    Args:
        predictions_dict: Mapping of model name to uplift score array, as
            returned by UpliftEnsemble.predict_all().
        y_test: Binary outcome array for the test set.
        treatment_test: Binary treatment indicator array for the test set.
        X_test: Normalized feature matrix for the test set.
        feature_names: List of feature name strings matching columns of X_test.

    Returns:
        Dict with keys 'results' (per-model metrics and curve data), 'best_model'
        (name of the highest-AUUC model), 'persuadable_idx', 'persuadable_scores',
        and 'persuadable_profile'.
    """
    os.makedirs('backend/app/ml/artifacts/results', exist_ok=True)

    results: dict[str, dict] = {}
    for model_name, uplift_scores in predictions_dict.items():
        auuc = compute_auuc(y_test, treatment_test, uplift_scores)
        qini = compute_qini_coefficient(y_test, treatment_test, uplift_scores)
        x_pct, gains = compute_uplift_curve(y_test, treatment_test, uplift_scores)
        results[model_name] = {
            'auuc': auuc,
            'qini': qini,
            'x_pct': x_pct,
            'cumulative_gains': gains,
            'uplift_scores': uplift_scores,
        }

    best_model = max(results, key=lambda k: results[k]['auuc'])

    persuadable_idx, persuadable_scores, persuadable_X, profile = identify_persuadables(
        X_test, results[best_model]['uplift_scores'], feature_names
    )

    print("=" * 60)
    print("  CAUSALIQ Week 3: Evaluation Results")
    print("=" * 60)
    print(f"{'Model':<15} {'AUUC':>10} {'Qini':>10}")
    print("-" * 40)
    for name, res in results.items():
        marker = " <- BEST" if name == best_model else ""
        print(f"{name:<15} {res['auuc']:>10.6f} {res['qini']:>10.6f}{marker}")
    print("=" * 60)
    print(f"\nBest model: {best_model.upper()}")
    print(f"Persuadables identified: {len(persuadable_idx):,} customers")
    print("=" * 60)

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    ax1 = axes[0]
    for name, res in results.items():
        ax1.plot(res['x_pct'], res['cumulative_gains'], label=name)
    ax1.set_xlabel('% Population Targeted')
    ax1.set_ylabel('Cumulative Uplift Gain')
    ax1.set_title('Uplift Curves - CausalIQ (Criteo Dataset)')
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    ax2 = axes[1]
    model_names = list(results.keys())
    auuc_values = [results[m]['auuc'] for m in model_names]
    qini_values = [results[m]['qini'] for m in model_names]
    x_pos = np.arange(len(model_names))
    width = 0.35
    ax2.bar(x_pos - width / 2, auuc_values, width, label='AUUC', color='steelblue')
    ax2_twin = ax2.twinx()
    ax2_twin.bar(x_pos + width / 2, qini_values, width, label='Qini', color='darkorange', alpha=0.7)
    ax2.set_xticks(x_pos)
    ax2.set_xticklabels([m.replace('_', '\n') for m in model_names])
    ax2.set_ylabel('AUUC', color='steelblue')
    ax2_twin.set_ylabel('Qini', color='darkorange')
    ax2.set_title('AUUC and Qini by Model')

    plt.tight_layout()
    plt.savefig('backend/app/ml/artifacts/results/evaluation_curves.png', dpi=150, bbox_inches='tight')
    print("✓ Saved evaluation_curves.png")
    plt.close()

    return {
        'results': results,
        'best_model': best_model,
        'persuadable_idx': persuadable_idx,
        'persuadable_scores': persuadable_scores,
        'persuadable_profile': profile,
    }
