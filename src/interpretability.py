"""
Experiment 3: EEG feature importance and age-related analysis.
"""

import numpy as np
import pandas as pd
import logging
import time
import os
from sklearn.inspection import permutation_importance
from sklearn.impute import SimpleImputer
from scipy.stats import pearsonr

logger = logging.getLogger(__name__)


def get_feature_family(col_name):
    """Classify a feature column into its family."""
    if "alpha_coh" in col_name:
        return "Alpha Coherence"
    if "theta_alpha_ratio" in col_name:
        return "Theta/Alpha Ratio"
    if "spectral_entropy" in col_name:
        return "Spectral Entropy"
    if "sample_entropy" in col_name:
        return "Sample Entropy"
    prefix = col_name.split("_")[0]
    band_map = {"delta": "Delta Power", "theta": "Theta Power", "alpha": "Alpha Power",
                "beta": "Beta Power", "gamma": "Gamma Power"}
    return band_map.get(prefix, "Other")


def get_channel_number(col_name):
    """Extract channel number from feature name, or -1 if not applicable."""
    parts = col_name.split("_ch")
    if len(parts) == 2:
        try:
            return int(parts[1])
        except ValueError:
            pass
    # For coherence: alpha_coh_i_j
    if "alpha_coh" in col_name:
        parts = col_name.split("_")
        if len(parts) >= 4:
            try:
                return int(parts[2])  # first channel
            except ValueError:
                pass
    if col_name == "theta_alpha_ratio_mean":
        return -1
    return -1


def compute_permutation_importance(model, X, y, feature_names, n_repeats=10, random_state=42):
    """Compute permutation importance for the trained model."""
    logger.info("Computing permutation importance (%d repeats)...", n_repeats)
    imp = SimpleImputer(strategy="median")
    X_imp = imp.fit_transform(X)

    result = permutation_importance(
        model, X_imp, y, n_repeats=n_repeats, random_state=random_state, n_jobs=-1, scoring="r2",
    )

    imp_df = pd.DataFrame({
        "feature": feature_names,
        "importance_mean": result.importances_mean,
        "importance_std": result.importances_std,
        "family": [get_feature_family(f) for f in feature_names],
        "channel": [get_channel_number(f) for f in feature_names],
    }).sort_values("importance_mean", ascending=False)

    return imp_df


def compute_rf_importance(model, feature_names):
    """Extract Gini importance from a Random Forest model."""
    rf = model.named_steps.get("model") or model[-1]
    if not hasattr(rf, "feature_importances_"):
        logger.warning("Model does not have feature_importances_")
        return None

    # Features go through imputer first, so indices match
    imp = rf.feature_importances_
    imp_df = pd.DataFrame({
        "feature": feature_names,
        "gini_importance": imp,
        "family": [get_feature_family(f) for f in feature_names],
        "channel": [get_channel_number(f) for f in feature_names],
    }).sort_values("gini_importance", ascending=False)

    return imp_df


def compute_age_feature_correlations(X, age, feature_names):
    """Compute correlation of each EEG feature with age."""
    rows = []
    for i, fname in enumerate(feature_names):
        vals = X[:, i]
        mask = np.isfinite(vals) & np.isfinite(age)
        if mask.sum() < 20:
            rows.append({"feature": fname, "r": np.nan, "p": np.nan, "abs_r": np.nan})
            continue
        r, p = pearsonr(vals[mask], age[mask])
        rows.append({"feature": fname, "r": r, "p": p, "abs_r": abs(r),
                      "family": get_feature_family(fname)})
    return pd.DataFrame(rows).sort_values("abs_r", ascending=False)


def run_interpretability(data, age_pred_results, results_dir):
    """
    Run feature importance analysis.

    Args:
        data: dict from data_loader.build_dataset
        age_pred_results: dict from age_prediction
        results_dir: path to save results
    """
    logger.info("=" * 60)
    logger.info("EXPERIMENT 3: EEG Feature Importance & Age-Related Analysis")
    logger.info("=" * 60)
    start_time = time.time()

    X = data["X_eeg"]
    y = data["targets"]["age"]
    feature_names = data["feature_names"]
    final_model = age_pred_results["final_model"]

    # RF Gini importance
    logger.info("\n--- Random Forest Gini Importance ---")
    gini_df = compute_rf_importance(final_model, feature_names)
    if gini_df is not None:
        logger.info("Top 10 features by Gini importance:")
        for _, row in gini_df.head(10).iterrows():
            logger.info("  %s (%.5f) [%s]", row["feature"], row["gini_importance"], row["family"])

    # Permutation importance
    logger.info("\n--- Permutation Importance ---")
    perm_df = compute_permutation_importance(final_model, X, y, feature_names)
    logger.info("Top 10 features by permutation importance:")
    for _, row in perm_df.head(10).iterrows():
        logger.info("  %s: %.5f ± %.5f [%s]", row["feature"], row["importance_mean"], row["importance_std"], row["family"])

    # Age-feature correlations
    logger.info("\n--- Age-Feature Correlations ---")
    corr_df = compute_age_feature_correlations(X, y, feature_names)
    logger.info("Top 10 features by |correlation| with age:")
    for _, row in corr_df.head(10).iterrows():
        logger.info("  %s: r=%.4f [%s]", row["feature"], row["r"], row.get("family", ""))

    # Aggregate by family
    logger.info("\n--- Importance by Feature Family ---")
    if gini_df is not None:
        family_gini = gini_df.groupby("family")["gini_importance"].agg(["mean", "sum", "count"])
        family_gini = family_gini.sort_values("mean", ascending=False)
        logger.info("Gini importance by family:")
        for fam, row in family_gini.iterrows():
            logger.info("  %s: mean=%.5f, total=%.4f, n=%d", fam, row["mean"], row["sum"], int(row["count"]))

    family_perm = perm_df.groupby("family")["importance_mean"].agg(["mean", "sum", "count"])
    family_perm = family_perm.sort_values("mean", ascending=False)

    family_corr = corr_df.groupby("family")["abs_r"].agg(["mean", "max", "count"])
    family_corr = family_corr.sort_values("mean", ascending=False)

    # Save results
    os.makedirs(os.path.join(results_dir, "tables"), exist_ok=True)
    os.makedirs(os.path.join(results_dir, "metrics"), exist_ok=True)

    if gini_df is not None:
        gini_df.to_csv(os.path.join(results_dir, "tables", "feature_importance_gini.csv"), index=False)
    perm_df.to_csv(os.path.join(results_dir, "tables", "feature_importance_permutation.csv"), index=False)
    corr_df.to_csv(os.path.join(results_dir, "tables", "feature_age_correlations.csv"), index=False)
    family_perm.to_csv(os.path.join(results_dir, "tables", "importance_by_family_permutation.csv"))
    family_corr.to_csv(os.path.join(results_dir, "tables", "correlation_by_family.csv"))

    total_time = time.time() - start_time
    logger.info("\nExperiment 3 complete in %.1f seconds", total_time)

    return {
        "gini_importance": gini_df,
        "permutation_importance": perm_df,
        "age_correlations": corr_df,
        "family_perm": family_perm,
        "family_corr": family_corr,
        "total_time": total_time,
    }
