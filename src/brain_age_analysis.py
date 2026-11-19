"""
Experiment 2: Brain-age gap analysis.
Investigates associations between EEG-predicted brain-age gap and behavioral symptoms.
"""

import numpy as np
import pandas as pd
import logging
import time
import os
from scipy.stats import pearsonr, spearmanr
from sklearn.linear_model import LinearRegression

logger = logging.getLogger(__name__)


def compute_brain_age_gap(actual_age, predicted_age):
    """Brain-age gap = predicted age - actual age."""
    return predicted_age - actual_age


def correct_age_bias(gap, actual_age):
    """
    Correct the well-known age bias in brain-age prediction.
    Young subjects tend to be over-predicted, old subjects under-predicted.
    Fits a linear model: gap ~ age, and returns residuals.
    """
    mask = np.isfinite(gap) & np.isfinite(actual_age)
    if mask.sum() < 10:
        return gap
    lr = LinearRegression()
    lr.fit(actual_age[mask].reshape(-1, 1), gap[mask])
    predicted_bias = lr.predict(actual_age.reshape(-1, 1))
    corrected = gap - predicted_bias
    return corrected


def correlation_with_ci(x, y, method="pearson", n_boot=1000, seed=42):
    """
    Compute correlation with 95% bootstrap confidence interval.
    """
    mask = np.isfinite(x) & np.isfinite(y)
    x, y = x[mask], y[mask]
    n = len(x)
    if n < 10:
        return {"r": np.nan, "p": np.nan, "ci_low": np.nan, "ci_high": np.nan, "n": n}

    if method == "pearson":
        r, p = pearsonr(x, y)
    else:
        r, p = spearmanr(x, y)

    rng = np.random.RandomState(seed)
    boot_rs = []
    for _ in range(n_boot):
        idx = rng.choice(n, n, replace=True)
        if method == "pearson":
            br, _ = pearsonr(x[idx], y[idx])
        else:
            br, _ = spearmanr(x[idx], y[idx])
        boot_rs.append(br)
    ci_low, ci_high = np.percentile(boot_rs, [2.5, 97.5])

    return {"r": r, "p": p, "ci_low": ci_low, "ci_high": ci_high, "n": n}


def run_brain_age_analysis(data, age_pred_results, results_dir):
    """
    Run the brain-age gap analysis.

    Args:
        data: dict from data_loader.build_dataset
        age_pred_results: dict from age_prediction.run_age_prediction
        results_dir: path to save results
    """
    logger.info("=" * 60)
    logger.info("EXPERIMENT 2: Brain-Age Gap Analysis")
    logger.info("=" * 60)
    start_time = time.time()

    pred_df = age_pred_results["predictions_df"]
    actual_age = pred_df["actual_age"].values
    predicted_age = pred_df["predicted_age_rf_kfold"].values

    # Compute gap
    valid = np.isfinite(predicted_age)
    gap = compute_brain_age_gap(actual_age, predicted_age)
    logger.info("Brain-age gap computed for %d subjects", valid.sum())
    logger.info("Gap: mean=%.3f, std=%.3f, range=[%.2f, %.2f]",
                np.nanmean(gap), np.nanstd(gap), np.nanmin(gap), np.nanmax(gap))

    # Age-bias analysis
    logger.info("\nAge-bias analysis...")
    r_bias, _ = pearsonr(actual_age[valid], gap[valid])
    logger.info("Gap-age correlation (before correction): r=%.4f", r_bias)

    gap_corrected = correct_age_bias(gap, actual_age)
    r_corrected, _ = pearsonr(actual_age[valid], gap_corrected[valid])
    logger.info("Gap-age correlation (after correction): r=%.4f", r_corrected)

    # Behavioral associations
    behavioral_targets = {
        "attention": data["targets"]["attention"],
        "internalizing": data["targets"]["internalizing"],
        "externalizing": data["targets"]["externalizing"],
        "p_factor": data["targets"]["p_factor"],
    }

    logger.info("\n--- Behavioral associations (corrected gap) ---")
    association_results = []

    for target_name, target_vals in behavioral_targets.items():
        # Uncorrected
        res_raw = correlation_with_ci(gap, target_vals)
        res_raw["target"] = target_name
        res_raw["gap_type"] = "uncorrected"

        # Corrected
        res_corr = correlation_with_ci(gap_corrected, target_vals)
        res_corr["target"] = target_name
        res_corr["gap_type"] = "age_corrected"

        association_results.append(res_raw)
        association_results.append(res_corr)

        logger.info(
            "  %s: raw r=%.4f (p=%.3e), corrected r=%.4f (p=%.3e, 95%%CI [%.4f, %.4f])",
            target_name, res_raw["r"], res_raw["p"],
            res_corr["r"], res_corr["p"], res_corr["ci_low"], res_corr["ci_high"],
        )

    # Bonferroni correction (4 tests)
    n_tests = len(behavioral_targets)
    for res in association_results:
        res["p_bonferroni"] = min(res["p"] * n_tests, 1.0) if np.isfinite(res["p"]) else np.nan

    # Sex differences in gap
    logger.info("\n--- Gap by sex ---")
    sex = data["sex"]
    for sex_val, sex_label in [(1, "Male"), (0, "Female")]:
        mask = (sex == sex_val) & valid
        g = gap_corrected[mask]
        logger.info("  %s: n=%d, mean gap=%.3f, std=%.3f", sex_label, mask.sum(), g.mean(), g.std())

    # Save results
    os.makedirs(os.path.join(results_dir, "predictions"), exist_ok=True)
    os.makedirs(os.path.join(results_dir, "tables"), exist_ok=True)
    os.makedirs(os.path.join(results_dir, "metrics"), exist_ok=True)

    # Save gap values
    gap_df = pd.DataFrame({
        "subject_id": data["subject_ids"],
        "actual_age": actual_age,
        "predicted_age": predicted_age,
        "brain_age_gap": gap,
        "brain_age_gap_corrected": gap_corrected,
        "sex": data["sex_labels"],
        "release": data["release"],
        "attention": data["targets"]["attention"],
        "internalizing": data["targets"]["internalizing"],
        "externalizing": data["targets"]["externalizing"],
        "p_factor": data["targets"]["p_factor"],
    })
    gap_df.to_csv(os.path.join(results_dir, "predictions", "brain_age_gap.csv"), index=False)

    # Save association table
    assoc_df = pd.DataFrame(association_results)
    assoc_df.to_csv(os.path.join(results_dir, "tables", "brain_age_associations.csv"), index=False)

    total_time = time.time() - start_time
    logger.info("\nExperiment 2 complete in %.1f seconds", total_time)

    return {
        "gap": gap,
        "gap_corrected": gap_corrected,
        "gap_df": gap_df,
        "association_results": assoc_df,
        "age_bias_r_before": r_bias,
        "age_bias_r_after": r_corrected,
        "total_time": total_time,
    }
