"""
Experiment 5: Developmental analysis.
Examines how EEG features and cognitive performance vary with age,
and whether brain-age gap relates to cognitive outcomes after controlling for age.
"""

import numpy as np
import pandas as pd
import logging
import time
import os
from scipy.stats import pearsonr, spearmanr
from sklearn.linear_model import LinearRegression

logger = logging.getLogger(__name__)


AGE_BINS = [5, 8, 10, 12, 14, 17, 23]
AGE_LABELS = ["5-7", "8-9", "10-11", "12-13", "14-16", "17-22"]


def age_bin(ages):
    return pd.cut(ages, bins=AGE_BINS, labels=AGE_LABELS, right=False)


def compute_age_trends(X, age, feature_names, feature_info):
    """
    Compute developmental trends: how each EEG feature family changes with age.
    Returns per-family summary with mean correlation, plus per-feature details.
    """
    logger.info("Computing age trends for %d features...", len(feature_names))

    rows = []
    for i, fname in enumerate(feature_names):
        vals = X[:, i]
        mask = np.isfinite(vals) & np.isfinite(age)
        if mask.sum() < 20:
            continue
        r, p = pearsonr(vals[mask], age[mask])
        rows.append({"feature": fname, "r_with_age": r, "p": p, "abs_r": abs(r)})

    detail = pd.DataFrame(rows)

    family_map = {}
    for family_key, cols in feature_info.items():
        if family_key == "all":
            continue
        for c in cols:
            family_map[c] = family_key

    detail["family"] = detail["feature"].map(family_map).fillna("other")

    family_summary = detail.groupby("family").agg(
        mean_r=("r_with_age", "mean"),
        mean_abs_r=("abs_r", "mean"),
        max_abs_r=("abs_r", "max"),
        n_features=("feature", "count"),
        n_sig=("p", lambda x: (x < 0.05).sum()),
    ).sort_values("mean_abs_r", ascending=False)

    logger.info("\nAge-trend summary by feature family:")
    for fam, row in family_summary.iterrows():
        logger.info("  %s: mean|r|=%.4f, max|r|=%.4f, %d/%d significant",
                     fam, row["mean_abs_r"], row["max_abs_r"],
                     int(row["n_sig"]), int(row["n_features"]))

    return detail, family_summary


def compute_cognitive_age_curves(targets, age, sex_labels):
    """
    Compute how cognitive measures vary across age groups.
    """
    logger.info("Computing cognitive-age curves...")

    age_groups = age_bin(age)
    results = []

    cog_targets = {
        "symbolSearch_accuracy": targets["symbolSearch_accuracy"],
        "seqLearning_partial": targets["seqLearning_partial"],
    }

    for target_name, vals in cog_targets.items():
        for grp in AGE_LABELS:
            mask = (age_groups == grp) & np.isfinite(vals)
            n = mask.sum()
            if n < 5:
                continue
            v = vals[mask]
            results.append({
                "target": target_name,
                "age_group": grp,
                "n": n,
                "mean": np.mean(v),
                "std": np.std(v),
                "median": np.median(v),
                "q25": np.percentile(v, 25),
                "q75": np.percentile(v, 75),
            })

        # Also by sex
        for sex_val in ["M", "F"]:
            sex_mask = (sex_labels == sex_val) & np.isfinite(vals)
            if sex_mask.sum() < 10:
                continue
            r, p = pearsonr(age[sex_mask], vals[sex_mask])
            results.append({
                "target": target_name,
                "age_group": f"all_{sex_val}",
                "n": sex_mask.sum(),
                "mean": np.mean(vals[sex_mask]),
                "std": np.std(vals[sex_mask]),
                "median": np.median(vals[sex_mask]),
                "q25": np.percentile(vals[sex_mask], 25),
                "q75": np.percentile(vals[sex_mask], 75),
                "r_with_age": r,
                "p_age": p,
            })

    return pd.DataFrame(results)


def gap_cognitive_association(gap_corrected, targets, age, n_boot=1000, seed=42):
    """
    Test whether age-bias-corrected brain-age gap predicts cognitive performance
    after controlling for chronological age.

    Uses partial correlation: regress both gap and cognitive target on age,
    then correlate residuals.
    """
    logger.info("Testing gap-cognitive associations (partial correlations)...")
    rng = np.random.RandomState(seed)

    cog_targets = {
        "symbolSearch_accuracy": targets["symbolSearch_accuracy"],
        "seqLearning_partial": targets["seqLearning_partial"],
    }

    results = []
    for target_name, vals in cog_targets.items():
        mask = np.isfinite(gap_corrected) & np.isfinite(vals) & np.isfinite(age)
        n = mask.sum()
        if n < 20:
            logger.warning("  %s: too few subjects (%d)", target_name, n)
            results.append({
                "target": target_name, "n": n,
                "partial_r": np.nan, "partial_p": np.nan,
                "raw_r": np.nan, "raw_p": np.nan,
            })
            continue

        g, v, a = gap_corrected[mask], vals[mask], age[mask]

        # Raw correlation
        raw_r, raw_p = pearsonr(g, v)

        # Partial correlation: residualize both on age
        lr = LinearRegression()
        lr.fit(a.reshape(-1, 1), g)
        gap_resid = g - lr.predict(a.reshape(-1, 1))

        lr2 = LinearRegression()
        lr2.fit(a.reshape(-1, 1), v)
        cog_resid = v - lr2.predict(a.reshape(-1, 1))

        partial_r, partial_p = pearsonr(gap_resid, cog_resid)

        # Bootstrap CI
        boot_rs = []
        for _ in range(n_boot):
            idx = rng.choice(n, n, replace=True)
            try:
                br, _ = pearsonr(gap_resid[idx], cog_resid[idx])
                boot_rs.append(br)
            except Exception:
                pass
        ci_low, ci_high = np.percentile(boot_rs, [2.5, 97.5]) if boot_rs else (np.nan, np.nan)

        results.append({
            "target": target_name,
            "n": n,
            "raw_r": raw_r,
            "raw_p": raw_p,
            "partial_r": partial_r,
            "partial_p": partial_p,
            "partial_ci_low": ci_low,
            "partial_ci_high": ci_high,
        })

        logger.info("  %s (n=%d): raw r=%.4f (p=%.3e), partial r=%.4f (p=%.3e, 95%%CI [%.4f, %.4f])",
                     target_name, n, raw_r, raw_p, partial_r, partial_p, ci_low, ci_high)

    return pd.DataFrame(results)


def compute_behavioral_age_curves(targets, age):
    """Compute behavioral symptom means across age groups."""
    logger.info("Computing behavioral-age curves...")
    age_groups = age_bin(age)

    behav_targets = {
        "attention": targets["attention"],
        "internalizing": targets["internalizing"],
        "externalizing": targets["externalizing"],
        "p_factor": targets["p_factor"],
    }

    rows = []
    for name, vals in behav_targets.items():
        for grp in AGE_LABELS:
            mask = (age_groups == grp) & np.isfinite(vals)
            n = mask.sum()
            if n < 5:
                continue
            v = vals[mask]
            rows.append({
                "target": name, "age_group": grp, "n": n,
                "mean": np.mean(v), "std": np.std(v),
                "median": np.median(v),
            })

        # Overall age correlation
        mask = np.isfinite(vals) & np.isfinite(age)
        if mask.sum() >= 20:
            r, p = pearsonr(age[mask], vals[mask])
            rows.append({
                "target": name, "age_group": "overall_corr",
                "n": mask.sum(), "mean": np.nan, "std": np.nan, "median": np.nan,
                "r_with_age": r, "p_age": p,
            })

    return pd.DataFrame(rows)


def run_developmental_analysis(data, brain_age_results, results_dir):
    """
    Run Experiment 5: developmental analysis.
    """
    logger.info("=" * 60)
    logger.info("EXPERIMENT 5: Developmental Analysis")
    logger.info("=" * 60)
    start_time = time.time()

    X = data["X_eeg"]
    age = data["targets"]["age"]
    feature_names = data["feature_names"]
    feature_info = data["feature_info"]
    targets = data["targets"]
    sex_labels = data["sex_labels"]

    os.makedirs(os.path.join(results_dir, "tables"), exist_ok=True)
    os.makedirs(os.path.join(results_dir, "metrics"), exist_ok=True)

    # 5A: EEG feature age trends
    logger.info("\n--- 5A: EEG Feature Age Trends ---")
    age_detail, family_summary = compute_age_trends(X, age, feature_names, feature_info)
    age_detail.to_csv(os.path.join(results_dir, "tables", "eeg_age_trends_detail.csv"), index=False)
    family_summary.to_csv(os.path.join(results_dir, "tables", "eeg_age_trends_by_family.csv"))

    # 5B: Cognitive-age curves
    logger.info("\n--- 5B: Cognitive Performance Across Age Groups ---")
    cog_curves = compute_cognitive_age_curves(targets, age, sex_labels)
    cog_curves.to_csv(os.path.join(results_dir, "tables", "cognitive_age_curves.csv"), index=False)
    logger.info("Cognitive-age curves computed for %d rows", len(cog_curves))

    # 5C: Behavioral-age curves
    logger.info("\n--- 5C: Behavioral Symptoms Across Age Groups ---")
    behav_curves = compute_behavioral_age_curves(targets, age)
    behav_curves.to_csv(os.path.join(results_dir, "tables", "behavioral_age_curves.csv"), index=False)

    # 5D: Gap-cognitive association (partial correlation)
    gap_cog = pd.DataFrame()
    if brain_age_results is not None and "gap_corrected" in brain_age_results:
        logger.info("\n--- 5D: Brain-Age Gap and Cognitive Performance ---")
        gap_cog = gap_cognitive_association(
            brain_age_results["gap_corrected"], targets, age,
        )
        gap_cog.to_csv(os.path.join(results_dir, "tables", "gap_cognitive_partial_correlations.csv"), index=False)
    else:
        logger.warning("Brain-age gap results not available; skipping gap-cognitive analysis")

    total_time = time.time() - start_time
    logger.info("\nExperiment 5 complete in %.1f seconds", total_time)

    return {
        "age_trends_detail": age_detail,
        "age_trends_family": family_summary,
        "cognitive_age_curves": cog_curves,
        "behavioral_age_curves": behav_curves,
        "gap_cognitive": gap_cog,
        "total_time": total_time,
    }
