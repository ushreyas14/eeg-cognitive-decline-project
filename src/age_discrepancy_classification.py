"""
Experiment 6: EEG-Based Developmental Age-Discrepancy Classification.

Uses existing OOF age predictions and brain-age gap from Experiments 1-2.
No models are retrained here.

Three exploratory categories are defined from the corrected-gap distribution:
  - Lower EEG-Derived Age  : corrected gap < -threshold
  - Age-Consistent EEG     : -threshold <= corrected gap <= +threshold
  - Higher EEG-Derived Age : corrected gap > +threshold

Threshold is chosen data-driven (0.5 SD of the corrected gap), and sensitivity
analysis is run across a range of choices.
"""

import numpy as np
import pandas as pd
import logging
import time
import os
from scipy import stats

logger = logging.getLogger(__name__)

# Category labels (intentionally neutral — not clinical)
CAT_LOWER = "Lower EEG-Derived Age"
CAT_CONSISTENT = "Age-Consistent EEG"
CAT_HIGHER = "Higher EEG-Derived Age"
CAT_ORDER = [CAT_LOWER, CAT_CONSISTENT, CAT_HIGHER]

OUTCOMES = {
    "symbolSearch_accuracy": "Symbol Search Accuracy",
    "seqLearning_partial": "Sequential Learning (Partial)",
    "attention": "Attention Problems",
    "internalizing": "Internalizing",
    "externalizing": "Externalizing",
}


# ─── Threshold selection ────────────────────────────────────────────────────

def compute_category_thresholds(gap_corrected):
    """
    Derive thresholds from the corrected-gap distribution.

    Primary: ±0.5 SD  (splits ~38% middle, ~31% each tail)
    Secondary options tested in sensitivity analysis: ±0.33 SD, ±0.75 SD, ±1.0 SD

    Returns:
        dict with 'primary' (float) and 'alternatives' (list of float)
    """
    valid = gap_corrected[np.isfinite(gap_corrected)]
    sd = np.std(valid, ddof=1)
    mean = np.mean(valid)

    primary = 0.5 * sd
    alternatives = [0.33 * sd, 0.75 * sd, 1.0 * sd]

    logger.info("Corrected gap: mean=%.4f, SD=%.4f", mean, sd)
    logger.info("Primary threshold: ±%.4f years (±0.5 SD)", primary)
    logger.info(
        "Expected category splits at primary threshold: lower=%.1f%%, "
        "consistent=%.1f%%, higher=%.1f%%",
        100 * np.mean(valid < (mean - primary)),
        100 * np.mean((valid >= (mean - primary)) & (valid <= (mean + primary))),
        100 * np.mean(valid > (mean + primary)),
    )

    return {
        "primary": primary,
        "mean": mean,
        "sd": sd,
        "alternatives": alternatives,
    }


def assign_categories(gap_corrected, threshold_info):
    """
    Assign each subject to a category based on the corrected gap.
    Uses mean ± threshold (not 0 ± threshold, since the residualized
    gap is approximately zero-mean but let's use the actual empirical mean).
    """
    mean = threshold_info["mean"]
    thr = threshold_info["primary"]

    cats = np.full(len(gap_corrected), CAT_CONSISTENT, dtype=object)
    cats[gap_corrected < (mean - thr)] = CAT_LOWER
    cats[gap_corrected > (mean + thr)] = CAT_HIGHER
    return cats


# ─── Category statistics ────────────────────────────────────────────────────

def compute_category_stats(gap_df, threshold_info):
    """
    Counts, percentages, and demographic summaries per category.
    """
    gap_df = gap_df.copy()
    gap_df["category"] = assign_categories(
        gap_df["brain_age_gap_corrected"].values, threshold_info
    )

    rows = []
    n_total = len(gap_df)
    for cat in CAT_ORDER:
        sub = gap_df[gap_df["category"] == cat]
        n = len(sub)
        age = sub["actual_age"]
        pct_male = (sub["sex"] == "M").mean() * 100

        rows.append({
            "category": cat,
            "n": n,
            "pct_total": 100 * n / n_total,
            "gap_mean": sub["brain_age_gap_corrected"].mean(),
            "gap_sd": sub["brain_age_gap_corrected"].std(),
            "age_mean": age.mean(),
            "age_sd": age.std(),
            "age_min": age.min(),
            "age_max": age.max(),
            "pct_male": pct_male,
        })

    return pd.DataFrame(rows), gap_df


# ─── Between-category comparisons ───────────────────────────────────────────

def _cohens_d(a, b):
    """Cohen's d between two 1-D arrays (pooled SD)."""
    a = a[np.isfinite(a)]
    b = b[np.isfinite(b)]
    if len(a) < 2 or len(b) < 2:
        return np.nan
    pooled_sd = np.sqrt(((len(a) - 1) * np.var(a, ddof=1) +
                          (len(b) - 1) * np.var(b, ddof=1)) /
                         (len(a) + len(b) - 2))
    return (np.mean(a) - np.mean(b)) / (pooled_sd + 1e-12)


def _bootstrap_mean_diff(a, b, n_boot=1000, seed=42):
    """Bootstrap 95% CI for mean(a) - mean(b)."""
    a = a[np.isfinite(a)]
    b = b[np.isfinite(b)]
    rng = np.random.RandomState(seed)
    diffs = []
    for _ in range(n_boot):
        da = np.mean(rng.choice(a, len(a), replace=True))
        db = np.mean(rng.choice(b, len(b), replace=True))
        diffs.append(da - db)
    return np.percentile(diffs, [2.5, 97.5])


def ancova_comparison(outcome_vals, category_labels, age_vals, sex_vals, release_vals):
    """
    One-way ANCOVA: outcome ~ category + age + sex + release (OLS).
    Returns F, p, partial eta-squared for the category factor.
    Falls back to Kruskal-Wallis if fewer than 10 valid observations per group.
    """
    try:
        import statsmodels.formula.api as smf
        import statsmodels.stats.anova as sma

        df = pd.DataFrame({
            "y": outcome_vals,
            "cat": pd.Categorical(category_labels, categories=CAT_ORDER),
            "age": age_vals,
            "sex": sex_vals.astype(float),
            "release": release_vals,
        }).dropna(subset=["y"])

        # Need at least 3 observations per category
        counts = df["cat"].value_counts()
        if (counts < 3).any():
            raise ValueError("Insufficient observations per category for ANCOVA")

        # Encode release as numeric to avoid too many dummies blowing up with n_releases=11
        df["release_num"] = pd.factorize(df["release"])[0].astype(float)

        model = smf.ols("y ~ C(cat) + age + sex + release_num", data=df).fit()

        # Partial eta² for 'cat' factor
        anova_table = sma.anova_lm(model, typ=2)
        ss_cat = anova_table.loc["C(cat)", "sum_sq"]
        ss_res = anova_table.loc["Residual", "sum_sq"]
        eta2_partial = ss_cat / (ss_cat + ss_res)

        f_val = anova_table.loc["C(cat)", "F"]
        p_val = anova_table.loc["C(cat)", "PR(>F)"]

        return {
            "method": "ANCOVA (OLS, covariates: age, sex, release)",
            "F": float(f_val),
            "p": float(p_val),
            "eta2_partial": float(eta2_partial),
            "n": int(len(df)),
        }

    except Exception as e:
        logger.debug("ANCOVA failed (%s), falling back to Kruskal-Wallis", e)
        # Fallback: Kruskal-Wallis
        groups = []
        for cat in CAT_ORDER:
            mask = (category_labels == cat) & np.isfinite(outcome_vals)
            groups.append(outcome_vals[mask])
        groups = [g for g in groups if len(g) >= 3]
        if len(groups) < 2:
            return {"method": "insufficient_data", "F": np.nan, "p": np.nan,
                    "eta2_partial": np.nan, "n": 0}

        h, p_kw = stats.kruskal(*groups)
        # Epsilon² (non-parametric effect size) ≈ H / (n-1)
        n_total = sum(len(g) for g in groups)
        epsilon2 = h / (n_total - 1)
        return {
            "method": "Kruskal-Wallis (fallback)",
            "F": float(h),
            "p": float(p_kw),
            "eta2_partial": float(epsilon2),
            "n": int(n_total),
        }


def compare_across_categories(gap_df, data, threshold_info):
    """
    For each outcome variable, run ANCOVA (outcome ~ category + age + sex + release)
    and all pairwise post-hoc tests with Cohen's d + bootstrap CI.
    Applies Bonferroni correction over the number of outcomes tested.
    """
    gap_df = gap_df.copy()
    gap_df["category"] = assign_categories(
        gap_df["brain_age_gap_corrected"].values, threshold_info
    )

    # Merge cognitive targets onto gap_df
    targets = data["targets"]
    subject_ids = data["subject_ids"]

    target_df = pd.DataFrame({"subject_id": subject_ids})
    target_df["seqLearning_partial"] = targets.get("seqLearning_partial", np.full(len(subject_ids), np.nan))
    target_df["symbolSearch_accuracy"] = targets.get("symbolSearch_accuracy", np.full(len(subject_ids), np.nan))

    merged = gap_df.merge(target_df, on="subject_id", how="left")

    n_outcomes = len(OUTCOMES)
    omnibus_rows = []
    posthoc_rows = []

    pairs = [
        (CAT_LOWER, CAT_CONSISTENT),
        (CAT_LOWER, CAT_HIGHER),
        (CAT_CONSISTENT, CAT_HIGHER),
    ]

    for col, label in OUTCOMES.items():
        if col in merged.columns:
            outcome_vals = merged[col].values
        elif col in gap_df.columns:
            outcome_vals = gap_df[col].values
        else:
            logger.warning("Outcome column '%s' not found, skipping", col)
            continue

        cat_labels = merged["category"].values
        age_vals = merged["actual_age"].values
        sex_vals = merged["sex"].map({"M": 1, "F": 0}).fillna(0).values
        release_vals = merged["release"].values

        # Omnibus test
        result = ancova_comparison(outcome_vals, cat_labels, age_vals, sex_vals, release_vals)
        result["outcome"] = col
        result["outcome_label"] = label
        result["p_bonferroni"] = min(result["p"] * n_outcomes, 1.0) if np.isfinite(result["p"]) else np.nan
        omnibus_rows.append(result)

        # Pairwise comparisons
        cat_means = {}
        cat_ns = {}
        for cat in CAT_ORDER:
            mask = (cat_labels == cat) & np.isfinite(outcome_vals)
            cat_means[cat] = np.nanmean(outcome_vals[mask])
            cat_ns[cat] = int(mask.sum())

        for cat_a, cat_b in pairs:
            mask_a = (cat_labels == cat_a) & np.isfinite(outcome_vals)
            mask_b = (cat_labels == cat_b) & np.isfinite(outcome_vals)
            vals_a = outcome_vals[mask_a]
            vals_b = outcome_vals[mask_b]

            if len(vals_a) < 3 or len(vals_b) < 3:
                continue

            t_stat, p_t = stats.ttest_ind(vals_a, vals_b, equal_var=False)  # Welch
            ci_low, ci_high = _bootstrap_mean_diff(vals_a, vals_b)
            d = _cohens_d(vals_a, vals_b)

            # Bonferroni over number of pairs × outcomes
            n_comparisons = len(pairs) * n_outcomes
            p_bonf = min(p_t * n_comparisons, 1.0)

            posthoc_rows.append({
                "outcome": col,
                "outcome_label": label,
                "group_a": cat_a,
                "group_b": cat_b,
                "mean_a": float(np.mean(vals_a)),
                "mean_b": float(np.mean(vals_b)),
                "mean_diff": float(np.mean(vals_a) - np.mean(vals_b)),
                "ci_low": float(ci_low),
                "ci_high": float(ci_high),
                "cohens_d": float(d),
                "t_stat": float(t_stat),
                "p_welch": float(p_t),
                "p_bonferroni": float(p_bonf),
                "n_a": int(len(vals_a)),
                "n_b": int(len(vals_b)),
            })

    return pd.DataFrame(omnibus_rows), pd.DataFrame(posthoc_rows)


# ─── Sensitivity analysis ────────────────────────────────────────────────────

def sensitivity_analysis(gap_df, data, threshold_info):
    """
    Repeat the category comparison for each alternative threshold.
    Reports how many omnibus tests remain significant at p < 0.05 (uncorrected).
    """
    rows = []
    thresholds_to_test = [threshold_info["primary"]] + threshold_info["alternatives"]
    thr_labels = ["0.50 SD (primary)", "0.33 SD", "0.75 SD", "1.00 SD"]

    for thr, label in zip(thresholds_to_test, thr_labels):
        thr_info = {**threshold_info, "primary": thr}
        gap_copy = gap_df.copy()
        gap_copy["category"] = assign_categories(
            gap_copy["brain_age_gap_corrected"].values, thr_info
        )

        counts = gap_copy["category"].value_counts()
        omnibus, _ = compare_across_categories(gap_copy, data, thr_info)

        n_sig = int((omnibus["p"] < 0.05).sum()) if len(omnibus) > 0 else 0
        n_sig_bonf = int((omnibus["p_bonferroni"] < 0.05).sum()) if len(omnibus) > 0 else 0

        rows.append({
            "threshold_label": label,
            "threshold_years": float(thr),
            "threshold_sd_multiple": thr / threshold_info["sd"],
            "n_lower": int(counts.get(CAT_LOWER, 0)),
            "n_consistent": int(counts.get(CAT_CONSISTENT, 0)),
            "n_higher": int(counts.get(CAT_HIGHER, 0)),
            "pct_lower": 100 * counts.get(CAT_LOWER, 0) / len(gap_copy),
            "pct_consistent": 100 * counts.get(CAT_CONSISTENT, 0) / len(gap_copy),
            "pct_higher": 100 * counts.get(CAT_HIGHER, 0) / len(gap_copy),
            "n_outcomes_p05_raw": n_sig,
            "n_outcomes_p05_bonferroni": n_sig_bonf,
        })

    return pd.DataFrame(rows)


# ─── Main entry point ────────────────────────────────────────────────────────

def run_age_discrepancy_classification(data, exp2_results, results_dir):
    """
    Experiment 6: Classify subjects by EEG-age discrepancy and compare outcomes.

    Args:
        data:         dict from data_loader.build_dataset()
        exp2_results: dict from brain_age_analysis.run_brain_age_analysis()
        results_dir:  str, path to save results

    Returns:
        dict with all results DataFrames and metadata
    """
    logger.info("=" * 60)
    logger.info("EXPERIMENT 6: EEG-Based Age-Discrepancy Classification")
    logger.info("=" * 60)
    start_time = time.time()

    # 1. Obtain corrected gap values from Exp 2
    gap_df = exp2_results["gap_df"].copy()
    gap_corrected = gap_df["brain_age_gap_corrected"].values
    logger.info("Using OOF RF predictions from Experiment 1 (n=%d)", len(gap_df))

    # 2. Compute data-driven thresholds
    logger.info("\n--- Step 1: Threshold Selection ---")
    threshold_info = compute_category_thresholds(gap_corrected)

    # 3. Category stats
    logger.info("\n--- Step 2: Category Descriptives ---")
    category_stats, gap_df_cat = compute_category_stats(gap_df, threshold_info)
    for _, row in category_stats.iterrows():
        logger.info(
            "  %s: n=%d (%.1f%%), age=%.1f±%.1f yr, %.0f%% male, gap=%.3f±%.3f",
            row["category"], row["n"], row["pct_total"],
            row["age_mean"], row["age_sd"], row["pct_male"],
            row["gap_mean"], row["gap_sd"],
        )

    # 4. Outcome comparisons (ANCOVA + post-hoc)
    logger.info("\n--- Step 3: Outcome Comparisons (ANCOVA, age/sex/release-adjusted) ---")
    omnibus_df, posthoc_df = compare_across_categories(gap_df_cat, data, threshold_info)

    for _, row in omnibus_df.iterrows():
        sig = "*" if row["p_bonferroni"] < 0.05 else ""
        logger.info(
            "  %-38s F/H=%.3f  p=%.4f  p_bonf=%.4f  η²p=%.4f  n=%d  %s",
            row["outcome_label"], row["F"], row["p"],
            row["p_bonferroni"], row["eta2_partial"], row["n"], sig,
        )

    # 5. Sensitivity analysis
    logger.info("\n--- Step 4: Sensitivity Analysis ---")
    sensitivity_df = sensitivity_analysis(gap_df_cat, data, threshold_info)
    for _, row in sensitivity_df.iterrows():
        logger.info(
            "  Threshold %s (±%.3f yr): lower=%d  consistent=%d  higher=%d  "
            "sig_outcomes(raw)=%d  sig_outcomes(Bonf)=%d",
            row["threshold_label"], row["threshold_years"],
            row["n_lower"], row["n_consistent"], row["n_higher"],
            row["n_outcomes_p05_raw"], row["n_outcomes_p05_bonferroni"],
        )

    # 6. Save results
    logger.info("\n--- Step 5: Saving Results ---")
    for subdir in ["tables", "predictions", "metrics"]:
        os.makedirs(os.path.join(results_dir, subdir), exist_ok=True)

    # Category-level subject file
    gap_df_cat.to_csv(
        os.path.join(results_dir, "predictions", "age_discrepancy_categories.csv"),
        index=False,
    )

    # Threshold info
    thr_df = pd.DataFrame([{
        "threshold_primary_years": threshold_info["primary"],
        "gap_mean": threshold_info["mean"],
        "gap_sd": threshold_info["sd"],
        "lower_bound": threshold_info["mean"] - threshold_info["primary"],
        "upper_bound": threshold_info["mean"] + threshold_info["primary"],
    }])
    thr_df.to_csv(
        os.path.join(results_dir, "tables", "exp6_threshold_info.csv"), index=False
    )

    category_stats.to_csv(
        os.path.join(results_dir, "tables", "exp6_category_stats.csv"), index=False
    )

    omnibus_df.to_csv(
        os.path.join(results_dir, "tables", "exp6_omnibus_comparisons.csv"), index=False
    )

    posthoc_df.to_csv(
        os.path.join(results_dir, "tables", "exp6_posthoc_comparisons.csv"), index=False
    )

    sensitivity_df.to_csv(
        os.path.join(results_dir, "tables", "exp6_sensitivity_analysis.csv"), index=False
    )

    total_time = time.time() - start_time
    logger.info("\nExperiment 6 complete in %.1f seconds", total_time)

    return {
        "threshold_info": threshold_info,
        "category_stats": category_stats,
        "gap_df_cat": gap_df_cat,
        "omnibus_df": omnibus_df,
        "posthoc_df": posthoc_df,
        "sensitivity_df": sensitivity_df,
        "total_time": total_time,
    }
