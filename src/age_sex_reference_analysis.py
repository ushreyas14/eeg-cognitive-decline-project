"""
Experiment 6: Age- and Sex-Specific Brain-Age Reference Analysis.

Builds a reference distribution of brain-age gaps stratified by age group and sex,
then computes leave-one-out (LOO) reference statistics for each participant so that
no one is compared against a distribution that includes their own observation.

Pipeline:
  1. Bin participants into age groups (where n ≥ MIN_GROUP_N per age-sex cell).
  2. Compute LOO reference mean and SD for each participant.
  3. Compute reference-adjusted gap = raw_gap − loo_mean.
  4. Assign exploratory categories via a data-driven threshold (±0.5 SD of the
     reference-adjusted gap distribution).
  5. Compare cognitive/behavioural outcomes across categories with age-/sex-/
     release-adjusted ANCOVA and Bonferroni correction.
  6. Sensitivity analysis over four threshold choices.

Important constraints:
  - No models are retrained; Exp 1 OOF predictions are used throughout.
  - Categories are exploratory and descriptive, NOT clinical diagnoses.
  - Sparse groups (n < MIN_GROUP_N) are flagged; their z-scores/percentiles are
    marked unreliable so they are never presented as reference standards.
  - This is cross-sectional; individual trajectories cannot be inferred.
"""

import numpy as np
import pandas as pd
import logging
import time
import os
from scipy import stats

logger = logging.getLogger(__name__)

# ─── Constants ───────────────────────────────────────────────────────────────

AGE_BINS = [5, 8, 10, 12, 14, 17, 23]
AGE_LABELS = ["5-7", "8-9", "10-11", "12-13", "14-16", "17-22"]
MIN_GROUP_N = 15   # minimum group size for a reliable reference

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


# ─── Step 1: Age-sex reference distributions ─────────────────────────────────

def bin_age(age_array):
    """Map continuous ages to age-group labels."""
    return pd.cut(
        age_array,
        bins=AGE_BINS,
        labels=AGE_LABELS,
        right=False,
    ).astype(str)


def compute_age_sex_reference(gap_df):
    """
    Compute reference statistics (mean, median, SD, percentiles) for each
    age-sex group using all available subjects.

    Returns a DataFrame with one row per (age_bin, sex) cell, flagged
    'reliable' when n >= MIN_GROUP_N.
    """
    df = gap_df.copy()
    df["age_bin"] = bin_age(df["actual_age"].values)

    rows = []
    for (age_bin, sex), grp in df.groupby(["age_bin", "sex"]):
        raw_gaps = grp["brain_age_gap"].dropna().values
        n = len(raw_gaps)
        rows.append({
            "age_bin": age_bin,
            "sex": sex,
            "n": n,
            "reliable": n >= MIN_GROUP_N,
            "mean_raw": np.mean(raw_gaps) if n > 0 else np.nan,
            "median_raw": np.median(raw_gaps) if n > 0 else np.nan,
            "sd_raw": np.std(raw_gaps, ddof=1) if n > 1 else np.nan,
            "p10": np.percentile(raw_gaps, 10) if n >= MIN_GROUP_N else np.nan,
            "p25": np.percentile(raw_gaps, 25) if n >= MIN_GROUP_N else np.nan,
            "p75": np.percentile(raw_gaps, 75) if n >= MIN_GROUP_N else np.nan,
            "p90": np.percentile(raw_gaps, 90) if n >= MIN_GROUP_N else np.nan,
        })

    ref_df = pd.DataFrame(rows)
    return ref_df


# ─── Step 2: Leave-one-out individual reference stats ────────────────────────

def compute_loo_reference_stats(gap_df, ref_stats_df):
    """
    For each participant, compute LOO reference statistics:
      - loo_ref_mean  : mean of their age-sex group excluding themselves
      - loo_ref_sd    : SD of their age-sex group excluding themselves
      - ref_adjusted_gap : raw_gap − loo_ref_mean
      - z_score       : (raw_gap − loo_ref_mean) / loo_ref_sd
      - percentile    : rank-based percentile within group (excluding self)
      - reliable      : whether the remaining group has n ≥ MIN_GROUP_N

    Groups with insufficient n get NaN for z_score and percentile but
    still get ref_adjusted_gap (useful for classification even when
    uncertainty is high).
    """
    df = gap_df.copy()
    df["age_bin"] = bin_age(df["actual_age"].values)

    # Pre-compute sum and sum-of-squares per (age_bin, sex) for fast LOO
    group_stats = {}
    for (age_bin, sex), grp in df.groupby(["age_bin", "sex"]):
        raw = grp["brain_age_gap"].values
        valid = raw[np.isfinite(raw)]
        group_stats[(age_bin, sex)] = {
            "n": len(valid),
            "sum": valid.sum(),
            "sum_sq": (valid ** 2).sum(),
            "values": valid,
            "index": grp.index.tolist(),
        }

    loo_records = []
    for idx, row in df.iterrows():
        ab = row["age_bin"]
        sx = row["sex"]
        gap_i = row["brain_age_gap"]
        key = (ab, sx)

        gs = group_stats.get(key, None)
        if gs is None or not np.isfinite(gap_i):
            loo_records.append({
                "index": idx,
                "loo_ref_mean": np.nan,
                "loo_ref_sd": np.nan,
                "ref_adjusted_gap": gap_i if np.isfinite(gap_i) else np.nan,
                "z_score": np.nan,
                "percentile_within_group": np.nan,
                "group_n": 0,
                "reliable": False,
            })
            continue

        n_grp = gs["n"]
        # LOO: remove this subject's contribution
        # LOO mean = (sum - gap_i) / (n - 1)
        if n_grp < 2 or not np.isfinite(gap_i):
            loo_mean = np.nan
            loo_sd = np.nan
        else:
            loo_sum = gs["sum"] - gap_i
            loo_n = n_grp - 1
            loo_mean = loo_sum / loo_n
            # Welford-style LOO variance
            loo_sum_sq = gs["sum_sq"] - gap_i ** 2
            if loo_n > 1:
                loo_var = (loo_sum_sq - (loo_sum ** 2) / loo_n) / (loo_n - 1)
                loo_sd = np.sqrt(max(loo_var, 0.0))
            else:
                loo_sd = np.nan

        ref_adj = gap_i - loo_mean if np.isfinite(loo_mean) else np.nan

        # Z-score (reliable only when LOO group n ≥ MIN_GROUP_N)
        loo_n_remaining = n_grp - 1
        reliable = loo_n_remaining >= MIN_GROUP_N

        if reliable and np.isfinite(loo_sd) and loo_sd > 0:
            z = (gap_i - loo_mean) / loo_sd
        else:
            z = np.nan

        # Percentile: fraction of LOO group below this value
        if reliable:
            others = gs["values"]
            others = others[others != gap_i] if gap_i in others else others
            pct = float(stats.percentileofscore(others, gap_i, kind="rank"))
        else:
            pct = np.nan

        loo_records.append({
            "index": idx,
            "loo_ref_mean": float(loo_mean) if np.isfinite(loo_mean) else np.nan,
            "loo_ref_sd": float(loo_sd) if np.isfinite(loo_sd) else np.nan,
            "ref_adjusted_gap": float(ref_adj) if np.isfinite(ref_adj) else np.nan,
            "z_score": float(z) if np.isfinite(z) else np.nan,
            "percentile_within_group": float(pct) if np.isfinite(pct) else np.nan,
            "group_n": int(n_grp),
            "reliable": reliable,
        })

    loo_df = pd.DataFrame(loo_records).set_index("index")
    df = df.join(loo_df, how="left")
    return df


# ─── Step 3: Threshold and category assignment ───────────────────────────────

def compute_reference_thresholds(ref_adjusted_gap):
    """
    Derive category thresholds from the reference-adjusted gap distribution.

    Uses the SD of participants in reliable groups only (those where the LOO
    reference was computable and trustworthy).  Primary threshold = ±0.5 SD.
    """
    valid = ref_adjusted_gap[np.isfinite(ref_adjusted_gap)]
    sd = float(np.std(valid, ddof=1))
    mean = float(np.mean(valid))
    primary = 0.5 * sd

    logger.info(
        "Reference-adjusted gap: n=%d, mean=%.4f, SD=%.4f, threshold=±%.4f yr (±0.5 SD)",
        len(valid), mean, sd, primary,
    )
    logger.info(
        "Expected splits — lower=%.1f%%, consistent=%.1f%%, higher=%.1f%%",
        100 * np.mean(valid < (mean - primary)),
        100 * np.mean((valid >= (mean - primary)) & (valid <= (mean + primary))),
        100 * np.mean(valid > (mean + primary)),
    )
    return {"primary": primary, "mean": mean, "sd": sd,
            "alternatives": [0.33 * sd, 0.75 * sd, 1.0 * sd]}


def assign_categories(ref_adjusted_gap, threshold_info):
    """Assign neutral category labels based on reference-adjusted gap."""
    mean = threshold_info["mean"]
    thr = threshold_info["primary"]
    cats = np.full(len(ref_adjusted_gap), CAT_CONSISTENT, dtype=object)
    finite = np.isfinite(ref_adjusted_gap)
    cats[finite & (ref_adjusted_gap < (mean - thr))] = CAT_LOWER
    cats[finite & (ref_adjusted_gap > (mean + thr))] = CAT_HIGHER
    cats[~finite] = "Unknown"
    return cats


# ─── Step 4: Category statistics ─────────────────────────────────────────────

def compute_category_stats(gap_df_ref):
    """Counts, percentages, and demographics per category."""
    rows = []
    n_total = len(gap_df_ref)
    for cat in CAT_ORDER:
        sub = gap_df_ref[gap_df_ref["category"] == cat]
        n = len(sub)
        rows.append({
            "category": cat,
            "n": n,
            "pct_total": 100 * n / n_total,
            "gap_mean": sub["brain_age_gap"].mean(),
            "gap_sd": sub["brain_age_gap"].std(),
            "ref_adj_mean": sub["ref_adjusted_gap"].mean(),
            "ref_adj_sd": sub["ref_adjusted_gap"].std(),
            "age_mean": sub["actual_age"].mean(),
            "age_sd": sub["actual_age"].std(),
            "pct_male": (sub["sex"] == "M").mean() * 100,
        })
    return pd.DataFrame(rows)


# ─── Step 5: Outcome comparisons (ANCOVA) ────────────────────────────────────

def _cohens_d(a, b):
    a = a[np.isfinite(a)]; b = b[np.isfinite(b)]
    if len(a) < 2 or len(b) < 2:
        return np.nan
    pooled = np.sqrt(((len(a)-1)*np.var(a, ddof=1) + (len(b)-1)*np.var(b, ddof=1)) /
                     (len(a)+len(b)-2))
    return (np.mean(a) - np.mean(b)) / (pooled + 1e-12)


def _bootstrap_mean_diff(a, b, n_boot=1000, seed=42):
    a = a[np.isfinite(a)]; b = b[np.isfinite(b)]
    rng = np.random.RandomState(seed)
    diffs = [np.mean(rng.choice(a, len(a), True)) - np.mean(rng.choice(b, len(b), True))
             for _ in range(n_boot)]
    return np.percentile(diffs, [2.5, 97.5])


def ancova_comparison(outcome_vals, cat_labels, age_vals, sex_vals, release_vals):
    """OLS ANCOVA: outcome ~ category + age + sex + release_num."""
    try:
        import statsmodels.formula.api as smf
        import statsmodels.stats.anova as sma

        df = pd.DataFrame({
            "y": outcome_vals, "cat": pd.Categorical(cat_labels, categories=CAT_ORDER),
            "age": age_vals, "sex": sex_vals.astype(float),
            "release_num": pd.factorize(release_vals)[0].astype(float),
        }).dropna(subset=["y"])

        counts = df["cat"].value_counts()
        if (counts < 3).any():
            raise ValueError("Group too small for ANCOVA")

        model = smf.ols("y ~ C(cat) + age + sex + release_num", data=df).fit()
        tbl = sma.anova_lm(model, typ=2)
        ss_cat = tbl.loc["C(cat)", "sum_sq"]
        ss_res = tbl.loc["Residual", "sum_sq"]
        return {
            "method": "ANCOVA",
            "F": float(tbl.loc["C(cat)", "F"]),
            "p": float(tbl.loc["C(cat)", "PR(>F)"]),
            "eta2_partial": float(ss_cat / (ss_cat + ss_res)),
            "n": int(len(df)),
        }
    except Exception as e:
        logger.debug("ANCOVA failed (%s), using Kruskal-Wallis", e)
        groups = [outcome_vals[(cat_labels == c) & np.isfinite(outcome_vals)]
                  for c in CAT_ORDER]
        groups = [g for g in groups if len(g) >= 3]
        if len(groups) < 2:
            return {"method": "insufficient", "F": np.nan, "p": np.nan,
                    "eta2_partial": np.nan, "n": 0}
        h, p = stats.kruskal(*groups)
        n_total = sum(len(g) for g in groups)
        return {"method": "Kruskal-Wallis", "F": float(h), "p": float(p),
                "eta2_partial": float(h / (n_total - 1)), "n": int(n_total)}


def compare_across_categories(gap_df_ref, data):
    """ANCOVA + pairwise Welch t-tests for each outcome."""
    # Merge cognitive targets
    target_df = pd.DataFrame({"subject_id": data["subject_ids"],
                               "seqLearning_partial": data["targets"].get("seqLearning_partial"),
                               "symbolSearch_accuracy": data["targets"].get("symbolSearch_accuracy")})
    merged = gap_df_ref.merge(target_df, on="subject_id", how="left")

    n_outcomes = len(OUTCOMES)
    omnibus_rows, posthoc_rows = [], []
    pairs = [(CAT_LOWER, CAT_CONSISTENT), (CAT_LOWER, CAT_HIGHER), (CAT_CONSISTENT, CAT_HIGHER)]

    for col, label in OUTCOMES.items():
        outcome_vals = merged[col].values if col in merged.columns else \
                       gap_df_ref[col].values if col in gap_df_ref.columns else \
                       np.full(len(merged), np.nan)

        cats = merged["category"].values
        age_v = merged["actual_age"].values
        sex_v = merged["sex"].map({"M": 1, "F": 0}).fillna(0).values
        rel_v = merged["release"].values

        result = ancova_comparison(outcome_vals, cats, age_v, sex_v, rel_v)
        result.update({"outcome": col, "outcome_label": label,
                        "p_bonferroni": min(result["p"] * n_outcomes, 1.0)
                        if np.isfinite(result["p"]) else np.nan})
        omnibus_rows.append(result)

        for cat_a, cat_b in pairs:
            va = outcome_vals[(cats == cat_a) & np.isfinite(outcome_vals)]
            vb = outcome_vals[(cats == cat_b) & np.isfinite(outcome_vals)]
            if len(va) < 3 or len(vb) < 3:
                continue
            t, p_t = stats.ttest_ind(va, vb, equal_var=False)
            ci = _bootstrap_mean_diff(va, vb)
            d = _cohens_d(va, vb)
            posthoc_rows.append({
                "outcome": col, "outcome_label": label,
                "group_a": cat_a, "group_b": cat_b,
                "mean_a": float(np.mean(va)), "mean_b": float(np.mean(vb)),
                "mean_diff": float(np.mean(va) - np.mean(vb)),
                "ci_low": float(ci[0]), "ci_high": float(ci[1]),
                "cohens_d": float(d),
                "t_stat": float(t), "p_welch": float(p_t),
                "p_bonferroni": min(float(p_t) * len(pairs) * n_outcomes, 1.0),
                "n_a": int(len(va)), "n_b": int(len(vb)),
            })

    return pd.DataFrame(omnibus_rows), pd.DataFrame(posthoc_rows)


# ─── Step 6: Sensitivity analysis ────────────────────────────────────────────

def sensitivity_analysis(gap_df_ref, data, threshold_info):
    """Repeat comparisons at four threshold choices."""
    thresholds = [threshold_info["primary"]] + threshold_info["alternatives"]
    labels = ["0.50 SD (primary)", "0.33 SD", "0.75 SD", "1.00 SD"]
    rows = []
    for thr, label in zip(thresholds, labels):
        thr_info = {**threshold_info, "primary": thr}
        df = gap_df_ref.copy()
        df["category"] = assign_categories(df["ref_adjusted_gap"].values, thr_info)
        counts = df["category"].value_counts()
        omnibus, _ = compare_across_categories(df, data)
        rows.append({
            "threshold_label": label,
            "threshold_years": float(thr),
            "threshold_sd_multiple": thr / threshold_info["sd"],
            "n_lower": int(counts.get(CAT_LOWER, 0)),
            "n_consistent": int(counts.get(CAT_CONSISTENT, 0)),
            "n_higher": int(counts.get(CAT_HIGHER, 0)),
            "pct_lower": 100 * counts.get(CAT_LOWER, 0) / len(df),
            "pct_consistent": 100 * counts.get(CAT_CONSISTENT, 0) / len(df),
            "pct_higher": 100 * counts.get(CAT_HIGHER, 0) / len(df),
            "n_sig_raw": int((omnibus["p"] < 0.05).sum()) if len(omnibus) else 0,
            "n_sig_bonferroni": int((omnibus["p_bonferroni"] < 0.05).sum()) if len(omnibus) else 0,
        })
    return pd.DataFrame(rows)


# ─── Main entry point ─────────────────────────────────────────────────────────

def run_age_sex_reference_analysis(data, exp2_results, results_dir):
    """
    Experiment 6: Age- and Sex-Specific Brain-Age Reference Analysis.

    Uses OOF RF predictions from Exp 1 (via Exp 2's gap_df).
    No models are retrained.
    """
    logger.info("=" * 60)
    logger.info("EXPERIMENT 6: Age- and Sex-Specific Brain-Age Reference Analysis")
    logger.info("=" * 60)
    start = time.time()

    gap_df = exp2_results["gap_df"].copy()
    logger.info("Using OOF predictions: n=%d", len(gap_df))

    # 1. Population-level age-sex reference
    logger.info("\n--- Step 1: Age-Sex Reference Distributions ---")
    ref_stats = compute_age_sex_reference(gap_df)
    for _, r in ref_stats.iterrows():
        flag = "✓" if r["reliable"] else "⚠ sparse"
        logger.info("  %s / %s: n=%d %s  mean=%.3f  SD=%.3f",
                    r["age_bin"], r["sex"], r["n"], flag, r["mean_raw"], r["sd_raw"])

    # 2. LOO individual reference stats
    logger.info("\n--- Step 2: LOO Individual Reference Statistics ---")
    gap_df_ref = compute_loo_reference_stats(gap_df, ref_stats)
    n_reliable = gap_df_ref["reliable"].sum()
    logger.info("  Reliable LOO reference available for %d/%d subjects (%.1f%%)",
                n_reliable, len(gap_df_ref), 100 * n_reliable / len(gap_df_ref))

    # 3. Thresholds from reference-adjusted gap
    logger.info("\n--- Step 3: Threshold Selection ---")
    ref_adj = gap_df_ref["ref_adjusted_gap"].values
    threshold_info = compute_reference_thresholds(ref_adj)

    # 4. Assign categories
    gap_df_ref["category"] = assign_categories(ref_adj, threshold_info)
    cat_stats = compute_category_stats(gap_df_ref)
    logger.info("\n--- Step 4: Category Counts ---")
    for _, r in cat_stats.iterrows():
        logger.info("  %s: n=%d (%.1f%%)  age=%.1f±%.1f yr  %.0f%% male",
                    r["category"], r["n"], r["pct_total"],
                    r["age_mean"], r["age_sd"], r["pct_male"])

    # 5. Outcome comparisons
    logger.info("\n--- Step 5: Outcome Comparisons (ANCOVA) ---")
    omnibus_df, posthoc_df = compare_across_categories(gap_df_ref, data)
    for _, r in omnibus_df.iterrows():
        sig = "*" if r["p_bonferroni"] < 0.05 else ""
        logger.info("  %-38s F=%.3f p=%.4f p_bonf=%.4f η²p=%.4f %s",
                    r["outcome_label"], r["F"], r["p"], r["p_bonferroni"],
                    r["eta2_partial"], sig)

    # 6. Sensitivity analysis
    logger.info("\n--- Step 6: Sensitivity Analysis ---")
    sens_df = sensitivity_analysis(gap_df_ref, data, threshold_info)
    for _, r in sens_df.iterrows():
        logger.info("  %s: lower=%d consistent=%d higher=%d  sig(Bonf)=%d",
                    r["threshold_label"], r["n_lower"],
                    r["n_consistent"], r["n_higher"], r["n_sig_bonferroni"])

    # 7. Save
    logger.info("\n--- Step 7: Saving Results ---")
    for d in ["tables", "predictions", "metrics"]:
        os.makedirs(os.path.join(results_dir, d), exist_ok=True)

    gap_df_ref.to_csv(
        os.path.join(results_dir, "predictions", "exp6_individual_reference.csv"), index=False)
    ref_stats.to_csv(
        os.path.join(results_dir, "tables", "exp6_age_sex_reference_stats.csv"), index=False)
    cat_stats.to_csv(
        os.path.join(results_dir, "tables", "exp6_category_stats.csv"), index=False)
    pd.DataFrame([{
        "threshold_primary_years": threshold_info["primary"],
        "ref_adj_gap_mean": threshold_info["mean"],
        "ref_adj_gap_sd": threshold_info["sd"],
        "lower_bound": threshold_info["mean"] - threshold_info["primary"],
        "upper_bound": threshold_info["mean"] + threshold_info["primary"],
    }]).to_csv(os.path.join(results_dir, "tables", "exp6_threshold_info.csv"), index=False)
    omnibus_df.to_csv(
        os.path.join(results_dir, "tables", "exp6_omnibus_comparisons.csv"), index=False)
    posthoc_df.to_csv(
        os.path.join(results_dir, "tables", "exp6_posthoc_comparisons.csv"), index=False)
    sens_df.to_csv(
        os.path.join(results_dir, "tables", "exp6_sensitivity_analysis.csv"), index=False)

    total_time = time.time() - start
    logger.info("Experiment 6 complete in %.1f seconds", total_time)

    return {
        "ref_stats": ref_stats,
        "gap_df_ref": gap_df_ref,
        "threshold_info": threshold_info,
        "category_stats": cat_stats,
        "omnibus_df": omnibus_df,
        "posthoc_df": posthoc_df,
        "sensitivity_df": sens_df,
        "total_time": total_time,
    }
