"""
Visualization module: generates publication-quality figures for all experiments.
Saves all figures as PNG files to results/figures/.
"""

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import logging
import os

logger = logging.getLogger(__name__)

FIGSIZE_WIDE = (10, 6)
FIGSIZE_SQUARE = (7, 7)
FIGSIZE_TALL = (8, 10)
DPI = 200

FAMILY_COLORS = {
    "Delta Power": "#2563eb",
    "Theta Power": "#7c3aed",
    "Alpha Power": "#0891b2",
    "Beta Power": "#059669",
    "Gamma Power": "#d97706",
    "Theta/Alpha Ratio": "#dc2626",
    "Sample Entropy": "#4f46e5",
    "Spectral Entropy": "#0d9488",
    "Alpha Coherence": "#ca8a04",
    "Other": "#6b7280",
}

MODEL_COLORS = {
    "Random Forest": "#2563eb",
    "Ridge Regression": "#d97706",
    "Mean Baseline": "#d1d5db",
}

AGE_GROUP_ORDER = ["5-7", "8-9", "10-11", "12-13", "14-16", "17-22"]


def _style_ax(ax):
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.tick_params(labelsize=10)


def _save(fig, path):
    fig.savefig(path, dpi=DPI, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    logger.info("Saved figure: %s", os.path.basename(path))


def plot_age_prediction_scatter(pred_df, results_dir):
    """Scatter: predicted vs actual age for RF KFold."""
    fig, ax = plt.subplots(figsize=FIGSIZE_SQUARE)
    _style_ax(ax)
    actual = pred_df["actual_age"]
    predicted = pred_df["predicted_age_rf_kfold"]
    valid = predicted.notna()

    ax.scatter(actual[valid], predicted[valid], alpha=0.25, s=14,
               c="#2563eb", edgecolors="none", zorder=2)

    lims = [actual.min() - 1, actual.max() + 1]
    ax.plot(lims, lims, "--", color="#9ca3af", linewidth=1.2, label="Perfect prediction", zorder=1)

    from scipy.stats import pearsonr as _pr
    z = np.polyfit(actual[valid], predicted[valid], 1)
    xs = np.linspace(lims[0], lims[1], 100)
    ax.plot(xs, np.polyval(z, xs), "-", color="#dc2626", linewidth=1.5, alpha=0.7, label="Best fit", zorder=3)

    ax.set_xlabel("Actual Age (years)", fontsize=12)
    ax.set_ylabel("Predicted Age (years)", fontsize=12)
    ax.set_title("EEG-Based Age Prediction (Random Forest, 10-Fold CV)", fontsize=13, fontweight="bold")
    ax.set_xlim(lims)
    ax.set_ylim(lims)
    ax.set_aspect("equal")
    ax.legend(fontsize=10, loc="lower right")

    from sklearn.metrics import r2_score, mean_absolute_error
    r2 = r2_score(actual[valid], predicted[valid])
    mae = mean_absolute_error(actual[valid], predicted[valid])
    ax.text(0.05, 0.95, f"R² = {r2:.3f}\nMAE = {mae:.2f} yr\nn = {valid.sum()}",
            transform=ax.transAxes, fontsize=12, verticalalignment="top", fontweight="bold",
            bbox=dict(boxstyle="round,pad=0.4", facecolor="#f0f4ff", edgecolor="#2563eb", alpha=0.9))

    _save(fig, os.path.join(results_dir, "figures", "age_prediction_scatter.png"))


def plot_age_prediction_by_group(age_group_metrics, results_dir):
    """Bar chart of MAE by age group, properly ordered."""
    df = pd.DataFrame(age_group_metrics)
    df["age_group"] = pd.Categorical(df["age_group"], categories=AGE_GROUP_ORDER, ordered=True)
    df = df.sort_values("age_group")

    fig, ax = plt.subplots(figsize=FIGSIZE_WIDE)
    _style_ax(ax)

    cmap = plt.cm.Blues
    norm_vals = (df["MAE"] - df["MAE"].min()) / (df["MAE"].max() - df["MAE"].min() + 1e-9)
    colors = [cmap(0.35 + 0.55 * v) for v in norm_vals]

    bars = ax.bar(range(len(df)), df["MAE"], color=colors, edgecolor="white", linewidth=1.5)
    ax.set_xticks(range(len(df)))
    ax.set_xticklabels(df["age_group"], fontsize=11)

    for bar, (_, row) in zip(bars, df.iterrows()):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.08,
                f"{row['MAE']:.2f}\n(n={int(row['n'])})", ha="center", va="bottom", fontsize=9)

    ax.set_xlabel("Age Group (years)", fontsize=12)
    ax.set_ylabel("MAE (years)", fontsize=12)
    ax.set_title("Age Prediction Error by Age Group (RF, 10-Fold CV)", fontsize=13, fontweight="bold")
    ax.set_ylim(0, df["MAE"].max() * 1.25)
    _save(fig, os.path.join(results_dir, "figures", "age_prediction_by_age_group.png"))


def plot_model_comparison(summary_df, results_dir):
    """Grouped bar: R² for each model and CV type."""
    kfold = summary_df[summary_df["cv_type"] == "kfold"].copy()
    kfold = kfold.sort_values("R2", ascending=True)

    fig, ax = plt.subplots(figsize=(10, 4.5))
    _style_ax(ax)

    colors = [MODEL_COLORS.get(m, "#6b7280") for m in kfold["model"]]
    bars = ax.barh(range(len(kfold)), kfold["R2"], color=colors, edgecolor="white", linewidth=1.5, height=0.6)
    ax.set_yticks(range(len(kfold)))
    ax.set_yticklabels(kfold["model"], fontsize=12, fontweight="bold")

    for bar, r2 in zip(bars, kfold["R2"]):
        x_pos = max(bar.get_width(), 0) + 0.012
        ax.text(x_pos, bar.get_y() + bar.get_height() / 2,
                f"R² = {r2:.3f}", va="center", fontsize=12, fontweight="bold")

    ax.set_xlabel("R² (10-Fold Cross-Validation)", fontsize=12)
    ax.set_title("Age Prediction: Model Comparison", fontsize=13, fontweight="bold")
    ax.set_xlim([-0.05, max(kfold["R2"]) * 1.25])
    ax.axvline(0, color="#d1d5db", linewidth=0.8, zorder=0)
    _save(fig, os.path.join(results_dir, "figures", "model_comparison.png"))


def plot_brain_age_gap_distribution(gap_df, results_dir):
    """Histogram of corrected brain-age gap."""
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))

    for ax in axes:
        _style_ax(ax)

    g = gap_df["brain_age_gap"].dropna()
    axes[0].hist(g, bins=40, color="#2563eb", alpha=0.75, edgecolor="white", linewidth=0.8)
    axes[0].axvline(0, color="#dc2626", linestyle="--", linewidth=1.2)
    axes[0].axvline(g.mean(), color="#f59e0b", linestyle="-", linewidth=1.2, label=f"Mean = {g.mean():.2f}")
    axes[0].set_xlabel("Brain-Age Gap (years)", fontsize=11)
    axes[0].set_ylabel("Count", fontsize=11)
    axes[0].set_title("Uncorrected Brain-Age Gap", fontsize=12, fontweight="bold")
    axes[0].legend(fontsize=9)

    gc = gap_df["brain_age_gap_corrected"].dropna()
    axes[1].hist(gc, bins=40, color="#059669", alpha=0.75, edgecolor="white", linewidth=0.8)
    axes[1].axvline(0, color="#dc2626", linestyle="--", linewidth=1.2)
    axes[1].axvline(gc.mean(), color="#f59e0b", linestyle="-", linewidth=1.2, label=f"Mean = {gc.mean():.2f}")
    axes[1].set_xlabel("Corrected Brain-Age Gap (years)", fontsize=11)
    axes[1].set_ylabel("Count", fontsize=11)
    axes[1].set_title("Age-Bias Corrected Brain-Age Gap", fontsize=12, fontweight="bold")
    axes[1].legend(fontsize=9)

    fig.suptitle("Brain-Age Gap Distributions (n = {})".format(len(g)), fontsize=14, fontweight="bold")
    fig.tight_layout(rect=[0, 0, 1, 0.94])
    _save(fig, os.path.join(results_dir, "figures", "brain_age_gap_distribution.png"))


def plot_gap_vs_age(gap_df, results_dir):
    """Scatter: gap vs actual age, before and after correction."""
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))

    from scipy.stats import pearsonr
    for ax, col, title, color in [
        (axes[0], "brain_age_gap", "Before Correction", "#2563eb"),
        (axes[1], "brain_age_gap_corrected", "After Correction", "#059669"),
    ]:
        _style_ax(ax)
        valid = gap_df[col].notna()
        x = gap_df.loc[valid, "actual_age"]
        y = gap_df.loc[valid, col]
        ax.scatter(x, y, alpha=0.15, s=8, c=color, edgecolors="none")
        ax.axhline(0, color="#dc2626", linestyle="--", linewidth=1)

        r, p = pearsonr(x, y)
        ax.text(0.05, 0.95, f"r = {r:.3f}", transform=ax.transAxes, fontsize=11,
                va="top", fontweight="bold",
                bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.8))

        ax.set_xlabel("Actual Age (years)", fontsize=11)
        ax.set_ylabel("Brain-Age Gap (years)", fontsize=11)
        ax.set_title(title, fontsize=12, fontweight="bold")

    fig.suptitle("Brain-Age Gap vs. Chronological Age", fontsize=14, fontweight="bold")
    fig.tight_layout(rect=[0, 0, 1, 0.94])
    _save(fig, os.path.join(results_dir, "figures", "gap_vs_age.png"))


def plot_feature_importance_top20(gini_df, perm_df, results_dir):
    """Top 20 features by Gini and permutation importance, color-coded by family."""
    fig, axes = plt.subplots(1, 2, figsize=(15, 8))

    for ax in axes:
        _style_ax(ax)

    if gini_df is not None and len(gini_df) > 0:
        top = gini_df.head(20).iloc[::-1]
        colors = [FAMILY_COLORS.get(f, "#6b7280") for f in top["family"]]
        axes[0].barh(range(len(top)), top["gini_importance"], color=colors, alpha=0.85, edgecolor="white")
        axes[0].set_yticks(range(len(top)))
        labels = [f"{feat}  [{fam}]" for feat, fam in zip(top["feature"], top["family"])]
        axes[0].set_yticklabels(labels, fontsize=7.5)
        axes[0].set_xlabel("Gini Importance", fontsize=11)
        axes[0].set_title("Top 20: Gini Importance", fontsize=12, fontweight="bold")

    if perm_df is not None and len(perm_df) > 0:
        top = perm_df.head(20).iloc[::-1]
        colors = [FAMILY_COLORS.get(f, "#6b7280") for f in top["family"]]
        axes[1].barh(range(len(top)), top["importance_mean"], color=colors, alpha=0.85,
                      xerr=top["importance_std"], ecolor="#9ca3af", capsize=2, edgecolor="white")
        axes[1].set_yticks(range(len(top)))
        labels = [f"{feat}  [{fam}]" for feat, fam in zip(top["feature"], top["family"])]
        axes[1].set_yticklabels(labels, fontsize=7.5)
        axes[1].set_xlabel("Permutation Importance (R² drop)", fontsize=11)
        axes[1].set_title("Top 20: Permutation Importance", fontsize=12, fontweight="bold")

    fig.suptitle("EEG Feature Importance for Age Prediction", fontsize=14, fontweight="bold")
    fig.tight_layout(rect=[0, 0, 1, 0.94])
    _save(fig, os.path.join(results_dir, "figures", "feature_importance_top20.png"))


def plot_importance_by_family(family_perm, results_dir):
    """Bar chart of mean permutation importance by feature family, color-coded."""
    fig, ax = plt.subplots(figsize=FIGSIZE_WIDE)
    _style_ax(ax)

    df = family_perm.sort_values("mean", ascending=True)
    colors = [FAMILY_COLORS.get(f, "#6b7280") for f in df.index]

    bars = ax.barh(range(len(df)), df["mean"], color=colors, alpha=0.85, edgecolor="white", height=0.6)
    ax.set_yticks(range(len(df)))
    ax.set_yticklabels(df.index, fontsize=11, fontweight="bold")

    for bar, val in zip(bars, df["mean"]):
        ax.text(val + df["mean"].max() * 0.02, bar.get_y() + bar.get_height() / 2,
                f"{val:.5f}", va="center", fontsize=10)

    ax.set_xlabel("Mean Permutation Importance (R² drop)", fontsize=11)
    ax.set_title("Feature Importance by EEG Feature Family", fontsize=13, fontweight="bold")
    _save(fig, os.path.join(results_dir, "figures", "importance_by_family.png"))


def plot_cognitive_prediction_comparison(cog_metrics_df, results_dir):
    """Compare cognitive prediction R² — RF models only, grouped by feature set."""
    if cog_metrics_df is None or len(cog_metrics_df) == 0:
        return

    subset = cog_metrics_df[cog_metrics_df["model"] != "Mean Baseline"].copy()
    targets = subset["target"].unique()

    fig, axes = plt.subplots(1, len(targets), figsize=(6.5 * len(targets), 5.5))
    if len(targets) == 1:
        axes = [axes]

    fs_colors = {"Age+Sex": "#059669", "EEG only": "#2563eb", "EEG+Age+Sex": "#7c3aed"}
    fs_order = ["Age+Sex", "EEG only", "EEG+Age+Sex"]

    for ax, target in zip(axes, targets):
        _style_ax(ax)
        t_data = subset[subset["target"] == target].copy()

        positions = []
        labels = []
        vals = []
        colors = []
        bar_idx = 0

        for fs in fs_order:
            fs_data = t_data[t_data["feature_set"] == fs].sort_values("model")
            for _, row in fs_data.iterrows():
                positions.append(bar_idx)
                short_model = "RF" if "Random" in row["model"] else "Ridge"
                labels.append(f"{fs}\n({short_model})")
                vals.append(row["R2"])
                colors.append(fs_colors.get(fs, "#6b7280"))
                bar_idx += 1

        bars = ax.bar(positions, vals, color=colors, alpha=0.85, edgecolor="white", linewidth=1.5, width=0.7)
        ax.set_xticks(positions)
        ax.set_xticklabels(labels, fontsize=8.5, ha="center")
        ax.axhline(0, color="#9ca3af", linewidth=0.8, zorder=0)

        for bar, v in zip(bars, vals):
            y_pos = max(v, 0) + 0.005
            ax.text(bar.get_x() + bar.get_width() / 2, y_pos,
                    f"{v:.3f}", ha="center", va="bottom", fontsize=9, fontweight="bold")

        readable = target.replace("_", " ").replace("seqLearning partial", "Sequential Learning").replace("symbolSearch accuracy", "Symbol Search")
        ax.set_title(readable.title(), fontsize=12, fontweight="bold")
        ax.set_ylabel("R² (10-Fold CV)", fontsize=11)

        y_min = min(vals) * 1.15 if min(vals) < 0 else -0.05
        ax.set_ylim(y_min, max(vals) * 1.4 + 0.02)

    fig.suptitle("Cognitive Prediction from EEG: Model Comparison", fontsize=14, fontweight="bold")
    fig.tight_layout(rect=[0, 0, 1, 0.93])
    _save(fig, os.path.join(results_dir, "figures", "cognitive_prediction_comparison.png"))


def plot_developmental_eeg_trends(family_summary, results_dir):
    """Bar chart of EEG family age trends, color-coded by direction."""
    fig, ax = plt.subplots(figsize=FIGSIZE_WIDE)
    _style_ax(ax)

    df = family_summary.sort_values("mean_abs_r", ascending=True)
    family_labels = {
        "power": "Band Power", "theta_alpha_ratio": "Theta/Alpha Ratio",
        "spectral_entropy": "Spectral Entropy", "sample_entropy": "Sample Entropy",
        "alpha_coherence": "Alpha Coherence",
    }
    display_names = [family_labels.get(f, f) for f in df.index]
    colors = ["#2563eb" if r > 0 else "#dc2626" for r in df["mean_r"]]

    bars = ax.barh(range(len(df)), df["mean_r"], color=colors, alpha=0.85, edgecolor="white", height=0.6)
    ax.set_yticks(range(len(df)))
    ax.set_yticklabels(display_names, fontsize=11, fontweight="bold")
    ax.axvline(0, color="#9ca3af", linewidth=0.8, zorder=0)

    for bar, val, abs_r in zip(bars, df["mean_r"], df["mean_abs_r"]):
        side = "left" if val < 0 else "right"
        x_off = -0.005 if val < 0 else 0.005
        ax.text(val + x_off, bar.get_y() + bar.get_height() / 2,
                f"r = {val:+.3f}", va="center", ha=side, fontsize=10, fontweight="bold")

    ax.set_xlabel("Mean Pearson r with Age", fontsize=12)
    ax.set_title("EEG Feature Family Correlations with Age", fontsize=13, fontweight="bold")

    from matplotlib.patches import Patch
    ax.legend([Patch(color="#2563eb"), Patch(color="#dc2626")],
              ["Increases with age", "Decreases with age"],
              loc="lower right", fontsize=10)

    _save(fig, os.path.join(results_dir, "figures", "eeg_age_trends_by_family.png"))


def plot_cognitive_age_curves(cog_curves_df, results_dir):
    """Line plots of cognitive scores across age groups, properly ordered."""
    if cog_curves_df is None or len(cog_curves_df) == 0:
        return

    age_rows = cog_curves_df[~cog_curves_df["age_group"].str.startswith("all_")].copy()
    age_rows["age_group"] = pd.Categorical(age_rows["age_group"], categories=AGE_GROUP_ORDER, ordered=True)
    targets = age_rows["target"].unique()

    fig, axes = plt.subplots(1, len(targets), figsize=(6.5 * len(targets), 5.5))
    if len(targets) == 1:
        axes = [axes]

    target_colors = {"symbolSearch_accuracy": "#2563eb", "seqLearning_partial": "#7c3aed"}
    target_names = {
        "symbolSearch_accuracy": "Symbol Search Accuracy",
        "seqLearning_partial": "Sequential Learning (Partial Credit)",
    }

    for ax, target in zip(axes, targets):
        _style_ax(ax)
        subset = age_rows[age_rows["target"] == target].sort_values("age_group")
        color = target_colors.get(target, "#2563eb")

        ax.fill_between(range(len(subset)),
                         subset["mean"] - subset["std"],
                         subset["mean"] + subset["std"],
                         alpha=0.15, color=color)
        ax.errorbar(range(len(subset)), subset["mean"], yerr=subset["std"],
                     fmt="o-", color=color, capsize=5, markersize=7, linewidth=2,
                     markeredgecolor="white", markeredgewidth=1.5, capthick=1.5)
        ax.set_xticks(range(len(subset)))
        ax.set_xticklabels(subset["age_group"], fontsize=10)
        ax.set_xlabel("Age Group (years)", fontsize=11)
        ax.set_ylabel("Score", fontsize=11)
        ax.set_title(target_names.get(target, target), fontsize=12, fontweight="bold")

        for i, row in enumerate(subset.itertuples()):
            ax.text(i, row.mean + row.std + 0.015, f"n={row.n}", ha="center", fontsize=8.5, color="#6b7280")

    fig.suptitle("Cognitive Performance Across Age Groups", fontsize=14, fontweight="bold")
    fig.tight_layout(rect=[0, 0, 1, 0.93])
    _save(fig, os.path.join(results_dir, "figures", "cognitive_age_curves.png"))


CAT_ORDER_VIZ = [
    "Lower EEG-Derived Age",
    "Age-Consistent EEG",
    "Higher EEG-Derived Age",
]
CAT_COLORS = {
    "Lower EEG-Derived Age": "#2563eb",
    "Age-Consistent EEG":    "#059669",
    "Higher EEG-Derived Age": "#dc2626",
}
SEX_COLORS = {"M": "#2563eb", "F": "#dc2626"}
AGE_GROUP_ORDER_6 = ["5-7", "8-9", "10-11", "12-13", "14-16", "17-22"]


# ═══ Experiment 6 figures ════════════════════════════════════════════════════

def plot_gap_by_age_sex(ref_stats_df, results_dir):
    """Mean ± SD of raw brain-age gap by age group, separately for M and F."""
    df = ref_stats_df[ref_stats_df["reliable"]].copy()
    if len(df) == 0:
        logger.warning("No reliable age-sex groups — skipping gap_by_age_sex figure")
        return

    df["age_bin"] = pd.Categorical(df["age_bin"], categories=AGE_GROUP_ORDER_6, ordered=True)
    df = df.sort_values("age_bin")

    fig, ax = plt.subplots(figsize=(10, 5.5))
    _style_ax(ax)

    for sex, grp in df.groupby("sex"):
        color = SEX_COLORS.get(sex, "#6b7280")
        x = [AGE_GROUP_ORDER_6.index(str(ab)) for ab in grp["age_bin"] if str(ab) in AGE_GROUP_ORDER_6]
        ax.errorbar(x, grp["mean_raw"], yerr=grp["sd_raw"],
                    fmt="o-", color=color, capsize=4, markersize=7, linewidth=2,
                    markeredgecolor="white", markeredgewidth=1.5, capthick=1.5,
                    label=f"{'Male' if sex=='M' else 'Female'}")
        for xi, (_, r) in zip(x, grp.iterrows()):
            ax.text(xi, r["mean_raw"] + r["sd_raw"] + 0.08, f"n={r['n']}",
                    ha="center", fontsize=7.5, color=color)

    ax.axhline(0, color="#9ca3af", linestyle="--", linewidth=1.0)
    ax.set_xticks(range(len(AGE_GROUP_ORDER_6)))
    ax.set_xticklabels(AGE_GROUP_ORDER_6, fontsize=10)
    ax.set_xlabel("Age Group (years)", fontsize=12)
    ax.set_ylabel("Mean Raw Brain-Age Gap ± SD (years)", fontsize=12)
    ax.set_title("Brain-Age Gap by Age Group and Sex", fontsize=13, fontweight="bold")
    ax.legend(fontsize=10)
    _save(fig, os.path.join(results_dir, "figures", "exp6_gap_by_age_sex.png"))


def plot_gap_distribution_by_sex(gap_df_ref, results_dir):
    """Overlapping histograms of raw brain-age gap by sex."""
    fig, ax = plt.subplots(figsize=(9, 5))
    _style_ax(ax)

    for sex, color, label in [("M", "#2563eb", "Male"), ("F", "#dc2626", "Female")]:
        vals = gap_df_ref[gap_df_ref["sex"] == sex]["brain_age_gap"].dropna()
        ax.hist(vals, bins=40, alpha=0.55, color=color, edgecolor="white",
                linewidth=0.6, label=f"{label} (n={len(vals)}, mean={vals.mean():.2f})")
        ax.axvline(vals.mean(), color=color, linewidth=1.5, linestyle="--")

    ax.axvline(0, color="#9ca3af", linewidth=1.0, linestyle=":", zorder=0)
    ax.set_xlabel("Raw Brain-Age Gap (years)", fontsize=12)
    ax.set_ylabel("Count", fontsize=12)
    ax.set_title("Brain-Age Gap Distribution by Sex", fontsize=13, fontweight="bold")
    ax.legend(fontsize=10)
    _save(fig, os.path.join(results_dir, "figures", "exp6_gap_distribution_by_sex.png"))


def plot_reference_percentile_curves(ref_stats_df, results_dir):
    """Percentile bands (10/25/50/75/90) of raw gap across age groups, by sex."""
    df = ref_stats_df[ref_stats_df["reliable"]].copy()
    if len(df) == 0:
        return

    df["age_bin"] = pd.Categorical(df["age_bin"], categories=AGE_GROUP_ORDER_6, ordered=True)
    df = df.sort_values(["sex", "age_bin"])

    sexes = df["sex"].unique()
    fig, axes = plt.subplots(1, len(sexes), figsize=(7 * len(sexes), 5.5), sharey=True)
    if len(sexes) == 1:
        axes = [axes]

    for ax, sex in zip(axes, sorted(sexes)):
        _style_ax(ax)
        grp = df[df["sex"] == sex].sort_values("age_bin")
        x = list(range(len(grp)))
        color = SEX_COLORS.get(sex, "#6b7280")

        ax.fill_between(x, grp["p10"], grp["p90"], alpha=0.12, color=color, label="10th–90th pct")
        ax.fill_between(x, grp["p25"], grp["p75"], alpha=0.22, color=color, label="25th–75th pct")
        ax.plot(x, grp["median_raw"], "o-", color=color, linewidth=2, markersize=7,
                markeredgecolor="white", markeredgewidth=1.5, label="Median (50th pct)")
        ax.axhline(0, color="#9ca3af", linewidth=1.0, linestyle="--")

        ax.set_xticks(x)
        ax.set_xticklabels(grp["age_bin"].tolist(), fontsize=10)
        ax.set_xlabel("Age Group (years)", fontsize=11)
        ax.set_ylabel("Raw Brain-Age Gap (years)", fontsize=11)
        ax.set_title(f"{'Male' if sex=='M' else 'Female'} Reference Percentiles", fontsize=12, fontweight="bold")
        ax.legend(fontsize=8, loc="upper right")

    fig.suptitle("Brain-Age Gap Reference Percentile Curves by Age and Sex", fontsize=13, fontweight="bold")
    fig.tight_layout(rect=[0, 0, 1, 0.94])
    _save(fig, os.path.join(results_dir, "figures", "exp6_reference_percentile_curves.png"))


def plot_gap_category_distribution(gap_df_cat, threshold_info, results_dir):
    """Histogram of reference-adjusted gap with category regions shaded."""
    fig, ax = plt.subplots(figsize=(10, 5.5))
    _style_ax(ax)

    # Use reference-adjusted gap if available, fall back to corrected gap
    gap_col = "ref_adjusted_gap" if "ref_adjusted_gap" in gap_df_cat.columns else "brain_age_gap_corrected"
    gap = gap_df_cat[gap_col].dropna().values
    lb = threshold_info["mean"] - threshold_info["primary"]
    ub = threshold_info["mean"] + threshold_info["primary"]

    xmin, xmax = gap.min() - 0.5, gap.max() + 0.5
    ax.axvspan(xmin, lb, alpha=0.10, color="#2563eb", zorder=0)
    ax.axvspan(lb, ub, alpha=0.10, color="#059669", zorder=0)
    ax.axvspan(ub, xmax, alpha=0.10, color="#dc2626", zorder=0)

    ax.hist(gap, bins=50, color="#6b7280", alpha=0.70, edgecolor="white", linewidth=0.6, zorder=2)
    ax.axvline(lb, color="#2563eb", linewidth=1.5, linestyle="--",
               label=f"Lower/Consistent boundary ({lb:+.2f} yr)")
    ax.axvline(ub, color="#dc2626", linewidth=1.5, linestyle="--",
               label=f"Consistent/Higher boundary ({ub:+.2f} yr)")
    ax.axvline(0, color="#9ca3af", linewidth=1.0, linestyle=":", zorder=1)

    n = {cat: (gap_df_cat["category"] == cat).sum() for cat in CAT_ORDER_VIZ}
    for cat, x_pos, va in [
        (CAT_ORDER_VIZ[0], (xmin + lb) / 2, 0.90),
        (CAT_ORDER_VIZ[1], (lb + ub) / 2,   0.90),
        (CAT_ORDER_VIZ[2], (ub + xmax) / 2,  0.90),
    ]:
        color = CAT_COLORS[cat]
        ax.text(x_pos, va, f"n={n.get(cat, 0)}", transform=ax.get_xaxis_transform(),
                ha="center", va="top", fontsize=9, color=color, fontweight="bold")

    xlabel = "Reference-Adjusted Brain-Age Gap (years)" if gap_col == "ref_adjusted_gap" else \
             "Age-Bias Corrected Brain-Age Gap (years)"
    ax.set_xlabel(xlabel, fontsize=12)
    ax.set_ylabel("Count", fontsize=12)
    ax.set_title(
        f"EEG Age-Discrepancy Categories  (threshold = ±{threshold_info['primary']:.2f} yr = ±0.5 SD)",
        fontsize=12, fontweight="bold",
    )
    ax.set_xlim(xmin, xmax)
    ax.legend(fontsize=9, loc="upper right")
    _save(fig, os.path.join(results_dir, "figures", "exp6_category_distribution.png"))


def plot_category_demographics(category_stats, results_dir):
    """Bar charts: n, age distribution, and sex ratio by category."""
    fig, axes = plt.subplots(1, 3, figsize=(14, 5))
    for ax in axes:
        _style_ax(ax)

    cats = category_stats["category"].tolist()
    colors = [CAT_COLORS.get(c, "#6b7280") for c in cats]
    x = np.arange(len(cats))

    # Panel 1: subject counts
    axes[0].bar(x, category_stats["n"], color=colors, alpha=0.85, edgecolor="white", width=0.6)
    for xi, (_, row) in zip(x, category_stats.iterrows()):
        axes[0].text(xi, row["n"] + 5, f"{row['pct_total']:.0f}%",
                     ha="center", fontsize=10, fontweight="bold")
    axes[0].set_xticks(x)
    axes[0].set_xticklabels([c.replace("EEG-Derived Age", "EEG\nAge").replace("Age-Consistent EEG", "Age-\nConsistent") for c in cats], fontsize=9)
    axes[0].set_ylabel("Number of subjects", fontsize=11)
    axes[0].set_title("Category Counts", fontsize=12, fontweight="bold")

    # Panel 2: mean age ± SD
    axes[1].bar(x, category_stats["age_mean"], color=colors, alpha=0.85, edgecolor="white", width=0.6,
                yerr=category_stats["age_sd"], capsize=4, error_kw={"elinewidth": 1.5})
    axes[1].set_xticks(x)
    axes[1].set_xticklabels([c.replace("EEG-Derived Age", "EEG\nAge").replace("Age-Consistent EEG", "Age-\nConsistent") for c in cats], fontsize=9)
    axes[1].set_ylabel("Chronological Age (years)", fontsize=11)
    axes[1].set_title("Age Distribution per Category", fontsize=12, fontweight="bold")

    # Panel 3: % male
    axes[2].bar(x, category_stats["pct_male"], color=colors, alpha=0.85, edgecolor="white", width=0.6)
    axes[2].axhline(50, color="#9ca3af", linestyle="--", linewidth=1.0, zorder=0)
    for xi, (_, row) in zip(x, category_stats.iterrows()):
        axes[2].text(xi, row["pct_male"] + 1, f"{row['pct_male']:.0f}%",
                     ha="center", fontsize=10, fontweight="bold")
    axes[2].set_xticks(x)
    axes[2].set_xticklabels([c.replace("EEG-Derived Age", "EEG\nAge").replace("Age-Consistent EEG", "Age-\nConsistent") for c in cats], fontsize=9)
    axes[2].set_ylabel("% Male", fontsize=11)
    axes[2].set_ylim(0, 100)
    axes[2].set_title("Sex Distribution per Category", fontsize=12, fontweight="bold")

    fig.suptitle("EEG Age-Discrepancy Category Demographics", fontsize=14, fontweight="bold")
    fig.tight_layout(rect=[0, 0, 1, 0.94])
    _save(fig, os.path.join(results_dir, "figures", "exp6_category_demographics.png"))


def plot_category_outcome_comparisons(gap_df_cat, omnibus_df, results_dir):
    """Boxplots of each outcome by category, annotated with p-values."""
    outcomes = {
        "symbolSearch_accuracy": "Symbol Search\nAccuracy",
        "seqLearning_partial": "Seq. Learning\n(Partial Credit)",
        "attention": "Attention\nProblems",
        "internalizing": "Internalizing",
        "externalizing": "Externalizing",
    }

    # Keep only outcomes that have data in gap_df_cat
    valid_outcomes = {k: v for k, v in outcomes.items() if k in gap_df_cat.columns}
    if not valid_outcomes:
        return

    ncols = len(valid_outcomes)
    fig, axes = plt.subplots(1, ncols, figsize=(3.5 * ncols, 5.5))
    if ncols == 1:
        axes = [axes]

    p_lookup = {}
    if omnibus_df is not None and len(omnibus_df) > 0:
        for _, row in omnibus_df.iterrows():
            p_lookup[row["outcome"]] = row["p_bonferroni"]

    colors = [CAT_COLORS[c] for c in CAT_ORDER_VIZ]

    for ax, (col, label) in zip(axes, valid_outcomes.items()):
        _style_ax(ax)

        groups = []
        for cat in CAT_ORDER_VIZ:
            mask = (gap_df_cat["category"] == cat) & gap_df_cat[col].notna()
            groups.append(gap_df_cat.loc[mask, col].values)

        bp = ax.boxplot(
            groups, patch_artist=True, widths=0.55,
            medianprops={"color": "white", "linewidth": 2.0},
            whiskerprops={"linewidth": 1.2},
            capprops={"linewidth": 1.5},
            flierprops={"marker": ".", "markersize": 3, "alpha": 0.3},
        )
        for patch, color in zip(bp["boxes"], colors):
            patch.set_facecolor(color)
            patch.set_alpha(0.75)

        ax.set_xticks([1, 2, 3])
        ax.set_xticklabels(["Lower", "Consistent", "Higher"], fontsize=9)
        ax.set_title(label, fontsize=11, fontweight="bold")

        # Annotate omnibus p
        p_bonf = p_lookup.get(col, np.nan)
        if np.isfinite(p_bonf):
            sig = "**" if p_bonf < 0.01 else ("*" if p_bonf < 0.05 else "ns")
            ax.text(0.98, 0.98, f"p_Bonf={p_bonf:.3f} {sig}",
                    transform=ax.transAxes, ha="right", va="top", fontsize=8,
                    bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.8))

    fig.suptitle("Outcome Scores by EEG Age-Discrepancy Category", fontsize=13, fontweight="bold")
    fig.tight_layout(rect=[0, 0, 1, 0.93])
    _save(fig, os.path.join(results_dir, "figures", "exp6_outcome_comparisons.png"))


def plot_individual_example(gap_df_ref, results_dir, subject_idx=None):
    """
    Show a single participant's raw gap vs the reference distribution for
    their age-sex group. Picks the subject closest to the median of the
    'Age-Consistent EEG' group by default, so the example is representative.
    """
    consistent = gap_df_ref[gap_df_ref["category"] == "Age-Consistent EEG"]
    if len(consistent) == 0:
        return

    # Choose a subject whose gap is near the group median
    if subject_idx is None:
        median_gap = consistent["brain_age_gap"].median()
        subject_idx = (consistent["brain_age_gap"] - median_gap).abs().idxmin()

    subj = gap_df_ref.loc[subject_idx]
    age_bin = subj["age_bin"]
    sex = subj["sex"]
    gap_val = subj["brain_age_gap"]
    ref_mean = subj.get("loo_ref_mean", np.nan)
    ref_sd = subj.get("loo_ref_sd", np.nan)
    z = subj.get("z_score", np.nan)
    pct = subj.get("percentile_within_group", np.nan)

    # Get peer group distribution (all subjects with same age_bin and sex, excluding subj)
    peers = gap_df_ref[
        (gap_df_ref["age_bin"] == age_bin) &
        (gap_df_ref["sex"] == sex) &
        (gap_df_ref.index != subject_idx)
    ]["brain_age_gap"].dropna().values

    if len(peers) < 5:
        return

    fig, ax = plt.subplots(figsize=(9, 5))
    _style_ax(ax)

    ax.hist(peers, bins=25, color="#6b7280", alpha=0.65, edgecolor="white", linewidth=0.7,
            label=f"Peers: {age_bin} yr, {'Male' if sex=='M' else 'Female'} (n={len(peers)})")
    ax.axvline(np.mean(peers), color="#9ca3af", linewidth=1.5, linestyle="--", label="Peer mean")

    # Mark the example subject
    ax.axvline(gap_val, color="#dc2626", linewidth=2.5, linestyle="-",
               label=f"This participant ({gap_val:+.2f} yr)")

    # Annotation
    info_lines = [
        f"Actual age: {subj['actual_age']:.1f} yr",
        f"EEG-predicted age: {subj['predicted_age']:.1f} yr",
        f"Raw gap: {gap_val:+.2f} yr",
        f"Reference mean (peers): {ref_mean:+.2f} yr" if np.isfinite(ref_mean) else "Reference: unreliable",
        f"Z-score: {z:.2f}" if np.isfinite(z) else "Z-score: N/A",
        f"Percentile: {pct:.0f}th" if np.isfinite(pct) else "Percentile: N/A",
        f"Category: {subj['category']}",
    ]
    ax.text(0.98, 0.97, "\n".join(info_lines), transform=ax.transAxes,
            ha="right", va="top", fontsize=9.5,
            bbox=dict(boxstyle="round,pad=0.5", facecolor="#f0f9ff", edgecolor="#2563eb", alpha=0.9))

    ax.set_xlabel("Raw Brain-Age Gap (years)", fontsize=12)
    ax.set_ylabel("Count", fontsize=12)
    ax.set_title(
        f"Individual Reference Comparison Example\n"
        f"({age_bin} yr, {'Male' if sex=='M' else 'Female'} reference group)",
        fontsize=12, fontweight="bold"
    )
    ax.legend(fontsize=9, loc="upper left")
    _save(fig, os.path.join(results_dir, "figures", "exp6_individual_example.png"))


def plot_sensitivity_analysis(sensitivity_df, results_dir):
    """Category sizes and significant outcomes across threshold choices."""
    if sensitivity_df is None or len(sensitivity_df) == 0:
        return

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    for ax in axes:
        _style_ax(ax)

    x = np.arange(len(sensitivity_df))
    labels = sensitivity_df["threshold_label"].tolist()

    bot = np.zeros(len(sensitivity_df))
    for col, cat, color in [
        ("pct_lower", "Lower EEG-Derived Age", "#2563eb"),
        ("pct_consistent", "Age-Consistent EEG", "#059669"),
        ("pct_higher", "Higher EEG-Derived Age", "#dc2626"),
    ]:
        vals = sensitivity_df[col].values
        axes[0].bar(x, vals, bottom=bot, color=color, alpha=0.80,
                    edgecolor="white", width=0.6, label=cat)
        bot += vals
    axes[0].set_xticks(x)
    axes[0].set_xticklabels(labels, fontsize=9, rotation=15, ha="right")
    axes[0].set_ylabel("% of subjects", fontsize=11)
    axes[0].set_ylim(0, 105)
    axes[0].set_title("Category Sizes Across Thresholds", fontsize=12, fontweight="bold")
    axes[0].legend(fontsize=8, loc="upper right")

    # Accept either old or new column names
    raw_col = "n_sig_raw" if "n_sig_raw" in sensitivity_df.columns else "n_outcomes_p05_raw"
    bonf_col = "n_sig_bonferroni" if "n_sig_bonferroni" in sensitivity_df.columns else "n_outcomes_p05_bonferroni"
    axes[1].plot(x, sensitivity_df[raw_col], "o-", color="#2563eb", linewidth=2, markersize=8, label="Raw p < 0.05")
    axes[1].plot(x, sensitivity_df[bonf_col], "s--", color="#dc2626", linewidth=2, markersize=8, label="Bonferroni p < 0.05")
    axes[1].set_xticks(x)
    axes[1].set_xticklabels(labels, fontsize=9, rotation=15, ha="right")
    axes[1].set_ylabel("Number of significant outcomes", fontsize=11)
    axes[1].set_yticks(range(0, 8))
    axes[1].set_title("Significant Outcomes Across Thresholds", fontsize=12, fontweight="bold")
    axes[1].legend(fontsize=9)
    axes[0].axvline(0, color="#f59e0b", linewidth=2, linestyle=":", alpha=0.8)
    axes[1].axvline(0, color="#f59e0b", linewidth=2, linestyle=":", alpha=0.8)

    fig.suptitle("Sensitivity Analysis: Threshold Choice Impact", fontsize=13, fontweight="bold")
    fig.tight_layout(rect=[0, 0, 1, 0.94])
    _save(fig, os.path.join(results_dir, "figures", "exp6_sensitivity_analysis.png"))


def generate_all_figures(exp1_results, exp2_results, exp3_results,
                          exp4_results, exp5_results, results_dir, exp6_results=None):
    """Generate all publication figures from experiment results."""
    logger.info("=" * 60)
    logger.info("GENERATING FIGURES")
    logger.info("=" * 60)

    fig_dir = os.path.join(results_dir, "figures")
    os.makedirs(fig_dir, exist_ok=True)

    if exp1_results:
        logger.info("\n--- Experiment 1 figures ---")
        plot_age_prediction_scatter(exp1_results["predictions_df"], results_dir)
        plot_age_prediction_by_group(exp1_results["age_group_metrics"], results_dir)
        plot_model_comparison(exp1_results["summary_df"], results_dir)

    if exp2_results:
        logger.info("\n--- Experiment 2 figures ---")
        plot_brain_age_gap_distribution(exp2_results["gap_df"], results_dir)
        plot_gap_vs_age(exp2_results["gap_df"], results_dir)

    if exp3_results:
        logger.info("\n--- Experiment 3 figures ---")
        plot_feature_importance_top20(
            exp3_results.get("gini_importance"),
            exp3_results.get("permutation_importance"),
            results_dir,
        )
        if exp3_results.get("family_perm") is not None:
            plot_importance_by_family(exp3_results["family_perm"], results_dir)

    if exp4_results:
        logger.info("\n--- Experiment 4 figures ---")
        plot_cognitive_prediction_comparison(exp4_results.get("metrics"), results_dir)

    if exp5_results:
        logger.info("\n--- Experiment 5 figures ---")
        if exp5_results.get("age_trends_family") is not None:
            plot_developmental_eeg_trends(exp5_results["age_trends_family"], results_dir)
        if exp5_results.get("cognitive_age_curves") is not None:
            plot_cognitive_age_curves(exp5_results["cognitive_age_curves"], results_dir)

    if exp6_results:
        logger.info("\n--- Experiment 6 figures ---")
        # Key: new module uses gap_df_ref; old module used gap_df_cat — handle both
        gap_df_key = "gap_df_ref" if "gap_df_ref" in exp6_results else "gap_df_cat"
        gap_df6 = exp6_results[gap_df_key]

        if exp6_results.get("ref_stats") is not None:
            plot_gap_by_age_sex(exp6_results["ref_stats"], results_dir)
            plot_gap_distribution_by_sex(gap_df6, results_dir)
            plot_reference_percentile_curves(exp6_results["ref_stats"], results_dir)

        plot_gap_category_distribution(gap_df6, exp6_results["threshold_info"], results_dir)
        plot_category_demographics(exp6_results["category_stats"], results_dir)
        plot_category_outcome_comparisons(gap_df6, exp6_results.get("omnibus_df"), results_dir)

        if gap_df_key == "gap_df_ref":
            plot_individual_example(gap_df6, results_dir)

        plot_sensitivity_analysis(exp6_results.get("sensitivity_df"), results_dir)

    logger.info("\nAll figures generated.")
