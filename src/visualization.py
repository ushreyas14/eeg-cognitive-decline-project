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


def generate_all_figures(exp1_results, exp2_results, exp3_results,
                          exp4_results, exp5_results, results_dir):
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

    logger.info("\nAll figures generated.")
