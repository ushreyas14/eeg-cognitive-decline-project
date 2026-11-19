"""
Experiment 1: EEG-based chronological age prediction.
"""

import numpy as np
import pandas as pd
import logging
import time
import json
import os
import joblib

from sklearn.linear_model import Ridge
from sklearn.ensemble import RandomForestRegressor
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.model_selection import KFold, GroupKFold
from sklearn.base import BaseEstimator, RegressorMixin
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
from scipy.stats import pearsonr

logger = logging.getLogger(__name__)


class MeanBaselineRegressor(BaseEstimator, RegressorMixin):
    """Predicts the training-set mean for every sample."""
    def __init__(self):
        self.mean_ = None
    def fit(self, X, y):
        self.mean_ = np.mean(y)
        return self
    def predict(self, X):
        return np.full(len(X), self.mean_)


def build_pipelines(random_state=42):
    """Return dict of named sklearn pipelines."""
    return {
        "Mean Baseline": Pipeline([
            ("imputer", SimpleImputer(strategy="median")),
            ("model", MeanBaselineRegressor()),
        ]),
        "Ridge Regression": Pipeline([
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
            ("model", Ridge(alpha=1.0)),
        ]),
        "Random Forest": Pipeline([
            ("imputer", SimpleImputer(strategy="median")),
            ("model", RandomForestRegressor(
                n_estimators=300, max_features="sqrt", min_samples_leaf=5,
                random_state=random_state, n_jobs=-1,
            )),
        ]),
    }


def evaluate_predictions(y_true, y_pred):
    """Compute regression metrics."""
    r2 = r2_score(y_true, y_pred)
    mae = mean_absolute_error(y_true, y_pred)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    r, pval = pearsonr(y_true, y_pred)
    bias = np.mean(y_pred - y_true)
    return {"R2": r2, "MAE": mae, "RMSE": rmse, "Pearson_r": r, "Pearson_p": pval, "Bias": bias}


def run_cv(X, y, groups=None, cv_type="kfold", n_splits=10, random_state=42, model_name="Model"):
    """
    Run cross-validation for a single pipeline.

    Returns:
        oof_preds: array of out-of-fold predictions
        fold_metrics: list of per-fold metric dicts
    """
    if cv_type == "kfold":
        cv = KFold(n_splits=n_splits, shuffle=True, random_state=random_state)
        split_iter = cv.split(X)
    elif cv_type == "group":
        cv = GroupKFold(n_splits=n_splits)
        split_iter = cv.split(X, y, groups=groups)
    else:
        raise ValueError(f"Unknown cv_type: {cv_type}")

    oof_preds = np.full(len(y), np.nan)
    fold_metrics = []
    pipelines = build_pipelines(random_state)
    pipeline = pipelines[model_name]

    for fold_idx, (train_idx, test_idx) in enumerate(split_iter):
        X_train, X_test = X[train_idx], X[test_idx]
        y_train, y_test = y[train_idx], y[test_idx]

        from sklearn.base import clone
        pipe = clone(pipeline)
        pipe.fit(X_train, y_train)
        preds = pipe.predict(X_test)
        oof_preds[test_idx] = preds

        metrics = evaluate_predictions(y_test, preds)
        metrics["fold"] = fold_idx
        metrics["n_train"] = len(train_idx)
        metrics["n_test"] = len(test_idx)
        fold_metrics.append(metrics)
        logger.info("  Fold %d/%d: R²=%.4f, MAE=%.3f yr", fold_idx + 1, n_splits, metrics["R2"], metrics["MAE"])

    return oof_preds, fold_metrics


def run_age_prediction(data, results_dir, random_state=42):
    """
    Run the complete age prediction experiment.

    Args:
        data: dict from data_loader.build_dataset
        results_dir: path to save results

    Returns:
        results: dict with all metrics, predictions, and model artifacts
    """
    logger.info("=" * 60)
    logger.info("EXPERIMENT 1: EEG-Based Age Prediction")
    logger.info("=" * 60)
    start_time = time.time()

    X = data["X_eeg"]
    y = data["targets"]["age"]
    groups = data["release"]
    n, p = X.shape
    logger.info("Input: %d subjects × %d EEG features", n, p)
    logger.info("Target: age (%.1f–%.1f years, mean=%.1f)", y.min(), y.max(), y.mean())

    model_names = ["Mean Baseline", "Ridge Regression", "Random Forest"]
    all_results = {}

    # --- Primary CV: 10-fold KFold ---
    logger.info("\n--- Primary evaluation: 10-fold KFold ---")
    for name in model_names:
        logger.info("\nTraining %s...", name)
        t0 = time.time()
        oof_preds, fold_metrics = run_cv(
            X, y, cv_type="kfold", n_splits=10, random_state=random_state, model_name=name,
        )
        elapsed = time.time() - t0

        overall = evaluate_predictions(y[~np.isnan(oof_preds)], oof_preds[~np.isnan(oof_preds)])
        overall["cv_type"] = "kfold"
        overall["model"] = name
        overall["elapsed_seconds"] = elapsed

        all_results[f"kfold_{name}"] = {
            "oof_predictions": oof_preds,
            "fold_metrics": fold_metrics,
            "overall_metrics": overall,
        }
        logger.info(
            "  Overall: R²=%.4f, MAE=%.3f yr, RMSE=%.3f yr, r=%.4f [%.1fs]",
            overall["R2"], overall["MAE"], overall["RMSE"], overall["Pearson_r"], elapsed,
        )

    # --- Supplementary: GroupKFold by release ---
    logger.info("\n--- Supplementary: GroupKFold by release ---")
    unique_releases = np.unique(groups)
    n_groups = len(unique_releases)
    logger.info("Groups: %d releases", n_groups)

    for name in ["Ridge Regression", "Random Forest"]:
        logger.info("\nTraining %s (GroupKFold)...", name)
        t0 = time.time()
        oof_preds, fold_metrics = run_cv(
            X, y, groups=groups, cv_type="group", n_splits=n_groups,
            random_state=random_state, model_name=name,
        )
        elapsed = time.time() - t0

        overall = evaluate_predictions(y[~np.isnan(oof_preds)], oof_preds[~np.isnan(oof_preds)])
        overall["cv_type"] = "group_kfold"
        overall["model"] = name
        overall["elapsed_seconds"] = elapsed

        all_results[f"group_{name}"] = {
            "oof_predictions": oof_preds,
            "fold_metrics": fold_metrics,
            "overall_metrics": overall,
        }
        logger.info(
            "  Overall: R²=%.4f, MAE=%.3f yr, RMSE=%.3f yr, r=%.4f [%.1fs]",
            overall["R2"], overall["MAE"], overall["RMSE"], overall["Pearson_r"], elapsed,
        )

    # --- Train final model on full data for feature importance ---
    logger.info("\nTraining final Random Forest on full data for feature importance...")
    final_pipe = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("model", RandomForestRegressor(
            n_estimators=300, max_features="sqrt", min_samples_leaf=5,
            random_state=random_state, n_jobs=-1,
        )),
    ])
    X_imputed = SimpleImputer(strategy="median").fit_transform(X)
    final_pipe.fit(X, y)

    # --- Per-group analysis (age group, sex, release) ---
    best_key = "kfold_Random Forest"
    best_preds = all_results[best_key]["oof_predictions"]

    # By age group
    age_bins = [5, 8, 10, 12, 14, 17, 23]
    age_labels = ["5-7", "8-9", "10-11", "12-13", "14-16", "17-22"]
    age_groups = pd.cut(y, bins=age_bins, labels=age_labels, right=False)
    age_group_metrics = []
    for grp in age_labels:
        mask = (age_groups == grp) & ~np.isnan(best_preds)
        if mask.sum() == 0:
            continue
        m = evaluate_predictions(y[mask], best_preds[mask])
        m["age_group"] = grp
        m["n"] = int(mask.sum())
        age_group_metrics.append(m)

    # By sex
    sex_metrics = []
    for sex_val, sex_label in [(1, "Male"), (0, "Female")]:
        mask = (data["sex"] == sex_val) & ~np.isnan(best_preds)
        m = evaluate_predictions(y[mask], best_preds[mask])
        m["sex"] = sex_label
        m["n"] = int(mask.sum())
        sex_metrics.append(m)

    # By release
    release_metrics = []
    for rel in sorted(np.unique(groups)):
        mask = (groups == rel) & ~np.isnan(best_preds)
        if mask.sum() == 0:
            continue
        m = evaluate_predictions(y[mask], best_preds[mask])
        m["release"] = rel
        m["n"] = int(mask.sum())
        release_metrics.append(m)

    # --- Save results ---
    os.makedirs(os.path.join(results_dir, "metrics"), exist_ok=True)
    os.makedirs(os.path.join(results_dir, "predictions"), exist_ok=True)
    os.makedirs(os.path.join(results_dir, "tables"), exist_ok=True)
    os.makedirs(os.path.join(results_dir, "models"), exist_ok=True)

    # Save OOF predictions
    pred_df = pd.DataFrame({
        "subject_id": data["subject_ids"],
        "actual_age": y,
        "predicted_age_rf_kfold": all_results["kfold_Random Forest"]["oof_predictions"],
        "predicted_age_ridge_kfold": all_results["kfold_Ridge Regression"]["oof_predictions"],
        "sex": data["sex_labels"],
        "release": groups,
    })
    pred_path = os.path.join(results_dir, "predictions", "age_predictions.csv")
    pred_df.to_csv(pred_path, index=False)
    logger.info("Saved predictions to %s", pred_path)

    # Save metrics summary
    summary_rows = []
    for key, res in all_results.items():
        row = res["overall_metrics"].copy()
        row["experiment"] = key
        summary_rows.append(row)
    summary_df = pd.DataFrame(summary_rows)
    summary_path = os.path.join(results_dir, "metrics", "age_prediction_metrics.csv")
    summary_df.to_csv(summary_path, index=False)

    # Save breakdown tables
    pd.DataFrame(age_group_metrics).to_csv(
        os.path.join(results_dir, "tables", "age_prediction_by_age_group.csv"), index=False
    )
    pd.DataFrame(sex_metrics).to_csv(
        os.path.join(results_dir, "tables", "age_prediction_by_sex.csv"), index=False
    )
    pd.DataFrame(release_metrics).to_csv(
        os.path.join(results_dir, "tables", "age_prediction_by_release.csv"), index=False
    )

    # Save model
    model_path = os.path.join(results_dir, "models", "age_rf_full.joblib")
    joblib.dump(final_pipe, model_path)
    logger.info("Saved final RF model to %s", model_path)

    total_time = time.time() - start_time
    logger.info("\nExperiment 1 complete in %.1f seconds", total_time)

    return {
        "all_results": all_results,
        "age_group_metrics": age_group_metrics,
        "sex_metrics": sex_metrics,
        "release_metrics": release_metrics,
        "final_model": final_pipe,
        "predictions_df": pred_df,
        "summary_df": summary_df,
        "total_time": total_time,
    }
