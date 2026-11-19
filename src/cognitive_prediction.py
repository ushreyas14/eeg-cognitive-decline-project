"""
Experiment 4: Cognitive performance prediction.
Predicts Symbol Search accuracy and Sequential Learning partial credit
from EEG features, age, and sex.
"""

import numpy as np
import pandas as pd
import logging
import time
import os

from sklearn.linear_model import Ridge
from sklearn.ensemble import RandomForestRegressor
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.model_selection import KFold
from sklearn.base import BaseEstimator, RegressorMixin
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
from scipy.stats import pearsonr

logger = logging.getLogger(__name__)


class MeanBaselineRegressor(BaseEstimator, RegressorMixin):
    def __init__(self):
        self.mean_ = None
    def fit(self, X, y):
        self.mean_ = np.mean(y)
        return self
    def predict(self, X):
        return np.full(len(X), self.mean_)


def build_cognitive_pipelines(random_state=42):
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
                n_estimators=200, max_features="sqrt", min_samples_leaf=5,
                random_state=random_state, n_jobs=-1,
            )),
        ]),
    }


def evaluate_regression(y_true, y_pred):
    mask = np.isfinite(y_true) & np.isfinite(y_pred)
    if mask.sum() < 5:
        return {"R2": np.nan, "MAE": np.nan, "RMSE": np.nan, "Pearson_r": np.nan}
    yt, yp = y_true[mask], y_pred[mask]
    r2 = r2_score(yt, yp)
    mae = mean_absolute_error(yt, yp)
    rmse = np.sqrt(mean_squared_error(yt, yp))
    r, p = pearsonr(yt, yp)
    return {"R2": r2, "MAE": mae, "RMSE": rmse, "Pearson_r": r, "Pearson_p": p}


def run_cognitive_cv(X, y, model_name, n_splits=10, random_state=42):
    """Run CV for a single model and return OOF predictions + fold metrics."""
    cv = KFold(n_splits=n_splits, shuffle=True, random_state=random_state)
    pipelines = build_cognitive_pipelines(random_state)
    pipeline = pipelines[model_name]

    oof_preds = np.full(len(y), np.nan)
    fold_metrics = []

    from sklearn.base import clone
    for fold_idx, (train_idx, test_idx) in enumerate(cv.split(X)):
        pipe = clone(pipeline)
        pipe.fit(X[train_idx], y[train_idx])
        preds = pipe.predict(X[test_idx])
        oof_preds[test_idx] = preds

        metrics = evaluate_regression(y[test_idx], preds)
        metrics["fold"] = fold_idx
        fold_metrics.append(metrics)

    return oof_preds, fold_metrics


def run_cognitive_experiment(X_eeg, age, sex, y, target_name, feature_names, results_dir,
                              n_splits=10, random_state=42):
    """
    Run prediction for a single cognitive target with multiple feature sets and models.

    Feature sets:
    1. Age + Sex only (baseline)
    2. EEG only
    3. EEG + Age + Sex (full model)
    """
    logger.info("\n--- Target: %s (n=%d) ---", target_name, len(y))

    feature_sets = {
        "Age+Sex": np.column_stack([age, sex]),
        "EEG only": X_eeg,
        "EEG+Age+Sex": np.column_stack([X_eeg, age, sex]),
    }

    model_names = ["Mean Baseline", "Ridge Regression", "Random Forest"]
    all_results = []

    for fs_name, X in feature_sets.items():
        for model_name in model_names:
            # Skip redundant: mean baseline is same for all feature sets
            if model_name == "Mean Baseline" and fs_name != "Age+Sex":
                continue

            logger.info("  %s / %s (n_features=%d)...", fs_name, model_name, X.shape[1])
            t0 = time.time()
            oof_preds, fold_metrics = run_cognitive_cv(X, y, model_name, n_splits, random_state)
            elapsed = time.time() - t0

            overall = evaluate_regression(y, oof_preds)
            overall["feature_set"] = fs_name
            overall["model"] = model_name
            overall["target"] = target_name
            overall["n"] = len(y)
            overall["n_features"] = X.shape[1]
            overall["elapsed_seconds"] = elapsed
            all_results.append(overall)

            logger.info("    R²=%.4f, MAE=%.4f, r=%.4f [%.1fs]",
                        overall["R2"], overall["MAE"], overall["Pearson_r"], elapsed)

    results_df = pd.DataFrame(all_results)

    # Save per-target OOF predictions for the best model
    best_row = results_df.loc[results_df["R2"].idxmax()]
    logger.info("  Best: %s / %s (R²=%.4f)", best_row["feature_set"], best_row["model"], best_row["R2"])

    # Re-run best config to get OOF predictions
    best_fs = feature_sets[best_row["feature_set"]]
    best_oof, _ = run_cognitive_cv(best_fs, y, best_row["model"], n_splits, random_state)

    return results_df, best_oof


def run_cognitive_prediction(data, results_dir, random_state=42):
    """
    Run Experiment 4: cognitive performance prediction.
    """
    logger.info("=" * 60)
    logger.info("EXPERIMENT 4: Cognitive Performance Prediction")
    logger.info("=" * 60)
    start_time = time.time()

    X_eeg = data["X_eeg"]
    age = data["targets"]["age"]
    sex = data["sex"]
    feature_names = data["feature_names"]

    all_metrics = []
    all_predictions = {}

    # --- 4A: Sequential Learning (partial credit) ---
    seq_partial = data["targets"]["seqLearning_partial"]
    seq_mask = np.isfinite(seq_partial)
    n_seq = seq_mask.sum()
    logger.info("\nSequential Learning partial credit: %d subjects available", n_seq)

    if n_seq >= 50:
        seq_results, seq_oof = run_cognitive_experiment(
            X_eeg[seq_mask], age[seq_mask], sex[seq_mask], seq_partial[seq_mask],
            "seqLearning_partial", feature_names, results_dir, random_state=random_state,
        )
        all_metrics.append(seq_results)

        pred_df_seq = pd.DataFrame({
            "subject_id": data["subject_ids"][seq_mask],
            "actual": seq_partial[seq_mask],
            "predicted": seq_oof,
            "age": age[seq_mask],
            "sex": data["sex_labels"][seq_mask],
        })
        all_predictions["seqLearning_partial"] = pred_df_seq
    else:
        logger.warning("Too few subjects for seqLearning_partial (%d < 50)", n_seq)

    # --- 4B: Symbol Search accuracy ---
    sym_acc = data["targets"]["symbolSearch_accuracy"]
    sym_mask = np.isfinite(sym_acc)
    n_sym = sym_mask.sum()
    logger.info("\nSymbol Search accuracy: %d subjects available", n_sym)

    if n_sym >= 50:
        sym_results, sym_oof = run_cognitive_experiment(
            X_eeg[sym_mask], age[sym_mask], sex[sym_mask], sym_acc[sym_mask],
            "symbolSearch_accuracy", feature_names, results_dir, random_state=random_state,
        )
        all_metrics.append(sym_results)

        pred_df_sym = pd.DataFrame({
            "subject_id": data["subject_ids"][sym_mask],
            "actual": sym_acc[sym_mask],
            "predicted": sym_oof,
            "age": age[sym_mask],
            "sex": data["sex_labels"][sym_mask],
        })
        all_predictions["symbolSearch_accuracy"] = pred_df_sym
    else:
        logger.warning("Too few subjects for symbolSearch_accuracy (%d < 50)", n_sym)

    # Combine and save
    if all_metrics:
        combined = pd.concat(all_metrics, ignore_index=True)
        os.makedirs(os.path.join(results_dir, "metrics"), exist_ok=True)
        os.makedirs(os.path.join(results_dir, "predictions"), exist_ok=True)
        os.makedirs(os.path.join(results_dir, "tables"), exist_ok=True)

        combined.to_csv(os.path.join(results_dir, "metrics", "cognitive_prediction_metrics.csv"), index=False)
        logger.info("\nCognitive prediction metrics saved")

        # Save per-target summary
        summary = combined.groupby(["target", "feature_set", "model"]).first().reset_index()
        summary.to_csv(os.path.join(results_dir, "tables", "cognitive_model_comparison.csv"), index=False)

        for target_name, pred_df in all_predictions.items():
            pred_df.to_csv(
                os.path.join(results_dir, "predictions", f"cognitive_{target_name}_predictions.csv"),
                index=False,
            )

    total_time = time.time() - start_time
    logger.info("\nExperiment 4 complete in %.1f seconds", total_time)

    return {
        "metrics": pd.concat(all_metrics, ignore_index=True) if all_metrics else pd.DataFrame(),
        "predictions": all_predictions,
        "total_time": total_time,
    }
