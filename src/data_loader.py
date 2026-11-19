"""
Data loading and preprocessing for the neurodevelopment project.
Reads the existing HBN-EEG feature matrix and prepares it for modeling.
"""

import os
import numpy as np
import pandas as pd
import logging

logger = logging.getLogger(__name__)

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
REPO_ROOT = os.path.dirname(PROJECT_ROOT)
FEATURE_MATRIX_PATH = os.path.join(REPO_ROOT, "data", "processed", "feature_matrix.csv")
RAW_DATA_ROOT = os.path.join(REPO_ROOT, "data", "raw")

META_COLS = [
    "subject_id", "attention", "attention_high", "p_factor",
    "internalizing", "externalizing", "age", "sex", "release",
    "attention_source_value", "attention_source_scale",
]
BEHAVIORAL_COLS = [
    "symbolSearch_mean_rt", "symbolSearch_std_rt", "symbolSearch_accuracy", "symbolSearch_rt_iqr",
    "seqLearning_mean_rt", "seqLearning_std_rt", "seqLearning_accuracy", "seqLearning_rt_iqr",
]
POWER_BANDS = ["delta", "theta", "alpha", "beta", "gamma"]
HIGH_NAN_THRESHOLD = 0.5  # drop features with >50% NaN


def load_feature_matrix(path=None):
    """Load the raw feature matrix CSV."""
    path = path or FEATURE_MATRIX_PATH
    logger.info("Loading feature matrix from %s", path)
    df = pd.read_csv(path)
    logger.info("Loaded %d subjects × %d columns", len(df), len(df.columns))
    return df


def identify_eeg_columns(df):
    """Return list of EEG feature column names."""
    return [c for c in df.columns if c not in META_COLS + BEHAVIORAL_COLS]


def _is_power_feature(col_name):
    """Check if a column is a band-power feature (needs log transform)."""
    prefix = col_name.split("_")[0]
    if prefix not in POWER_BANDS:
        return False
    if any(kw in col_name for kw in ["ratio", "asym", "coh", "entropy"]):
        return False
    return True


def prepare_eeg_features(df, log_transform_power=True):
    """
    Prepare EEG feature matrix for modeling.

    Steps:
    1. Identify EEG columns
    2. Drop features with >50% NaN
    3. Log10-transform band-power features
    4. Return feature DataFrame and column metadata

    Returns:
        X_df: DataFrame of processed EEG features
        feature_info: dict with column lists by family
    """
    eeg_cols = identify_eeg_columns(df)
    logger.info("Total EEG columns: %d", len(eeg_cols))

    # Drop high-NaN features
    nan_frac = df[eeg_cols].isna().mean()
    keep_cols = [c for c in eeg_cols if nan_frac[c] < HIGH_NAN_THRESHOLD]
    dropped = len(eeg_cols) - len(keep_cols)
    if dropped > 0:
        logger.info("Dropped %d features with >%.0f%% NaN", dropped, HIGH_NAN_THRESHOLD * 100)
    logger.info("Retained %d EEG features", len(keep_cols))

    X_df = df[keep_cols].copy()

    # Classify features
    power_cols = [c for c in keep_cols if _is_power_feature(c)]
    ratio_cols = [c for c in keep_cols if "theta_alpha_ratio" in c]
    spectral_cols = [c for c in keep_cols if "spectral_entropy" in c]
    sample_cols = [c for c in keep_cols if "sample_entropy" in c]
    coh_cols = [c for c in keep_cols if "alpha_coh" in c]
    other_cols = [c for c in keep_cols if c not in power_cols + ratio_cols + spectral_cols + sample_cols + coh_cols]

    # Log-transform power features
    if log_transform_power and power_cols:
        logger.info("Log10-transforming %d band-power features", len(power_cols))
        for c in power_cols:
            vals = X_df[c]
            min_pos = vals[vals > 0].min() if (vals > 0).any() else 1e-30
            X_df[c] = np.log10(vals.clip(lower=min_pos))

    feature_info = {
        "all": keep_cols,
        "power": power_cols,
        "theta_alpha_ratio": ratio_cols,
        "spectral_entropy": spectral_cols,
        "sample_entropy": sample_cols,
        "alpha_coherence": coh_cols,
        "other": other_cols,
    }
    logger.info(
        "Feature families: power=%d, ratio=%d, spectral_ent=%d, sample_ent=%d, coherence=%d, other=%d",
        len(power_cols), len(ratio_cols), len(spectral_cols), len(sample_cols), len(coh_cols), len(other_cols),
    )
    return X_df, feature_info


def compute_seq_learning_partial_credit(raw_root=None):
    """
    Re-score Sequential Learning from raw event files using position-wise partial credit.

    For each subject, scores the best attempt by counting how many positions in the
    recalled sequence match the correct sequence, divided by sequence length.

    Returns:
        DataFrame with columns: subject_id, seq_partial_credit, seq_exact_match, n_attempts
    """
    raw_root = raw_root or RAW_DATA_ROOT
    logger.info("Computing seqLearning partial credit from %s", raw_root)

    all_files = []
    for root, dirs, files in os.walk(raw_root):
        for f in files:
            if f.endswith("_events.tsv") and "seqLearning" in f:
                all_files.append(os.path.join(root, f))

    logger.info("Found %d seqLearning event files", len(all_files))
    if not all_files:
        logger.warning("No seqLearning event files found")
        return pd.DataFrame(columns=["subject_id", "seq_partial_credit", "seq_exact_match", "n_attempts"])

    results = {}
    for fp in all_files:
        df = pd.read_csv(fp, sep="\t")
        if "user_answer" not in df.columns or "correct_answer" not in df.columns:
            continue

        bname = os.path.basename(fp)
        subj = bname.split("_")[0].replace("sub-", "")
        resp = df[["user_answer", "correct_answer"]].replace({"n/a": np.nan, "": np.nan}).dropna()
        if len(resp) == 0:
            continue

        correct_seq = str(resp["correct_answer"].iloc[0])
        ca_parts = correct_seq.split("-")
        seq_len = len(ca_parts)

        n_exact = 0
        best_partial = 0.0
        for ua in resp["user_answer"]:
            ua_str = str(ua).strip()
            if ua_str == correct_seq:
                n_exact += 1
            ua_parts = ua_str.split("-")
            if len(ua_parts) == seq_len:
                pos_correct = sum(u == c for u, c in zip(ua_parts, ca_parts))
                best_partial = max(best_partial, pos_correct / seq_len)

        if subj not in results:
            results[subj] = {"exact_scores": [], "partial_scores": [], "n_attempts": 0}
        results[subj]["exact_scores"].append(n_exact / len(resp))
        results[subj]["partial_scores"].append(best_partial)
        results[subj]["n_attempts"] += len(resp)

    rows = []
    for subj, d in results.items():
        rows.append({
            "subject_id": subj,
            "seq_partial_credit": np.mean(d["partial_scores"]),
            "seq_exact_match": np.mean(d["exact_scores"]),
            "n_attempts": d["n_attempts"],
        })

    out = pd.DataFrame(rows)
    logger.info("Computed partial credit for %d subjects", len(out))
    return out


def build_dataset(feature_matrix_path=None, raw_root=None):
    """
    Build the complete modeling dataset.

    Returns:
        data: dict with keys:
            - df: full merged DataFrame
            - X_eeg: processed EEG feature array
            - feature_names: list of EEG feature names
            - feature_info: dict of feature family lists
            - age: array of ages
            - sex: binary array (1=M, 0=F)
            - release: array of release labels
            - subject_ids: array of subject IDs
            - targets: dict of target arrays
    """
    df = load_feature_matrix(feature_matrix_path)

    # Prepare EEG features
    X_df, feature_info = prepare_eeg_features(df)

    # Compute seqLearning partial credit
    seq_pc = compute_seq_learning_partial_credit(raw_root)
    if len(seq_pc) > 0:
        df = df.merge(seq_pc[["subject_id", "seq_partial_credit"]], on="subject_id", how="left")
        logger.info("seqLearning partial credit: %d subjects matched", df["seq_partial_credit"].notna().sum())
    else:
        df["seq_partial_credit"] = np.nan

    # Encode sex
    sex_binary = (df["sex"] == "M").astype(int).values

    # Build target dict
    targets = {
        "age": df["age"].values.copy(),
        "attention": df["attention"].values.copy(),
        "internalizing": df["internalizing"].values.copy(),
        "externalizing": df["externalizing"].values.copy(),
        "p_factor": df["p_factor"].values.copy(),
        "symbolSearch_accuracy": df["symbolSearch_accuracy"].values.copy(),
        "seqLearning_exact": df["seqLearning_accuracy"].values.copy(),
        "seqLearning_partial": df["seq_partial_credit"].values.copy() if "seq_partial_credit" in df.columns else np.full(len(df), np.nan),
    }

    data = {
        "df": df,
        "X_eeg": X_df.values,
        "feature_names": list(X_df.columns),
        "feature_info": feature_info,
        "age": df["age"].values,
        "sex": sex_binary,
        "sex_labels": df["sex"].values,
        "release": df["release"].values,
        "subject_ids": df["subject_id"].values,
        "targets": targets,
        "n_subjects": len(df),
    }
    logger.info("Dataset built: %d subjects, %d EEG features", data["n_subjects"], data["X_eeg"].shape[1])
    return data
