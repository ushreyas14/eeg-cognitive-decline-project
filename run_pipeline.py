"""
run_pipeline.py — Run the complete neurodevelopment ML project pipeline.

Usage:
    python run_pipeline.py              # Run all experiments
    python run_pipeline.py --exp 1 3    # Run only experiments 1 and 3
    python run_pipeline.py --skip-viz   # Skip figure generation

All results are saved to results/ and models/.
"""

import argparse
import logging
import os
import sys
import time
import json
import traceback

PROJECT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, PROJECT_DIR)

RESULTS_DIR = os.path.join(PROJECT_DIR, "results")
MODELS_DIR = os.path.join(PROJECT_DIR, "models")
LOGS_DIR = os.path.join(PROJECT_DIR, "logs")


def setup_logging():
    os.makedirs(LOGS_DIR, exist_ok=True)
    log_file = os.path.join(LOGS_DIR, "pipeline.log")

    root = logging.getLogger()
    root.setLevel(logging.INFO)

    fmt = logging.Formatter(
        "%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    fh = logging.FileHandler(log_file, mode="w", encoding="utf-8")
    fh.setLevel(logging.INFO)
    fh.setFormatter(fmt)
    root.addHandler(fh)

    ch = logging.StreamHandler(sys.stdout)
    ch.setLevel(logging.INFO)
    ch.setFormatter(fmt)
    root.addHandler(ch)

    return logging.getLogger("pipeline")


def print_banner(logger):
    logger.info("=" * 70)
    logger.info("  NEURODEVELOPMENTAL AGE ESTIMATION & ANALYSIS PIPELINE")
    logger.info("  EEG-Based Age Prediction | Brain-Age Gap | Cognitive Analysis")
    logger.info("=" * 70)


def print_step(logger, step_num, total, description):
    logger.info("")
    logger.info("-" * 60)
    logger.info("  STEP %d/%d: %s", step_num, total, description)
    logger.info("-" * 60)


def run_pipeline(args):
    logger = setup_logging()
    print_banner(logger)
    pipeline_start = time.time()

    run_exps = set(args.exp) if args.exp else {1, 2, 3, 4, 5}
    total_steps = len(run_exps) + (0 if args.skip_viz else 1) + 2  # +2 for data loading and report

    logger.info("Experiments to run: %s", sorted(run_exps))
    logger.info("Results directory: %s", RESULTS_DIR)
    logger.info("")

    # Ensure output directories
    for subdir in ["metrics", "predictions", "tables", "figures", "reports", "models"]:
        os.makedirs(os.path.join(RESULTS_DIR, subdir), exist_ok=True)
    os.makedirs(MODELS_DIR, exist_ok=True)

    step = 0
    results = {}

    # ── Step: Load Data ──────────────────────────────────────────────
    step += 1
    print_step(logger, step, total_steps, "LOAD AND PREPARE DATA")
    t0 = time.time()
    try:
        from src.data_loader import build_dataset
        data = build_dataset()
        logger.info("Data loaded in %.1f seconds", time.time() - t0)
        logger.info("  Subjects: %d", data["n_subjects"])
        logger.info("  EEG features: %d", data["X_eeg"].shape[1])
        logger.info("  Age range: %.1f–%.1f years", data["targets"]["age"].min(), data["targets"]["age"].max())
        logger.info("  seqLearning partial credit: %d available",
                     (~np.isnan(data["targets"]["seqLearning_partial"])).sum())
        logger.info("  symbolSearch accuracy: %d available",
                     (~np.isnan(data["targets"]["symbolSearch_accuracy"])).sum())
    except Exception as e:
        logger.error("FAILED to load data: %s", e)
        traceback.print_exc()
        sys.exit(1)

    # ── Experiment 1: Age Prediction ─────────────────────────────────
    exp1_results = None
    if 1 in run_exps:
        step += 1
        print_step(logger, step, total_steps, "EXPERIMENT 1 — EEG Age Prediction")
        t0 = time.time()
        try:
            from src.age_prediction import run_age_prediction
            exp1_results = run_age_prediction(data, RESULTS_DIR)
            results["exp1"] = {"status": "success", "time": exp1_results["total_time"]}
            logger.info("✓ Experiment 1 complete (%.1fs)", time.time() - t0)
        except Exception as e:
            logger.error("✗ Experiment 1 FAILED: %s", e)
            traceback.print_exc()
            results["exp1"] = {"status": "failed", "error": str(e)}

    # ── Experiment 2: Brain-Age Gap ──────────────────────────────────
    exp2_results = None
    if 2 in run_exps:
        step += 1
        print_step(logger, step, total_steps, "EXPERIMENT 2 — Brain-Age Gap Analysis")
        if exp1_results is None:
            logger.warning("Experiment 1 results needed for Experiment 2; skipping")
            results["exp2"] = {"status": "skipped", "reason": "exp1 not available"}
        else:
            t0 = time.time()
            try:
                from src.brain_age_analysis import run_brain_age_analysis
                exp2_results = run_brain_age_analysis(data, exp1_results, RESULTS_DIR)
                results["exp2"] = {"status": "success", "time": exp2_results["total_time"]}
                logger.info("✓ Experiment 2 complete (%.1fs)", time.time() - t0)
            except Exception as e:
                logger.error("✗ Experiment 2 FAILED: %s", e)
                traceback.print_exc()
                results["exp2"] = {"status": "failed", "error": str(e)}

    # ── Experiment 3: Interpretability ───────────────────────────────
    exp3_results = None
    if 3 in run_exps:
        step += 1
        print_step(logger, step, total_steps, "EXPERIMENT 3 — Feature Importance")
        if exp1_results is None:
            logger.warning("Experiment 1 results needed for Experiment 3; skipping")
            results["exp3"] = {"status": "skipped", "reason": "exp1 not available"}
        else:
            t0 = time.time()
            try:
                from src.interpretability import run_interpretability
                exp3_results = run_interpretability(data, exp1_results, RESULTS_DIR)
                results["exp3"] = {"status": "success", "time": exp3_results["total_time"]}
                logger.info("✓ Experiment 3 complete (%.1fs)", time.time() - t0)
            except Exception as e:
                logger.error("✗ Experiment 3 FAILED: %s", e)
                traceback.print_exc()
                results["exp3"] = {"status": "failed", "error": str(e)}

    # ── Experiment 4: Cognitive Prediction ───────────────────────────
    exp4_results = None
    if 4 in run_exps:
        step += 1
        print_step(logger, step, total_steps, "EXPERIMENT 4 — Cognitive Prediction")
        t0 = time.time()
        try:
            from src.cognitive_prediction import run_cognitive_prediction
            exp4_results = run_cognitive_prediction(data, RESULTS_DIR)
            results["exp4"] = {"status": "success", "time": exp4_results["total_time"]}
            logger.info("✓ Experiment 4 complete (%.1fs)", time.time() - t0)
        except Exception as e:
            logger.error("✗ Experiment 4 FAILED: %s", e)
            traceback.print_exc()
            results["exp4"] = {"status": "failed", "error": str(e)}

    # ── Experiment 5: Developmental Analysis ─────────────────────────
    exp5_results = None
    if 5 in run_exps:
        step += 1
        print_step(logger, step, total_steps, "EXPERIMENT 5 — Developmental Analysis")
        t0 = time.time()
        try:
            from src.developmental_analysis import run_developmental_analysis
            exp5_results = run_developmental_analysis(data, exp2_results, RESULTS_DIR)
            results["exp5"] = {"status": "success", "time": exp5_results["total_time"]}
            logger.info("✓ Experiment 5 complete (%.1fs)", time.time() - t0)
        except Exception as e:
            logger.error("✗ Experiment 5 FAILED: %s", e)
            traceback.print_exc()
            results["exp5"] = {"status": "failed", "error": str(e)}

    # ── Figure Generation ────────────────────────────────────────────
    if not args.skip_viz:
        step += 1
        print_step(logger, step, total_steps, "GENERATE FIGURES")
        t0 = time.time()
        try:
            from src.visualization import generate_all_figures
            generate_all_figures(exp1_results, exp2_results, exp3_results,
                                 exp4_results, exp5_results, RESULTS_DIR)
            results["figures"] = {"status": "success", "time": time.time() - t0}
            logger.info("✓ Figures generated (%.1fs)", time.time() - t0)
        except Exception as e:
            logger.error("✗ Figure generation FAILED: %s", e)
            traceback.print_exc()
            results["figures"] = {"status": "failed", "error": str(e)}

    # ── Final Report ─────────────────────────────────────────────────
    step += 1
    print_step(logger, step, total_steps, "GENERATE SUMMARY REPORT")
    total_time = time.time() - pipeline_start

    report_lines = [
        "# Pipeline Execution Summary",
        f"Total time: {total_time:.1f} seconds ({total_time/60:.1f} minutes)",
        "",
    ]

    for key in ["exp1", "exp2", "exp3", "exp4", "exp5", "figures"]:
        if key in results:
            r = results[key]
            status = r["status"]
            t = r.get("time", "")
            extra = f" ({t:.1f}s)" if isinstance(t, (int, float)) else ""
            if status == "failed":
                extra = f" — {r.get('error', 'unknown')}"
            report_lines.append(f"- {key}: {status}{extra}")

    report_lines.append("")

    # Key metrics summary
    if exp1_results:
        best = exp1_results["all_results"].get("kfold_Random Forest", {})
        om = best.get("overall_metrics", {})
        report_lines.append("## Key Results")
        report_lines.append(f"- Age prediction (RF, KFold): R²={om.get('R2', 'N/A'):.4f}, "
                             f"MAE={om.get('MAE', 'N/A'):.3f} yr")

    if exp2_results:
        report_lines.append(f"- Brain-age gap bias correction: r={exp2_results['age_bias_r_before']:.4f} → "
                             f"{exp2_results['age_bias_r_after']:.4f}")

    if exp4_results and len(exp4_results.get("metrics", [])) > 0:
        for _, row in exp4_results["metrics"].iterrows():
            if row["model"] != "Mean Baseline":
                report_lines.append(
                    f"- {row['target']} ({row['feature_set']}/{row['model']}): R²={row['R2']:.4f}"
                )

    report_text = "\n".join(report_lines)
    report_path = os.path.join(RESULTS_DIR, "reports", "pipeline_summary.md")
    with open(report_path, "w", encoding="utf-8") as f:
        f.write(report_text)

    # Save machine-readable results
    json_path = os.path.join(RESULTS_DIR, "reports", "pipeline_results.json")
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, default=str)

    logger.info("\n" + report_text)
    logger.info("\n" + "=" * 70)
    logger.info("  PIPELINE COMPLETE — %.1f seconds (%.1f minutes)", total_time, total_time / 60)
    logger.info("  Results: %s", RESULTS_DIR)
    logger.info("  Log: %s", os.path.join(LOGS_DIR, "pipeline.log"))
    logger.info("=" * 70)

    # Return exit code based on results
    failures = [k for k, v in results.items() if v.get("status") == "failed"]
    if failures:
        logger.warning("Failed steps: %s", failures)
        return 1
    return 0


if __name__ == "__main__":
    import numpy as np  # needed in the main scope for data checks

    parser = argparse.ArgumentParser(description="Run the neurodevelopment ML pipeline")
    parser.add_argument("--exp", type=int, nargs="+", choices=[1, 2, 3, 4, 5],
                        help="Run only specific experiments (default: all)")
    parser.add_argument("--skip-viz", action="store_true",
                        help="Skip figure generation")
    args = parser.parse_args()

    exit_code = run_pipeline(args)
    sys.exit(exit_code)
