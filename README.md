# EEG-Based Neurodevelopmental Age Estimation

**Final-Year Computer Engineering Project**

Analysis of cognitive performance and behavioral symptoms in children and adolescents using resting-state EEG and machine learning.

---

## Overview

This project trains machine learning models to predict chronological age from resting-state EEG features (brain-age estimation) and investigates whether deviations from typical brain maturity associate with cognitive performance or behavioral symptoms.

**Dataset**: HBN-EEG (Healthy Brain Network, Child Mind Institute) — 1,826 subjects, ages 5–22, 11 data releases  
**EEG**: 129-channel resting-state recordings → 1,083 features after preprocessing  
**Models**: Mean Baseline, Ridge Regression, Random Forest (300 trees)  
**Validation**: 10-fold KFold + GroupKFold by release

---

## Key Results

| Experiment | Result |
|---|---|
| Age Prediction (RF) | R² = 0.498, MAE = 1.82 years |
| GroupKFold Generalization | R² = 0.494 (robust across releases) |
| Brain-Age Gap vs Behavior | No significant associations (all p > 0.05 after Bonferroni) |
| Cognitive Prediction (best) | R² = 0.081 (Age+Sex only — EEG adds nothing) |
| Strongest age-EEG feature | Theta/alpha ratio (mean \|r\| = 0.280) |
| Top RF feature family | Delta power (permutation importance sum = 0.331) |

---

## Project Structure

```
neurodevelopment_project/
├── run_pipeline.py          # Entry point — runs all 5 experiments
├── requirements.txt
├── src/
│   ├── data_loader.py       # Feature loading, log-transform, partial credit scoring
│   ├── age_prediction.py    # Experiment 1: CV age prediction
│   ├── brain_age_analysis.py # Experiment 2: brain-age gap + behavioral associations
│   ├── interpretability.py  # Experiment 3: Gini + permutation feature importance
│   ├── cognitive_prediction.py # Experiment 4: cognitive performance prediction
│   ├── developmental_analysis.py # Experiment 5: age trends + gap-cognition
│   └── visualization.py     # All figure generation
├── pipeline.ipynb           # Reproducible notebook (runs everything)
├── results.ipynb            # Presentation notebook (loads pre-saved outputs)
└── results/
    ├── figures/             # 10 PNG figures
    ├── metrics/             # Model performance CSVs
    ├── tables/              # Summary tables
    ├── predictions/         # Per-subject prediction CSVs
    └── reports/             # Pipeline summary and JSON results
```

---

## Setup

```bash
pip install -r requirements.txt
```

**Data dependency**: Requires `../data/processed/feature_matrix.csv` from the HBN-EEG dataset (OpenNeuro ds005507). Raw EEG event files needed only for Sequential Learning partial-credit re-scoring.

---

## Running the Pipeline

```bash
# Full pipeline (all 5 experiments + figures, ~7 minutes)
python run_pipeline.py

# Specific experiments only
python run_pipeline.py --exp 1 2 3

# Skip figure generation
python run_pipeline.py --skip-viz
```

Logs are written to `logs/pipeline.log`.

---

## Experiments

1. **Age Prediction** — Predict chronological age from 1,083 EEG features using 10-fold CV
2. **Brain-Age Gap** — Compute predicted−actual age gap; test behavioral associations after age-bias correction
3. **Feature Importance** — Permutation and Gini importance; feature-age correlations by family
4. **Cognitive Prediction** — Predict Symbol Search and Sequential Learning performance from EEG
5. **Developmental Analysis** — EEG and cognitive trends across age groups; brain-age gap vs cognition

---

## EEG Feature Families

| Family | Features | Age Correlation |
|--------|----------|----------------|
| Band Power (δ θ α β γ) | 645 | mean \|r\| = 0.011 |
| Theta/Alpha Ratio | 130 | mean \|r\| = 0.280 ★ |
| Spectral Entropy | 129 | mean \|r\| = 0.157 |
| Sample Entropy | 129 | mean \|r\| = 0.171 |
| Alpha Coherence | 50 | mean \|r\| = 0.064 |

★ Strongest individual linear age signal; delta power dominates nonlinear RF importance.

---

## Important Notes

- **Cross-sectional study**: Age differences reflect between-subject variation, not within-person trajectories
- **Clinical-community sample**: HBN includes children with and without diagnoses — results may not generalize to neurotypical populations
- A positive brain-age gap does **not** imply accelerated aging or atypical development
