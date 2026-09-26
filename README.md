# AthleteIQ — Sleep Quality Prediction and Evaluation Audit

A small tabular sleep/lifestyle dataset produced unusually strong machine-learning results under conventional random splitting.

Rather than treating the high score as the main result, this project investigates **why the apparent performance was so strong**, how repeated predictor profiles affect validation, and how estimates change under stricter evaluation and deployment-oriented feature assumptions.

The main finding is methodological:

> Random row-wise validation can make a small tabular prediction problem appear easier when many observations share predictor profiles already represented in training.

This repository documents the investigation from the original high-performing models through profile-grouped validation, train-only preprocessing, prediction-time feature restrictions, repeated group holdouts, and Ridge coefficient stability analysis.

---

## Problem

The task is to predict the dataset variable:

`Quality of Sleep`

from a small sleep and lifestyle dataset containing demographic, behavioral, and physiological variables.

The repository name **AthleteIQ** is a project name. The dataset itself does **not** establish that the observations represent athletes, so the results should not be interpreted as athlete-specific evidence.

---

## Why the Initial Result Looked Suspicious

Early experiments produced very high performance.

For example:

| Evaluation | Model | R² |
|---|---|---:|
| Holdout split | Random Forest | 0.988 |
| Random 5-fold CV | Random Forest | 0.980 |

Those values motivated a closer inspection rather than immediate model tuning.

The dataset contains:

- 374 rows
- only 132 unique profiles when all predictors are considered
- 320 of 374 rows belonging to repeated full-predictor profiles

In one random train/test split, **65 of 75 test rows had an exact predictor profile already present in the training set**.

No target conflicts were found within identical full-predictor profiles.

This does not prove participant duplication or data leakage in the strict sense. It shows that conventional row-wise splitting often evaluates the model on predictor combinations already represented in training.

---

## Group-Aware Evaluation

To reduce exact-profile overlap between training and validation data, identical predictor profiles were assigned to the same group.

Using `GroupKFold`, train/validation profile overlap was zero.

Selected results:

| Evaluation | Model | Mean R² |
|---|---|---:|
| Random 5-fold CV | Random Forest | 0.980 |
| Full-profile GroupKFold | Random Forest | 0.963 |
| Pre-sleep GroupKFold | Random Forest | 0.922 |
| Deployment-feature GroupKFold | Random Forest | 0.915 |
| Deployment-feature GroupKFold | Ridge | 0.915 |

The exact folds differ across some experiments because increasingly strict feature sets define different predictor-profile groups. These numbers therefore illustrate the evaluation progression rather than a single controlled model ranking.

---

## Deployment-Oriented Evaluation

The current deployment-oriented feature set contains:

- Gender
- Age
- Occupation
- Physical Activity Level
- Stress Level
- BMI Category
- Heart Rate
- Daily Steps

The following variables were excluded from the deployment-oriented scenario:

- Sleep Duration
- Sleep Disorder
- Systolic BP
- Diastolic BP

Under 20 repeated deployment-profile holdouts:

| Model | Mean R² | R² Std |
|---|---:|---:|
| Ridge | 0.904 | 0.056 |
| Linear Regression | 0.892 | 0.072 |
| Random Forest | 0.887 | 0.082 |
| Dummy Regressor | -0.194 | 0.202 |

Ridge and Random Forest each achieved the highest R² in 9 of the 20 repeated holdouts, while Linear Regression did so in 2.

The goal of this experiment was not to declare a universal winner, but to examine stability under stricter profile separation.

---

## Repeated Predictor Profiles Are Not Participant IDs

The groups used in this repository are constructed from identical combinations of predictor values.

They are therefore best described as:

**repeated predictor profiles**

or:

**identical-feature profiles**

They are **not verified participant identities**.

Group-aware validation in this project tests generalization to unseen predictor profiles. It does not establish generalization to unseen individuals.

This distinction is important and remains a limitation of the dataset.

---

## Ridge Interpretation

A standardized, reference-coded Ridge specification was used to examine coefficient stability across profile-separated folds.

Across 18 transformed features:

- 15 had stable coefficient signs
- Stress Level showed a strong, consistently negative coefficient
- Age showed a consistently positive coefficient
- Heart Rate and Physical Activity Level had smaller but sign-stable coefficients
- Daily Steps had a small, sign-unstable coefficient
- a small number of occupation contrasts were also sign-unstable

These coefficients describe conditional model associations.

They should **not** be interpreted as causal effects or as evidence that modifying a variable will improve an individual's sleep.

The interpretation-oriented Ridge pipeline differs from the earlier Day 12 predictive Ridge specification, so its predictive performance is being re-evaluated before final model selection.

---

## Evaluation Progression

The project intentionally evolved as evidence accumulated:

1. Exploratory analysis
2. Baseline models and dummy regression
3. Random holdout and random cross-validation
4. Feature importance and suspicious-result investigation
5. Repeated-profile analysis
6. Profile-grouped validation
7. Train-only preprocessing pipelines
8. Prediction-time feature scenarios
9. Deployment-oriented feature restriction
10. Repeated group holdouts
11. Ridge coefficient stability analysis
12. Reproducibility and methodological audit

The research journal records the reasoning and corrections made during this process.

See:

`docs/research_journal.md`

---

## Repository Structure

```text
athleteiq-performance-prediction/
├── data/
│   ├── raw/
│   ├── processed/
│   └── README.md
├── docs/
│   └── research_journal.md
├── figures/
├── notebooks/
├── reports/
├── environment.yml
├── requirements.txt
└── README.md
```

---

## Reproducibility

The project was developed with Python 3.11.

### Conda

```bash
conda env create -f environment.yml
conda activate athleteiq-performance
```

### pip

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

A single command reproducing the main comparison table will be added as part of the final reproducibility cleanup.

---

## Data Provenance

This project uses the **Sleep Health and Lifestyle Dataset** created by **Laksika Tharmalingam** and distributed through Kaggle.

Dataset page:

https://www.kaggle.com/datasets/uom190346a/sleep-health-and-lifestyle-dataset

Additional provenance and licensing notes are documented in:

`data/README.md`

No claim is made that the third-party dataset is licensed under any license that may later be applied to original code in this repository.

---

## Limitations

Important limitations include:

- only 374 observations
- a small number of unique predictor profiles
- repeated predictor profiles are not verified participant identities
- no external validation dataset
- no confirmed unseen-person evaluation
- the dataset does not establish an athlete population
- some variables may not be available at the intended prediction time
- model associations are not causal effects
- results from one small tabular dataset should not be generalized broadly

---

## Current Research Question

The project has evolved from:

> Which model predicts sleep quality best?

toward the more methodological question:

> How do validation assumptions and repeated predictor profiles affect apparent model performance in a small tabular prediction problem?

A broader research study would require additional datasets and a predefined multi-dataset evaluation protocol.

---

## Project Status

The project is currently in the final methodology and reproducibility phase.

Remaining work includes:

- controlled comparison of the Day 12 and Day 13 Ridge specifications
- feature ablation for weak deployment predictors
- final feature specification
- modest Ridge hyperparameter tuning
- one-command experiment reproduction
- consolidated evaluation figure/table
- final repository cleanup and release
