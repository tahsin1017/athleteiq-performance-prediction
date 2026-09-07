# %%
# CELL 1 — Day 11 imports and project paths

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.compose import ColumnTransformer
from sklearn.dummy import DummyRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    r2_score,
)
from sklearn.model_selection import GroupKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder

PROJECT_ROOT = Path(
    "/Users/home/Desktop/AthleteIQ/athleteiq-performance-prediction"
)

DATA_DIR = PROJECT_ROOT / "data" / "processed"
REPORTS_DIR = PROJECT_ROOT / "reports"
FIGURES_DIR = PROJECT_ROOT / "figures"

REPORTS_DIR.mkdir(
    parents=True,
    exist_ok=True,
)

FIGURES_DIR.mkdir(
    parents=True,
    exist_ok=True,
)

print("Project root:", PROJECT_ROOT)
print("Day 11 environment ready.")


# %%
# CELL 2 — Load processed dataset

processed_data_path = (
    DATA_DIR
    / "processed_sleep_health_dataset.csv"
)

data = pd.read_csv(
    processed_data_path
)

print(
    "Dataset shape:",
    data.shape,
)

print(
    "Missing values:",
    int(
        data.isna().sum().sum()
    ),
)

print(
    "Target column present:",
    "Quality of Sleep" in data.columns,
)


# %%
# CELL 3 — Define target and prediction scenarios

TARGET_COLUMN = "Quality of Sleep"

X_full = data.drop(
    columns=[TARGET_COLUMN]
).copy()

y = data[
    TARGET_COLUMN
].copy()

X_pre_sleep = X_full.drop(
    columns=["Sleep Duration"]
).copy()

print(
    "Full feature set shape:",
    X_full.shape,
)

print(
    "Pre-sleep feature set shape:",
    X_pre_sleep.shape,
)

print(
    "\nSleep Duration in full feature set:",
    "Sleep Duration" in X_full.columns,
)

print(
    "Sleep Duration in pre-sleep feature set:",
    "Sleep Duration" in X_pre_sleep.columns,
)

print(
    "\nTarget present in predictors:",
    TARGET_COLUMN in X_full.columns,
)


# %%
# CELL 4 — Build exact predictor-profile groups

full_profile_data = X_full.fillna(
    "__MISSING__"
)

pre_sleep_profile_data = X_pre_sleep.fillna(
    "__MISSING__"
)

full_profile_tuples = pd.Series(
    list(
        map(
            tuple,
            full_profile_data.to_numpy(),
        )
    )
)

pre_sleep_profile_tuples = pd.Series(
    list(
        map(
            tuple,
            pre_sleep_profile_data.to_numpy(),
        )
    )
)

full_groups, full_unique_profiles = pd.factorize(
    full_profile_tuples
)

pre_sleep_groups, pre_sleep_unique_profiles = pd.factorize(
    pre_sleep_profile_tuples
)

profiles_lost = (
    len(full_unique_profiles)
    - len(pre_sleep_unique_profiles)
)

print(
    "Full-feature unique profiles:",
    len(full_unique_profiles),
)

print(
    "Pre-sleep unique profiles:",
    len(pre_sleep_unique_profiles),
)

print(
    "Profiles lost after removing Sleep Duration:",
    profiles_lost,
)


# %%
# CELL 5 — Check target consistency within pre-sleep profiles

pre_sleep_profile_check = (
    X_pre_sleep.copy()
)

pre_sleep_profile_check[
    "Quality of Sleep"
] = y.to_numpy()

pre_sleep_profile_check[
    "Profile Group"
] = pre_sleep_groups

target_values_per_profile = (
    pre_sleep_profile_check
    .groupby(
        "Profile Group"
    )["Quality of Sleep"]
    .nunique()
)

target_count_values = (
    target_values_per_profile
    .to_numpy(
        dtype=int
    )
)

conflicting_pre_sleep_profiles = int(
    np.sum(
        target_count_values > 1
    )
)

print(
    "Pre-sleep unique profiles:",
    len(pre_sleep_unique_profiles),
)

print(
    "Profiles with multiple target values:",
    conflicting_pre_sleep_profiles,
)


# %%
# CELL 6 — Identify numeric and categorical features

full_numeric_features = (
    X_full
    .select_dtypes(
        include=["number"]
    )
    .columns
    .tolist()
)

full_categorical_features = (
    X_full
    .select_dtypes(
        exclude=["number"]
    )
    .columns
    .tolist()
)

pre_sleep_numeric_features = (
    X_pre_sleep
    .select_dtypes(
        include=["number"]
    )
    .columns
    .tolist()
)

pre_sleep_categorical_features = (
    X_pre_sleep
    .select_dtypes(
        exclude=["number"]
    )
    .columns
    .tolist()
)

print(
    "Full numeric features:",
    full_numeric_features,
)

print(
    "\nFull categorical features:",
    full_categorical_features,
)

print(
    "\nPre-sleep numeric features:",
    pre_sleep_numeric_features,
)

print(
    "\nPre-sleep categorical features:",
    pre_sleep_categorical_features,
)


# %%
# CELL 7 — Build preprocessing function

def build_preprocessor(
    numeric_features,
    categorical_features,
):
    return ColumnTransformer(
        transformers=[
            (
                "numeric",
                "passthrough",
                numeric_features,
            ),
            (
                "categorical",
                OneHotEncoder(
                    handle_unknown="ignore"
                ),
                categorical_features,
            ),
        ]
    )


print(
    "Preprocessing builder ready."
)


# %%
# CELL 8 — Build model pipelines

models = {
    "Dummy": DummyRegressor(
        strategy="mean"
    ),
    "Linear": LinearRegression(),
    "Ridge": Ridge(
        alpha=1.0
    ),
    "Random Forest": RandomForestRegressor(
        n_estimators=200,
        random_state=42,
        n_jobs=-1,
    ),
}

full_pipelines = {
    model_name: Pipeline(
        steps=[
            (
                "preprocessor",
                build_preprocessor(
                    full_numeric_features,
                    full_categorical_features,
                ),
            ),
            (
                "model",
                clone(model),
            ),
        ]
    )
    for model_name, model in models.items()
}

pre_sleep_pipelines = {
    model_name: Pipeline(
        steps=[
            (
                "preprocessor",
                build_preprocessor(
                    pre_sleep_numeric_features,
                    pre_sleep_categorical_features,
                ),
            ),
            (
                "model",
                clone(model),
            ),
        ]
    )
    for model_name, model in models.items()
}

print(
    "Full pipelines:",
    list(
        full_pipelines.keys()
    ),
)

print(
    "Pre-sleep pipelines:",
    list(
        pre_sleep_pipelines.keys()
    ),
)


# %%
# CELL 9 — Build group-aware folds and verify zero overlap

group_cv = GroupKFold(
    n_splits=5
)

fold_diagnostics = []

for scenario_name, X_scenario, groups_scenario in [
    (
        "Full",
        X_full,
        full_groups,
    ),
    (
        "Pre-sleep",
        X_pre_sleep,
        pre_sleep_groups,
    ),
]:
    for fold_number, (
        train_indices,
        validation_indices,
    ) in enumerate(
        group_cv.split(
            X_scenario,
            y,
            groups_scenario,
        ),
        start=1,
    ):
        train_profiles = set(
            groups_scenario[
                train_indices
            ]
        )

        validation_profiles = set(
            groups_scenario[
                validation_indices
            ]
        )

        profile_overlap = len(
            train_profiles
            & validation_profiles
        )

        fold_diagnostics.append(
            {
                "Scenario": scenario_name,
                "Fold": fold_number,
                "Train Rows": len(
                    train_indices
                ),
                "Validation Rows": len(
                    validation_indices
                ),
                "Train Profiles": len(
                    train_profiles
                ),
                "Validation Profiles": len(
                    validation_profiles
                ),
                "Profile Overlap": profile_overlap,
            }
        )

fold_diagnostics_df = pd.DataFrame(
    fold_diagnostics
)

print(
    fold_diagnostics_df
)

overlap_values = (
    fold_diagnostics_df[
        "Profile Overlap"
    ]
    .to_numpy(
        dtype=int
    )
)

maximum_profile_overlap = int(
    np.max(
        overlap_values
    )
)

print(
    "\nMaximum profile overlap:",
    maximum_profile_overlap,
)




# %%
# CELL 10 — Evaluate models using scenario-specific groups

evaluation_results = []

scenario_evaluations = [
    (
        "Full",
        X_full,
        full_groups,
        full_pipelines,
    ),
    (
        "Pre-sleep",
        X_pre_sleep,
        pre_sleep_groups,
        pre_sleep_pipelines,
    ),
]

for (
    scenario_name,
    X_scenario,
    groups_scenario,
    pipelines,
) in scenario_evaluations:
    for model_name, pipeline in pipelines.items():
        for fold_number, (
            train_indices,
            validation_indices,
        ) in enumerate(
            group_cv.split(
                X_scenario,
                y,
                groups_scenario,
            ),
            start=1,
        ):
            X_train_fold = X_scenario.iloc[
                train_indices
            ]

            X_validation_fold = X_scenario.iloc[
                validation_indices
            ]

            y_train_fold = y.iloc[
                train_indices
            ]

            y_validation_fold = y.iloc[
                validation_indices
            ]

            fold_pipeline = clone(
                pipeline
            )

            assert isinstance(
                fold_pipeline,
                Pipeline,
            )

            fold_pipeline.fit(
                X_train_fold,
                y_train_fold,
            )

            predictions = (
                fold_pipeline.predict(
                    X_validation_fold
                )
            )

            mae = mean_absolute_error(
                y_validation_fold,
                predictions,
            )

            rmse = float(
                np.sqrt(
                    mean_squared_error(
                        y_validation_fold,
                        predictions,
                    )
                )
            )

            r2 = r2_score(
                y_validation_fold,
                predictions,
            )

            evaluation_results.append(
                {
                    "Scenario": scenario_name,
                    "Model": model_name,
                    "Fold": fold_number,
                    "MAE": float(mae),
                    "RMSE": rmse,
                    "R2": float(r2),
                }
            )

evaluation_results_df = pd.DataFrame(
    evaluation_results
)

print(
    evaluation_results_df
)

print(
    "\nTotal evaluation rows:",
    len(evaluation_results_df),
)




# %%
# CELL 11 — Fair same-fold comparison using pre-sleep groups

shared_folds = list(
    group_cv.split(
        X_pre_sleep,
        y,
        pre_sleep_groups,
    )
)

same_fold_results = []

same_fold_scenarios = [
    (
        "Full",
        X_full,
        full_pipelines,
    ),
    (
        "Pre-sleep",
        X_pre_sleep,
        pre_sleep_pipelines,
    ),
]

for (
    scenario_name,
    X_scenario,
    pipelines,
) in same_fold_scenarios:
    for model_name, pipeline in pipelines.items():
        for fold_number, (
            train_indices,
            validation_indices,
        ) in enumerate(
            shared_folds,
            start=1,
        ):
            X_train_fold = X_scenario.iloc[
                train_indices
            ]

            X_validation_fold = X_scenario.iloc[
                validation_indices
            ]

            y_train_fold = y.iloc[
                train_indices
            ]

            y_validation_fold = y.iloc[
                validation_indices
            ]

            fold_pipeline = clone(
                pipeline
            )

            assert isinstance(
                fold_pipeline,
                Pipeline,
            )

            fold_pipeline.fit(
                X_train_fold,
                y_train_fold,
            )

            predictions = (
                fold_pipeline.predict(
                    X_validation_fold
                )
            )

            mae = mean_absolute_error(
                y_validation_fold,
                predictions,
            )

            rmse = float(
                np.sqrt(
                    mean_squared_error(
                        y_validation_fold,
                        predictions,
                    )
                )
            )

            r2 = r2_score(
                y_validation_fold,
                predictions,
            )

            same_fold_results.append(
                {
                    "Scenario": scenario_name,
                    "Model": model_name,
                    "Fold": fold_number,
                    "MAE": float(mae),
                    "RMSE": rmse,
                    "R2": float(r2),
                }
            )

same_fold_results_df = pd.DataFrame(
    same_fold_results
)

print(
    same_fold_results_df
)

print(
    "\nTotal same-fold evaluation rows:",
    len(same_fold_results_df),
)


# %%
# CELL 12 — Summarize same-fold performance

same_fold_summary_df = pd.DataFrame(
    same_fold_results_df
    .groupby(
        [
            "Scenario",
            "Model",
        ],
        as_index=False,
    )
    .agg(
        Mean_MAE=("MAE", "mean"),
        Std_MAE=("MAE", "std"),
        Mean_RMSE=("RMSE", "mean"),
        Std_RMSE=("RMSE", "std"),
        Mean_R2=("R2", "mean"),
        Std_R2=("R2", "std"),
    )
)

same_fold_summary_df = (
    same_fold_summary_df
    .reset_index(
        drop=True
    )
)

print(
    same_fold_summary_df
)


# %%
# CELL 13 — Quantify performance change after removing Sleep Duration

full_mask = (
    same_fold_summary_df[
        "Scenario"
    ] == "Full"
)

pre_sleep_mask = (
    same_fold_summary_df[
        "Scenario"
    ] == "Pre-sleep"
)

full_summary = pd.DataFrame(
    same_fold_summary_df[
        full_mask
    ]
).copy()

pre_sleep_summary = pd.DataFrame(
    same_fold_summary_df[
        pre_sleep_mask
    ]
).copy()

full_summary = pd.DataFrame(
    full_summary[
        [
            "Model",
            "Mean_MAE",
            "Mean_RMSE",
            "Mean_R2",
        ]
    ]
)

pre_sleep_summary = pd.DataFrame(
    pre_sleep_summary[
        [
            "Model",
            "Mean_MAE",
            "Mean_RMSE",
            "Mean_R2",
        ]
    ]
)

full_summary = full_summary.rename(
    columns={
        "Mean_MAE": "Full_MAE",
        "Mean_RMSE": "Full_RMSE",
        "Mean_R2": "Full_R2",
    }
)

pre_sleep_summary = pre_sleep_summary.rename(
    columns={
        "Mean_MAE": "PreSleep_MAE",
        "Mean_RMSE": "PreSleep_RMSE",
        "Mean_R2": "PreSleep_R2",
    }
)

scenario_comparison_df = pd.merge(
    full_summary,
    pre_sleep_summary,
    on="Model",
    how="inner",
)

scenario_comparison_df[
    "MAE_Change"
] = (
    scenario_comparison_df[
        "PreSleep_MAE"
    ]
    - scenario_comparison_df[
        "Full_MAE"
    ]
)

scenario_comparison_df[
    "RMSE_Change"
] = (
    scenario_comparison_df[
        "PreSleep_RMSE"
    ]
    - scenario_comparison_df[
        "Full_RMSE"
    ]
)

scenario_comparison_df[
    "R2_Change"
] = (
    scenario_comparison_df[
        "PreSleep_R2"
    ]
    - scenario_comparison_df[
        "Full_R2"
    ]
)

print(
    scenario_comparison_df
)


# %%
# CELL 14 — Interpret prediction-scenario comparison

rf_mask = (
    scenario_comparison_df[
        "Model"
    ] == "Random Forest"
)

rf_rows = pd.DataFrame(
    scenario_comparison_df[
        rf_mask
    ]
)

rf_full_r2 = float(
    rf_rows[
        "Full_R2"
    ].to_numpy(
        dtype=float
    )[0]
)

rf_pre_sleep_r2 = float(
    rf_rows[
        "PreSleep_R2"
    ].to_numpy(
        dtype=float
    )[0]
)

rf_r2_change = float(
    rf_rows[
        "R2_Change"
    ].to_numpy(
        dtype=float
    )[0]
)

rf_full_mae = float(
    rf_rows[
        "Full_MAE"
    ].to_numpy(
        dtype=float
    )[0]
)

rf_pre_sleep_mae = float(
    rf_rows[
        "PreSleep_MAE"
    ].to_numpy(
        dtype=float
    )[0]
)

rf_mae_change = float(
    rf_rows[
        "MAE_Change"
    ].to_numpy(
        dtype=float
    )[0]
)

print(
    "DAY 11 INTERPRETATION"
)

print(
    "-" * 50
)

print(
    "Full-feature Random Forest mean R²:",
    round(
        rf_full_r2,
        4,
    ),
)

print(
    "Pre-sleep Random Forest mean R²:",
    round(
        rf_pre_sleep_r2,
        4,
    ),
)

print(
    "R² change after removing Sleep Duration:",
    round(
        rf_r2_change,
        6,
    ),
)

print(
    "MAE change after removing Sleep Duration:",
    round(
        rf_mae_change,
        6,
    ),
)

print(
    "\nUnique full-feature profiles:",
    len(full_unique_profiles),
)

print(
    "Unique pre-sleep profiles:",
    len(pre_sleep_unique_profiles),
)

print(
    "\nInterpretation:"
)

print(
    "Removing same-night Sleep Duration reduced Random Forest "
    "performance, but strong group-aware predictive performance "
    "remained."
)

print(
    "The smaller effect on Linear and Ridge suggests that the "
    "Random Forest relied more heavily on Sleep Duration or on "
    "interactions involving that feature."
)

print(
    "These results support evaluating a deployment-realistic "
    "feature set separately from the full descriptive feature set."
)


# %%
# CELL 15 — Save Day 11 report tables

fold_diagnostics_path = (
    REPORTS_DIR
    / "prediction_scenario_fold_diagnostics.csv"
)

same_fold_results_path = (
    REPORTS_DIR
    / "prediction_scenario_same_fold_results.csv"
)

same_fold_summary_path = (
    REPORTS_DIR
    / "prediction_scenario_same_fold_summary.csv"
)

scenario_comparison_path = (
    REPORTS_DIR
    / "prediction_scenario_comparison.csv"
)

fold_diagnostics_df.to_csv(
    fold_diagnostics_path,
    index=False,
)

same_fold_results_df.to_csv(
    same_fold_results_path,
    index=False,
)

same_fold_summary_df.to_csv(
    same_fold_summary_path,
    index=False,
)

scenario_comparison_df.to_csv(
    scenario_comparison_path,
    index=False,
)

print("Saved:")
print(fold_diagnostics_path.name)
print(same_fold_results_path.name)
print(same_fold_summary_path.name)
print(scenario_comparison_path.name)


# %%
# CELL 16 — Plot Full vs Pre-sleep mean R²

plot_data = pd.DataFrame(
    same_fold_summary_df.pivot(
        index="Model",
        columns="Scenario",
        values="Mean_R2",
    )
)

model_order = [
    "Dummy",
    "Linear",
    "Ridge",
    "Random Forest",
]

plot_data = plot_data.reindex(
    model_order
)

ax = plot_data.plot(
    kind="bar",
    figsize=(10, 6),
)

ax.set_title(
    "Day 11: Full vs Pre-sleep Group-Aware Performance"
)

ax.set_xlabel(
    "Model"
)

ax.set_ylabel(
    "Mean GroupKFold R²"
)

ax.axhline(
    0,
    linewidth=1,
)

ax.legend(
    title="Prediction Scenario"
)

plt.xticks(
    rotation=0
)

plt.tight_layout()

figure_path = (
    FIGURES_DIR
    / "full_vs_pre_sleep_group_cv_r2.png"
)

plt.savefig(
    figure_path,
    dpi=300,
    bbox_inches="tight",
)

plt.show()

print(
    "Saved figure:",
    figure_path.name,
)


# %%
# CELL 17 — Generate Day 11 research journal entry

journal_entry = f"""
## Day 11 — Prediction-Time Feature Scenario Analysis

### Objective
Evaluate whether AthleteIQ can predict `Quality of Sleep` under a more realistic pre-sleep prediction scenario where same-night `Sleep Duration` is unavailable.

### Motivation
The full feature set includes `Sleep Duration`, which may not be available at the time a prospective sleep-quality prediction is made. A model that depends heavily on same-night sleep duration could therefore perform well descriptively while being less useful for true pre-sleep prediction.

### Prediction Scenarios
Two feature scenarios were compared:

1. **Full feature set**
   - 12 predictors
   - Includes `Sleep Duration`

2. **Pre-sleep feature set**
   - 11 predictors
   - Excludes `Sleep Duration`

### Predictor-Profile Structure
- Full-feature unique profiles: {len(full_unique_profiles)}
- Pre-sleep unique profiles: {len(pre_sleep_unique_profiles)}
- Profiles lost after removing Sleep Duration: {profiles_lost}
- Pre-sleep profiles with multiple target values: {conflicting_pre_sleep_profiles}

Removing `Sleep Duration` reduced the number of unique predictor profiles from {len(full_unique_profiles)} to {len(pre_sleep_unique_profiles)}, but no identical pre-sleep predictor profile mapped to multiple target values.

### Validation Design
A 5-fold `GroupKFold` design was used.

For the direct Full vs Pre-sleep comparison, both scenarios were evaluated on the **same folds**, defined using the stricter pre-sleep profile groups.

Maximum exact predictor-profile overlap between training and validation folds was {maximum_profile_overlap}.

All preprocessing was performed inside scikit-learn `Pipeline` objects so one-hot encoding was fit only on each training fold.

### Same-Fold Model Performance

| Scenario | Model | Mean MAE | Mean RMSE | Mean R² | Std R² |
|---|---|---:|---:|---:|---:|
| Full | Random Forest | 0.0865 | 0.2293 | 0.9481 | 0.0432 |
| Full | Ridge | 0.1921 | 0.2912 | 0.9246 | 0.0437 |
| Full | Linear | 0.1913 | 0.3059 | 0.9187 | 0.0401 |
| Pre-sleep | Random Forest | 0.1091 | 0.2923 | 0.9222 | 0.0502 |
| Pre-sleep | Ridge | 0.1935 | 0.2984 | 0.9209 | 0.0466 |
| Pre-sleep | Linear | 0.1955 | 0.3114 | 0.9160 | 0.0410 |

### Random Forest Scenario Change
- Full-feature Mean R²: {rf_full_r2:.4f}
- Pre-sleep Mean R²: {rf_pre_sleep_r2:.4f}
- R² change: {rf_r2_change:.6f}

- Full-feature Mean MAE: {rf_full_mae:.4f}
- Pre-sleep Mean MAE: {rf_pre_sleep_mae:.4f}
- MAE change: {rf_mae_change:.6f}

Removing same-night `Sleep Duration` reduced Random Forest performance, with Mean R² decreasing by approximately {abs(rf_r2_change):.4f}.

However, the pre-sleep Random Forest still achieved Mean R² above 0.92 under profile-separated validation.

Linear Regression and Ridge were much less affected by removing `Sleep Duration`, suggesting that the Random Forest relied more strongly on this feature or on nonlinear interactions involving it.

### Interpretation
This experiment separates descriptive modeling from a more realistic prospective prediction scenario.

The full feature model remains useful for understanding relationships in the dataset, while the pre-sleep feature set is more defensible for a model intended to make predictions before the sleep episode occurs.

The relatively modest decline in performance suggests that predictive signal remains in variables such as stress, physical activity, age, cardiovascular measures, occupation, BMI category, and sleep-disorder status.

However, high predictive performance should not be interpreted as evidence of external generalization.

### Limitations
- The dataset contains only 374 rows.
- The pre-sleep feature set contains only {len(pre_sleep_unique_profiles)} unique predictor profiles.
- Exact-profile grouping reduces contamination but does not simulate genuinely unseen people or external populations.
- Repeated rows still give some profiles greater weight than others.
- `Sleep Disorder` contains missing values that are treated as a category by the one-hot encoder.
- Some predictors may themselves be unavailable or impractical at real prediction time depending on the intended deployment setting.
- No external validation dataset was used.
- These results do not establish clinical readiness.

### Saved Outputs
Reports:
- `reports/prediction_scenario_fold_diagnostics.csv`
- `reports/prediction_scenario_same_fold_results.csv`
- `reports/prediction_scenario_same_fold_summary.csv`
- `reports/prediction_scenario_comparison.csv`

Figure:
- `figures/full_vs_pre_sleep_group_cv_r2.png`

### Next Step
The next analysis should test a stricter deployment-oriented feature set by questioning whether variables such as `Sleep Disorder`, blood pressure, and other contemporaneous measurements would truly be available at prediction time.

Repeated group-aware evaluation can then be used to assess how stable the deployment-realistic model is across different group-separated splits.
"""

print(
    journal_entry
)


# %%
# CELL 18 — Final Day 11 verification

expected_report_files = [
    "prediction_scenario_fold_diagnostics.csv",
    "prediction_scenario_same_fold_results.csv",
    "prediction_scenario_same_fold_summary.csv",
    "prediction_scenario_comparison.csv",
]

expected_figure_file = (
    "full_vs_pre_sleep_group_cv_r2.png"
)

print(
    "DAY 11 FINAL VERIFICATION"
)

print(
    "-" * 50
)

print(
    "Dataset rows:",
    len(data),
)

print(
    "Full-feature profiles:",
    len(full_unique_profiles),
)

print(
    "Pre-sleep profiles:",
    len(pre_sleep_unique_profiles),
)

print(
    "Maximum profile overlap:",
    maximum_profile_overlap,
)

print(
    "Random Forest full R²:",
    round(
        rf_full_r2,
        4,
    ),
)

print(
    "Random Forest pre-sleep R²:",
    round(
        rf_pre_sleep_r2,
        4,
    ),
)

print(
    "Random Forest R² change:",
    round(
        rf_r2_change,
        6,
    ),
)

print(
    "\nReport files:"
)

for filename in expected_report_files:
    file_exists = (
        REPORTS_DIR
        / filename
    ).exists()

    print(
        filename,
        "->",
        file_exists,
    )

print(
    "\nFigure file:",
    expected_figure_file,
    "->",
    (
        FIGURES_DIR
        / expected_figure_file
    ).exists(),
)

print(
    "\nDay 11 analysis completed successfully."
)
