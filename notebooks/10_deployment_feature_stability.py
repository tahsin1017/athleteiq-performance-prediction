# %%
# CELL 1 — Day 12 imports and project paths

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
from sklearn.model_selection import (
    GroupKFold,
    GroupShuffleSplit,
)
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
print("Day 12 environment ready.")



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
    "Columns:",
    data.columns.tolist(),
)

print(
    "Missing values:",
    int(
        data.isna().sum().sum()
    ),
)

print(
    "Target present:",
    "Quality of Sleep" in data.columns,
)



# %%
# CELL 3 — Define target and feature scenarios

TARGET_COLUMN = "Quality of Sleep"

X_full = data.drop(
    columns=[TARGET_COLUMN]
).copy()

y = data[
    TARGET_COLUMN
].copy()

X_pre_sleep = X_full.drop(
    columns=[
        "Sleep Duration",
    ]
).copy()

X_deployment = X_full.drop(
    columns=[
        "Sleep Duration",
        "Sleep Disorder",
        "Systolic BP",
        "Diastolic BP",
    ]
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
    "Deployment feature set shape:",
    X_deployment.shape,
)

print(
    "\nDeployment features:",
    X_deployment.columns.tolist(),
)

print(
    "\nTarget present in predictors:",
    TARGET_COLUMN in X_deployment.columns,
)


# %%
# CELL 4 — Build exact predictor-profile groups

full_profile_data = X_full.fillna(
    "__MISSING__"
)

pre_sleep_profile_data = X_pre_sleep.fillna(
    "__MISSING__"
)

deployment_profile_data = X_deployment.fillna(
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

deployment_profile_tuples = pd.Series(
    list(
        map(
            tuple,
            deployment_profile_data.to_numpy(),
        )
    )
)

full_groups, full_unique_profiles = pd.factorize(
    full_profile_tuples
)

pre_sleep_groups, pre_sleep_unique_profiles = pd.factorize(
    pre_sleep_profile_tuples
)

deployment_groups, deployment_unique_profiles = pd.factorize(
    deployment_profile_tuples
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
    "Deployment unique profiles:",
    len(deployment_unique_profiles),
)

print(
    "\nProfiles lost: Full -> Pre-sleep:",
    (
        len(full_unique_profiles)
        - len(pre_sleep_unique_profiles)
    ),
)

print(
    "Profiles lost: Pre-sleep -> Deployment:",
    (
        len(pre_sleep_unique_profiles)
        - len(deployment_unique_profiles)
    ),
)



# %%
# CELL 5 — Check target consistency within deployment profiles

deployment_profile_check = (
    X_deployment.copy()
)

deployment_profile_check[
    "Quality of Sleep"
] = y.to_numpy()

deployment_profile_check[
    "Profile Group"
] = deployment_groups

target_values_per_profile = (
    deployment_profile_check
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

conflicting_deployment_profiles = int(
    np.sum(
        target_count_values > 1
    )
)

print(
    "Deployment unique profiles:",
    len(deployment_unique_profiles),
)

print(
    "Profiles with multiple target values:",
    conflicting_deployment_profiles,
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

deployment_numeric_features = (
    X_deployment
    .select_dtypes(
        include=["number"]
    )
    .columns
    .tolist()
)

deployment_categorical_features = (
    X_deployment
    .select_dtypes(
        exclude=["number"]
    )
    .columns
    .tolist()
)

print(
    "Deployment numeric features:",
    deployment_numeric_features,
)

print(
    "\nDeployment categorical features:",
    deployment_categorical_features,
)

print(
    "\nDeployment feature count:",
    (
        len(deployment_numeric_features)
        + len(deployment_categorical_features)
    ),
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

deployment_pipelines = {
    model_name: Pipeline(
        steps=[
            (
                "preprocessor",
                build_preprocessor(
                    deployment_numeric_features,
                    deployment_categorical_features,
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
    list(full_pipelines.keys()),
)

print(
    "Pre-sleep pipelines:",
    list(pre_sleep_pipelines.keys()),
)

print(
    "Deployment pipelines:",
    list(deployment_pipelines.keys()),
)



# %%
# CELL 9 — Build shared deployment-group folds

group_cv = GroupKFold(
    n_splits=5
)

shared_folds = list(
    group_cv.split(
        X_deployment,
        y,
        deployment_groups,
    )
)

fold_diagnostics = []

for fold_number, (
    train_indices,
    validation_indices,
) in enumerate(
    shared_folds,
    start=1,
):
    train_profiles = set(
        deployment_groups[
            train_indices
        ]
    )

    validation_profiles = set(
        deployment_groups[
            validation_indices
        ]
    )

    profile_overlap = len(
        train_profiles
        & validation_profiles
    )

    fold_diagnostics.append(
        {
            "Fold": fold_number,
            "Train Rows": len(
                train_indices
            ),
            "Validation Rows": len(
                validation_indices
            ),
            "Train Deployment Profiles": len(
                train_profiles
            ),
            "Validation Deployment Profiles": len(
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
    "\nMaximum deployment-profile overlap:",
    maximum_profile_overlap,
)



# %%
# CELL 10 — Same-fold comparison across all three scenarios

same_fold_results = []

comparison_scenarios = [
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
    (
        "Deployment",
        X_deployment,
        deployment_pipelines,
    ),
]

for (
    scenario_name,
    X_scenario,
    pipelines,
) in comparison_scenarios:
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
    "\nTotal evaluation rows:",
    len(same_fold_results_df),
)



# %%
# CELL 11 — Summarize same-fold performance

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
# CELL 12 — Compare performance changes across scenarios

full_summary = pd.DataFrame(
    same_fold_summary_df[
        same_fold_summary_df[
            "Scenario"
        ] == "Full"
    ]
).copy()

pre_sleep_summary = pd.DataFrame(
    same_fold_summary_df[
        same_fold_summary_df[
            "Scenario"
        ] == "Pre-sleep"
    ]
).copy()

deployment_summary = pd.DataFrame(
    same_fold_summary_df[
        same_fold_summary_df[
            "Scenario"
        ] == "Deployment"
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
).rename(
    columns={
        "Mean_MAE": "Full_MAE",
        "Mean_RMSE": "Full_RMSE",
        "Mean_R2": "Full_R2",
    }
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
).rename(
    columns={
        "Mean_MAE": "PreSleep_MAE",
        "Mean_RMSE": "PreSleep_RMSE",
        "Mean_R2": "PreSleep_R2",
    }
)

deployment_summary = pd.DataFrame(
    deployment_summary[
        [
            "Model",
            "Mean_MAE",
            "Mean_RMSE",
            "Mean_R2",
        ]
    ]
).rename(
    columns={
        "Mean_MAE": "Deployment_MAE",
        "Mean_RMSE": "Deployment_RMSE",
        "Mean_R2": "Deployment_R2",
    }
)

scenario_comparison_df = pd.merge(
    full_summary,
    pre_sleep_summary,
    on="Model",
    how="inner",
)

scenario_comparison_df = pd.merge(
    scenario_comparison_df,
    deployment_summary,
    on="Model",
    how="inner",
)

scenario_comparison_df[
    "PreSleep_R2_Change"
] = (
    scenario_comparison_df[
        "PreSleep_R2"
    ]
    - scenario_comparison_df[
        "Full_R2"
    ]
)

scenario_comparison_df[
    "Deployment_R2_Change"
] = (
    scenario_comparison_df[
        "Deployment_R2"
    ]
    - scenario_comparison_df[
        "Full_R2"
    ]
)

scenario_comparison_df[
    "Deployment_vs_PreSleep_R2_Change"
] = (
    scenario_comparison_df[
        "Deployment_R2"
    ]
    - scenario_comparison_df[
        "PreSleep_R2"
    ]
)

print(
    scenario_comparison_df
)




# %%
# CELL 13 — Repeated group-separated deployment evaluation

repeated_splitter = GroupShuffleSplit(
    n_splits=20,
    test_size=0.20,
    random_state=42,
)

repeated_results = []

for split_number, (
    train_indices,
    test_indices,
) in enumerate(
    repeated_splitter.split(
        X_deployment,
        y,
        deployment_groups,
    ),
    start=1,
):
    train_profiles = set(
        deployment_groups[
            train_indices
        ]
    )

    test_profiles = set(
        deployment_groups[
            test_indices
        ]
    )

    profile_overlap = len(
        train_profiles
        & test_profiles
    )

    X_train = X_deployment.iloc[
        train_indices
    ]

    X_test = X_deployment.iloc[
        test_indices
    ]

    y_train = y.iloc[
        train_indices
    ]

    y_test = y.iloc[
        test_indices
    ]

    for model_name, pipeline in deployment_pipelines.items():
        fold_pipeline = clone(
            pipeline
        )

        assert isinstance(
            fold_pipeline,
            Pipeline,
        )

        fold_pipeline.fit(
            X_train,
            y_train,
        )

        predictions = (
            fold_pipeline.predict(
                X_test
            )
        )

        mae = mean_absolute_error(
            y_test,
            predictions,
        )

        rmse = float(
            np.sqrt(
                mean_squared_error(
                    y_test,
                    predictions,
                )
            )
        )

        r2 = r2_score(
            y_test,
            predictions,
        )

        repeated_results.append(
            {
                "Split": split_number,
                "Model": model_name,
                "Train Rows": len(
                    train_indices
                ),
                "Test Rows": len(
                    test_indices
                ),
                "Train Profiles": len(
                    train_profiles
                ),
                "Test Profiles": len(
                    test_profiles
                ),
                "Profile Overlap": profile_overlap,
                "MAE": float(mae),
                "RMSE": rmse,
                "R2": float(r2),
            }
        )

repeated_results_df = pd.DataFrame(
    repeated_results
)

print(
    repeated_results_df
)

print(
    "\nTotal repeated evaluation rows:",
    len(repeated_results_df),
)

repeated_overlap_values = (
    repeated_results_df[
        "Profile Overlap"
    ]
    .to_numpy(
        dtype=int
    )
)

maximum_repeated_profile_overlap = int(
    np.max(
        repeated_overlap_values
    )
)

print(
    "Maximum profile overlap:",
    maximum_repeated_profile_overlap,
)



# %%
# CELL 14 — Summarize repeated deployment stability

repeated_summary_df = pd.DataFrame(
    repeated_results_df
    .groupby(
        "Model",
        as_index=False,
    )
    .agg(
        Mean_MAE=("MAE", "mean"),
        Std_MAE=("MAE", "std"),
        Mean_RMSE=("RMSE", "mean"),
        Std_RMSE=("RMSE", "std"),
        Mean_R2=("R2", "mean"),
        Std_R2=("R2", "std"),
        Min_R2=("R2", "min"),
        Max_R2=("R2", "max"),
    )
)

repeated_summary_df = (
    repeated_summary_df
    .reset_index(
        drop=True
    )
)

print(
    repeated_summary_df
)





# %%
# CELL 15 — Count model wins across repeated splits

split_winner_rows = []

for split_number in range(
    1,
    21,
):
    split_results = pd.DataFrame(
        repeated_results_df[
            repeated_results_df[
                "Split"
            ] == split_number
        ]
    ).copy()

    split_r2_values = (
        split_results[
            "R2"
        ]
        .to_numpy(
            dtype=float
        )
    )

    winning_position = int(
        np.argmax(
            split_r2_values
        )
    )

    winning_model = str(
        split_results[
            "Model"
        ]
        .to_numpy()[winning_position]
    )

    winning_r2 = float(
        split_r2_values[
            winning_position
        ]
    )

    split_winner_rows.append(
        {
            "Split": split_number,
            "Model": winning_model,
            "R2": winning_r2,
        }
    )

split_winners = pd.DataFrame(
    split_winner_rows
)

win_counts = (
    split_winners[
        "Model"
    ]
    .value_counts()
    .rename_axis(
        "Model"
    )
    .reset_index(
        name="Wins"
    )
)

print(
    split_winners
)

print(
    "\nModel win counts:"
)

print(
    win_counts
)




# %%
# CELL 16 — Plot mean R² across feature scenarios

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
    "Day 12: Full vs Pre-sleep vs Deployment Performance"
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
    title="Feature Scenario"
)

plt.xticks(
    rotation=0
)

plt.tight_layout()

figure_path = (
    FIGURES_DIR
    / "deployment_feature_scenario_group_cv_r2.png"
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
# CELL 17 — Plot repeated deployment R² stability

stability_plot_data = pd.DataFrame(
    repeated_results_df[
        repeated_results_df[
            "Model"
        ].isin(
            [
                "Linear",
                "Ridge",
                "Random Forest",
            ]
        )
    ]
).copy()

ax = stability_plot_data.boxplot(
    column="R2",
    by="Model",
    figsize=(10, 6),
)

plt.suptitle("")

ax.set_title(
    "Day 12: Deployment Model Stability Across 20 Group Holdouts"
)

ax.set_xlabel(
    "Model"
)

ax.set_ylabel(
    "R² Across Group-Separated Holdouts"
)

ax.axhline(
    0,
    linewidth=1,
)

plt.tight_layout()

stability_figure_path = (
    FIGURES_DIR
    / "deployment_repeated_group_holdout_r2.png"
)

plt.savefig(
    stability_figure_path,
    dpi=300,
    bbox_inches="tight",
)

plt.show()

print(
    "Saved figure:",
    stability_figure_path.name,
)




# %%
# CELL 18 — Save Day 12 report tables

fold_diagnostics_path = (
    REPORTS_DIR
    / "deployment_group_fold_diagnostics.csv"
)

same_fold_results_path = (
    REPORTS_DIR
    / "deployment_same_fold_results.csv"
)

same_fold_summary_path = (
    REPORTS_DIR
    / "deployment_same_fold_summary.csv"
)

scenario_comparison_path = (
    REPORTS_DIR
    / "deployment_scenario_comparison.csv"
)

repeated_results_path = (
    REPORTS_DIR
    / "deployment_repeated_group_holdout_results.csv"
)

repeated_summary_path = (
    REPORTS_DIR
    / "deployment_repeated_group_holdout_summary.csv"
)

split_winners_path = (
    REPORTS_DIR
    / "deployment_repeated_group_holdout_winners.csv"
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

repeated_results_df.to_csv(
    repeated_results_path,
    index=False,
)

repeated_summary_df.to_csv(
    repeated_summary_path,
    index=False,
)

split_winners.to_csv(
    split_winners_path,
    index=False,
)

print("Saved:")
print(fold_diagnostics_path.name)
print(same_fold_results_path.name)
print(same_fold_summary_path.name)
print(scenario_comparison_path.name)
print(repeated_results_path.name)
print(repeated_summary_path.name)
print(split_winners_path.name)




# %%
# CELL 19 — Generate Day 12 interpretation

ridge_summary = repeated_summary_df[
    repeated_summary_df[
        "Model"
    ] == "Ridge"
].iloc[0]

rf_summary = repeated_summary_df[
    repeated_summary_df[
        "Model"
    ] == "Random Forest"
].iloc[0]

linear_summary = repeated_summary_df[
    repeated_summary_df[
        "Model"
    ] == "Linear"
].iloc[0]

print(
    "DAY 12 INTERPRETATION"
)

print(
    "-" * 50
)

print(
    "Deployment unique profiles:",
    len(deployment_unique_profiles),
)

print(
    "Maximum shared-fold profile overlap:",
    maximum_profile_overlap,
)

print(
    "\nRepeated holdout mean R²:"
)

print(
    "Ridge:",
    round(
        float(
            ridge_summary[
                "Mean_R2"
            ]
        ),
        4,
    ),
)

print(
    "Random Forest:",
    round(
        float(
            rf_summary[
                "Mean_R2"
            ]
        ),
        4,
    ),
)

print(
    "Linear:",
    round(
        float(
            linear_summary[
                "Mean_R2"
            ]
        ),
        4,
    ),
)

print(
    "\nRepeated holdout R² standard deviation:"
)

print(
    "Ridge:",
    round(
        float(
            ridge_summary[
                "Std_R2"
            ]
        ),
        4,
    ),
)

print(
    "Random Forest:",
    round(
        float(
            rf_summary[
                "Std_R2"
            ]
        ),
        4,
    ),
)

print(
    "Linear:",
    round(
        float(
            linear_summary[
                "Std_R2"
            ]
        ),
        4,
    ),
)

print(
    "\nModel wins across 20 splits:"
)

print(
    win_counts
)

print(
    "\nInterpretation:"
)

print(
    "Ridge achieved the highest mean R² and the lowest R² "
    "variability across repeated deployment-profile holdouts."
)

print(
    "Random Forest remained highly competitive and matched Ridge "
    "in the number of individual split wins, but its performance "
    "was less stable across unseen deployment profiles."
)

print(
    "Linear Regression also remained strong, showing that much of "
    "the predictive structure in the deployment feature set can be "
    "captured by relatively simple models."
)

print(
    "These results favor Ridge as the most stable deployment-oriented "
    "candidate under the current dataset and validation design."
)




# %%
# CELL 20 — Generate Day 12 research journal entry

journal_entry = f"""
## Day 12 — Deployment-Oriented Feature Set and Stability Analysis

### Objective

Evaluate AthleteIQ using a stricter deployment-oriented feature set and determine how stable model performance remains across repeated unseen-profile train/test splits.

### Research Question

How much predictive performance remains when AthleteIQ uses only features that could realistically be available before sleep, and how stable is that performance across different unseen predictor profiles?

### Motivation

Day 11 removed same-night `Sleep Duration` to create a more realistic pre-sleep prediction scenario.

However, several remaining variables may also be unavailable or impractical in a real deployment setting.

Day 12 therefore created a stricter deployment-oriented feature set by excluding:

- `Sleep Duration`
- `Sleep Disorder`
- `Systolic BP`
- `Diastolic BP`

### Feature Scenarios

Three feature scenarios were compared:

1. **Full**
   - 12 predictors

2. **Pre-sleep**
   - 11 predictors
   - Excludes `Sleep Duration`

3. **Deployment**
   - 8 predictors
   - Excludes `Sleep Duration`
   - Excludes `Sleep Disorder`
   - Excludes `Systolic BP`
   - Excludes `Diastolic BP`

The deployment feature set contains:

- Gender
- Age
- Occupation
- Physical Activity Level
- Stress Level
- BMI Category
- Heart Rate
- Daily Steps

### Predictor-Profile Structure

- Full-feature unique profiles: {len(full_unique_profiles)}
- Pre-sleep unique profiles: {len(pre_sleep_unique_profiles)}
- Deployment unique profiles: {len(deployment_unique_profiles)}
- Profiles lost from Full to Pre-sleep: {len(full_unique_profiles) - len(pre_sleep_unique_profiles)}
- Profiles lost from Pre-sleep to Deployment: {len(pre_sleep_unique_profiles) - len(deployment_unique_profiles)}
- Deployment profiles with multiple target values: {conflicting_deployment_profiles}

Removing additional predictors reduced the number of exact predictor profiles from {len(pre_sleep_unique_profiles)} in the pre-sleep scenario to only {len(deployment_unique_profiles)} in the deployment scenario.

No identical deployment predictor profile was associated with multiple `Quality of Sleep` values.

### Shared Group-Aware Validation

A 5-fold `GroupKFold` design was used for the direct comparison among Full, Pre-sleep, and Deployment scenarios.

All three scenarios were evaluated using the **same folds**, defined by the strictest deployment-profile groups.

Maximum deployment-profile overlap between training and validation folds:

- **{maximum_profile_overlap} profiles**

This ensures that identical deployment profiles never appear in both training and validation within the same fold.

All categorical preprocessing was performed inside scikit-learn `Pipeline` objects so one-hot encoding was fit only on each training fold.

### Same-Fold Performance

| Scenario | Model | Mean MAE | Mean RMSE | Mean R² | Std R² |
|---|---|---:|---:|---:|---:|
| Full | Random Forest | 0.0951 | 0.2723 | 0.9393 | 0.0197 |
| Full | Ridge | 0.2077 | 0.3207 | 0.9065 | 0.0428 |
| Full | Linear | 0.2123 | 0.3422 | 0.8959 | 0.0428 |
| Pre-sleep | Random Forest | 0.1155 | 0.3206 | 0.9125 | 0.0260 |
| Pre-sleep | Ridge | 0.2118 | 0.3221 | 0.9032 | 0.0516 |
| Pre-sleep | Linear | 0.2211 | 0.3425 | 0.8926 | 0.0531 |
| Deployment | Ridge | 0.1982 | 0.3124 | 0.9153 | 0.0216 |
| Deployment | Random Forest | 0.1125 | 0.3148 | 0.9148 | 0.0284 |
| Deployment | Linear | 0.2032 | 0.3274 | 0.9082 | 0.0185 |

### Feature-Scenario Findings

Under deployment-defined shared folds:

Random Forest changed from:

- Full R²: **0.9393**
- Pre-sleep R²: **0.9125**
- Deployment R²: **0.9148**

The deployment Random Forest therefore remained highly predictive even after removing `Sleep Duration`, `Sleep Disorder`, and blood-pressure variables.

Ridge changed from:

- Full R²: **0.9065**
- Pre-sleep R²: **0.9032**
- Deployment R²: **0.9153**

Linear Regression also improved slightly under the deployment feature set.

These improvements should not be interpreted as the removed variables containing harmful information in a general sense. Instead, under these particular group-separated folds, the smaller feature set may reduce redundancy or noise for the linear models.

The Day 12 Full and Pre-sleep scores also should not be directly substituted for the Day 11 estimates because Day 12 uses the stricter deployment-profile grouping to define the shared folds.

### Repeated Group-Separated Stability Evaluation

To evaluate stability beyond a single 5-fold partition, `GroupShuffleSplit` was used to create 20 repeated deployment-profile-separated train/test splits.

Each split placed entire deployment profiles into either training or testing.

Maximum exact deployment-profile overlap across the repeated holdouts:

- **0 profiles**

A total of 80 model evaluations were produced:

- 20 splits
- 4 models per split

### Repeated Holdout Results

| Model | Mean MAE | Mean RMSE | Mean R² | Std R² | Min R² | Max R² |
|---|---:|---:|---:|---:|---:|---:|
| Ridge | 0.2048 | 0.3230 | 0.9043 | 0.0563 | 0.7505 | 0.9742 |
| Linear | 0.2105 | 0.3408 | 0.8917 | 0.0716 | 0.6868 | 0.9709 |
| Random Forest | 0.1337 | 0.3550 | 0.8872 | 0.0819 | 0.7092 | 0.9882 |
| Dummy | 1.0493 | 1.2170 | -0.1942 | 0.2019 | -0.6074 | -0.0000 |

### Model Wins Across 20 Splits

- Ridge: **9 wins**
- Random Forest: **9 wins**
- Linear Regression: **2 wins**

Ridge and Random Forest therefore won the same number of individual splits.

However, Ridge achieved the highest mean R² and the lowest R² standard deviation among the predictive models.

### Interpretation

Day 12 changes the model-selection picture.

Random Forest was the strongest model in the earlier full-feature analyses, but Ridge becomes the strongest stability-oriented candidate when evaluation focuses on a stricter deployment feature set and repeated unseen deployment profiles.

Ridge achieved:

- Mean repeated-holdout R²: **0.9043**
- R² standard deviation: **0.0563**
- Minimum R²: **0.7505**
- Maximum R²: **0.9742**

Random Forest achieved:

- Mean repeated-holdout R²: **0.8872**
- R² standard deviation: **0.0819**
- Minimum R²: **0.7092**
- Maximum R²: **0.9882**

Random Forest occasionally achieved extremely strong performance, but its results varied more substantially depending on which unseen profiles were selected for testing.

Ridge therefore provides the strongest combination of average performance and stability under the current deployment-oriented evaluation.

Linear Regression also remained strong, suggesting that much of the predictive structure available in the reduced feature set can be captured with relatively simple linear relationships.

### Scientific Interpretation

The results reinforce that the model with the highest score under one evaluation design is not automatically the best final model.

Model choice depends on:

- realistic feature availability,
- independence between training and evaluation data,
- average predictive performance,
- variability across different splits,
- model complexity,
- and the intended deployment setting.

For AthleteIQ, Ridge currently offers an attractive balance of predictive performance, stability, and simplicity under the deployment-oriented scenario.

### Limitations

- The dataset contains only 374 observations.
- The deployment feature set produces only {len(deployment_unique_profiles)} unique predictor profiles.
- Exact-profile grouping prevents identical profiles from crossing train/test boundaries but does not establish unseen-person or external-population generalization.
- Repeated rows within profiles can still influence model fitting.
- The dataset is observational.
- Several variables may be self-reported.
- Stress Level may itself be measured contemporaneously depending on the intended application.
- The dataset may not represent athletic populations specifically.
- No independent external dataset was available.
- Strong performance does not establish causal relationships or clinical readiness.
- Twenty repeated holdouts provide a more informative stability estimate than one split, but uncertainty remains because the dataset is small.

### Files Generated

Reports:

- `reports/deployment_group_fold_diagnostics.csv`
- `reports/deployment_same_fold_results.csv`
- `reports/deployment_same_fold_summary.csv`
- `reports/deployment_scenario_comparison.csv`
- `reports/deployment_repeated_group_holdout_results.csv`
- `reports/deployment_repeated_group_holdout_summary.csv`
- `reports/deployment_repeated_group_holdout_winners.csv`

Figures:

- `figures/deployment_feature_scenario_group_cv_r2.png`
- `figures/deployment_repeated_group_holdout_r2.png`

Notebook/script:

- `notebooks/10_deployment_feature_stability.py`

### Key Conclusion

The deployment-oriented feature set retained strong predictive performance despite removing four potentially unavailable predictors.

Across 20 repeated deployment-profile-separated holdouts, Ridge achieved the strongest average and most stable performance.

The current leading deployment-oriented model is therefore:

**Ridge Regression**

This selection is based on stability and realistic feature availability rather than simply maximizing the highest observed R².

### Next Steps

- Perform final model-selection analysis using the deployment feature set.
- Examine Ridge coefficients and their stability to understand which deployment features contribute most strongly.
- Compare Ridge interpretation with the earlier Random Forest feature-importance findings.
- Consider whether `Stress Level`, `Heart Rate`, and other remaining predictors are truly available at the intended prediction time.
- Establish a final feature specification before hyperparameter tuning.
- Only then consider modest hyperparameter tuning and final model packaging.
"""

print(
    journal_entry
)




# %%
# CELL 21 — Final Day 12 verification

expected_report_files = [
    "deployment_group_fold_diagnostics.csv",
    "deployment_same_fold_results.csv",
    "deployment_same_fold_summary.csv",
    "deployment_scenario_comparison.csv",
    "deployment_repeated_group_holdout_results.csv",
    "deployment_repeated_group_holdout_summary.csv",
    "deployment_repeated_group_holdout_winners.csv",
]

expected_figure_files = [
    "deployment_feature_scenario_group_cv_r2.png",
    "deployment_repeated_group_holdout_r2.png",
]

print(
    "DAY 12 FINAL VERIFICATION"
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
    "Deployment profiles:",
    len(deployment_unique_profiles),
)

print(
    "Deployment profiles with multiple targets:",
    conflicting_deployment_profiles,
)

print(
    "Maximum shared-fold profile overlap:",
    maximum_profile_overlap,
)

print(
    "\nRepeated holdout results:"
)

print(
    "Ridge mean R²:",
    round(
        float(
            ridge_summary[
                "Mean_R2"
            ]
        ),
        4,
    ),
)

print(
    "Ridge R² std:",
    round(
        float(
            ridge_summary[
                "Std_R2"
            ]
        ),
        4,
    ),
)

print(
    "Random Forest mean R²:",
    round(
        float(
            rf_summary[
                "Mean_R2"
            ]
        ),
        4,
    ),
)

print(
    "Linear mean R²:",
    round(
        float(
            linear_summary[
                "Mean_R2"
            ]
        ),
        4,
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
    "\nFigure files:"
)

for filename in expected_figure_files:
    file_exists = (
        FIGURES_DIR
        / filename
    ).exists()

    print(
        filename,
        "->",
        file_exists,
    )

print(
    "\nDay 12 analysis completed successfully."
)




# %%
# CELL 22 — Append Day 12 to research journal

journal_path = (
    PROJECT_ROOT
    / "docs"
    / "research_journal.md"
)

existing_journal = journal_path.read_text(
    encoding="utf-8"
)

day_12_heading = (
    "## Day 12 — Deployment-Oriented Feature Set and Stability Analysis"
)

if day_12_heading in existing_journal:
    print(
        "Day 12 already exists in research journal."
    )
else:
    with journal_path.open(
        "a",
        encoding="utf-8",
    ) as journal_file:
        journal_file.write(
            "\n\n---\n\n"
        )
        journal_file.write(
            journal_entry.strip()
        )
        journal_file.write(
            "\n"
        )

    print(
        "Day 12 appended to research journal."
    )

print(
    "Journal:",
    journal_path
)
