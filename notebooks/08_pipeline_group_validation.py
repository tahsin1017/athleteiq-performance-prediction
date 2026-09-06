# %%
# CELL 1 — Day 10 imports and project paths

from pathlib import Path

import numpy as np
import pandas as pd
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
print("Day 10 environment ready.")



# %%
# CELL 2 — Load processed dataset before one-hot encoding

processed_data_path = (
    DATA_DIR
    / "processed_sleep_health_dataset.csv"
)

data = pd.read_csv(
    processed_data_path
)

print("Dataset shape:", data.shape)
print("\nColumns:")
for column_name in data.columns:
    print("-", column_name)

print(
    "\nMissing values:",
    int(data.isna().sum().sum())
)



# %%
# CELL 3 — Define target, predictors, and profile groups

TARGET_COLUMN = "Quality of Sleep"

X = data.drop(
    columns=[TARGET_COLUMN]
).copy()

y = data[
    TARGET_COLUMN
].copy()

# Use a temporary copy only for defining exact profile groups.
# Missing values are replaced with a label so identical missing-value
# profiles are grouped consistently.
profile_data = X.fillna(
    "__MISSING__"
)

profile_tuples = pd.Series(
    list(
        map(
            tuple,
            profile_data.to_numpy(),
        )
    )
)

groups, unique_profiles = pd.factorize(
    profile_tuples
)

print("X shape:", X.shape)
print("y shape:", y.shape)

print(
    "Target present in X:",
    TARGET_COLUMN in X.columns,
)

print(
    "Unique predictor profiles:",
    len(unique_profiles),
)

print(
    "Total missing predictor values:",
    int(X.isna().sum().sum()),
)




# %%
# CELL 4 — Identify numerical and categorical features

numerical_features = X.select_dtypes(
    include=["number"]
).columns.tolist()

categorical_features = X.select_dtypes(
    exclude=["number"]
).columns.tolist()

print("Numerical features:")
for feature_name in numerical_features:
    print("-", feature_name)

print("\nCategorical features:")
for feature_name in categorical_features:
    print("-", feature_name)

print(
    "\nNumber of numerical features:",
    len(numerical_features),
)

print(
    "Number of categorical features:",
    len(categorical_features),
)





# %%
# CELL 5 — Build train-only preprocessing transformer

preprocessor = ColumnTransformer(
    transformers=[
        (
            "numerical",
            "passthrough",
            numerical_features,
        ),
        (
            "categorical",
            OneHotEncoder(
                handle_unknown="ignore",
            ),
            categorical_features,
        ),
    ]
)

print("Preprocessor created successfully.")
print(preprocessor)



# %%
# CELL 6 — Build preprocessing + model pipelines

models = {
    "Dummy Regressor": DummyRegressor(
        strategy="mean"
    ),
    "Linear Regression": LinearRegression(),
    "Ridge Regression": Ridge(
        alpha=1.0
    ),
    "Random Forest": RandomForestRegressor(
        n_estimators=200,
        random_state=42,
        n_jobs=-1,
    ),
}

pipelines = {
    model_name: Pipeline(
        steps=[
            (
                "preprocessor",
                preprocessor,
            ),
            (
                "model",
                model,
            ),
        ]
    )
    for model_name, model in models.items()
}

print("Pipelines created:")

for pipeline_name in pipelines:
    print("-", pipeline_name)



# %%
# CELL 7 — Configure and verify group-aware cross-validation

group_cv = GroupKFold(
    n_splits=5
)

group_fold_diagnostics = []

for fold_number, (train_idx, val_idx) in enumerate(
    group_cv.split(
        X,
        y,
        groups=groups,
    ),
    start=1,
):
    train_groups = set(
        groups[train_idx]
    )

    val_groups = set(
        groups[val_idx]
    )

    overlap = train_groups.intersection(
        val_groups
    )

    group_fold_diagnostics.append(
        {
            "Fold": fold_number,
            "Training Rows": len(train_idx),
            "Validation Rows": len(val_idx),
            "Training Profiles": len(train_groups),
            "Validation Profiles": len(val_groups),
            "Profile Overlap": len(overlap),
        }
    )

group_fold_diagnostics_df = pd.DataFrame(
    group_fold_diagnostics
)

print(
    group_fold_diagnostics_df.to_string(
        index=False
    )
)






# %%
# CELL 8 — Evaluate train-only pipelines with GroupKFold

pipeline_cv_results = []

for model_name, pipeline in pipelines.items():

    for fold_number, (train_idx, val_idx) in enumerate(
        group_cv.split(
            X,
            y,
            groups=groups,
        ),
        start=1,
    ):
        X_train_fold = X.iloc[train_idx]
        X_val_fold = X.iloc[val_idx]

        y_train_fold = y.iloc[train_idx]
        y_val_fold = y.iloc[val_idx]

        pipeline.fit(
            X_train_fold,
            y_train_fold,
        )

        predictions = pipeline.predict(
            X_val_fold
        )

        pipeline_cv_results.append(
            {
                "Model": model_name,
                "Fold": fold_number,
                "MAE": mean_absolute_error(
                    y_val_fold,
                    predictions,
                ),
                "RMSE": np.sqrt(
                    mean_squared_error(
                        y_val_fold,
                        predictions,
                    )
                ),
                "R2": r2_score(
                    y_val_fold,
                    predictions,
                ),
            }
        )

pipeline_cv_results_df = pd.DataFrame(
    pipeline_cv_results
)

print(
    pipeline_cv_results_df.head(10).to_string(
        index=False
    )
)

print(
    "\nTotal result rows:",
    len(pipeline_cv_results_df),
)




# %%
# CELL 9 — Summarize pipeline GroupKFold performance

pipeline_cv_summary_df = (
    pipeline_cv_results_df
    .groupby("Model")
    .agg(
        Mean_MAE=("MAE", "mean"),
        Std_MAE=("MAE", "std"),
        Mean_RMSE=("RMSE", "mean"),
        Std_RMSE=("RMSE", "std"),
        Mean_R2=("R2", "mean"),
        Std_R2=("R2", "std"),
    )
    .reset_index()
)

pipeline_cv_summary_df = (
    pipeline_cv_summary_df
    .sort_values(
        "Mean_R2",
        ascending=False,
    )
    .reset_index(drop=True)
)

print(
    pipeline_cv_summary_df.to_string(
        index=False
    )
)




# %%
# CELL 10 — Create globally encoded comparison dataset

global_preprocessor = ColumnTransformer(
    transformers=[
        (
            "numerical",
            "passthrough",
            numerical_features,
        ),
        (
            "categorical",
            OneHotEncoder(
                handle_unknown="ignore",
            ),
            categorical_features,
        ),
    ]
)

X_global_encoded = global_preprocessor.fit_transform(
    X
)

print(
    "Globally encoded shape:",
    X_global_encoded.shape,
)

print(
    "Global preprocessing fitted on all rows:",
    len(X),
)



# %%
# CELL 11 — Evaluate globally encoded data on the same group folds

global_cv_results = []

for model_name, model in models.items():

    for fold_number, (train_idx, val_idx) in enumerate(
        group_cv.split(
            X_global_encoded,
            y,
            groups=groups,
        ),
        start=1,
    ):
        X_train_fold = X_global_encoded[train_idx]
        X_val_fold = X_global_encoded[val_idx]

        y_train_fold = y.iloc[train_idx]
        y_val_fold = y.iloc[val_idx]

        model.fit(
            X_train_fold,
            y_train_fold,
        )

        predictions = model.predict(
            X_val_fold
        )

        global_cv_results.append(
            {
                "Model": model_name,
                "Fold": fold_number,
                "MAE": mean_absolute_error(
                    y_val_fold,
                    predictions,
                ),
                "RMSE": np.sqrt(
                    mean_squared_error(
                        y_val_fold,
                        predictions,
                    )
                ),
                "R2": r2_score(
                    y_val_fold,
                    predictions,
                ),
            }
        )

global_cv_results_df = pd.DataFrame(
    global_cv_results
)

print(
    global_cv_results_df.head(10).to_string(
        index=False
    )
)

print(
    "\nTotal result rows:",
    len(global_cv_results_df),
)




# %%
# CELL 12 — Compare global preprocessing with train-only pipelines

global_cv_summary_df = (
    global_cv_results_df
    .groupby("Model")
    .agg(
        Global_Mean_MAE=("MAE", "mean"),
        Global_Mean_RMSE=("RMSE", "mean"),
        Global_Mean_R2=("R2", "mean"),
    )
    .reset_index()
)

pipeline_comparison_df = (
    pipeline_cv_summary_df[
        [
            "Model",
            "Mean_MAE",
            "Mean_RMSE",
            "Mean_R2",
        ]
    ]
    .merge(
        global_cv_summary_df,
        on="Model",
        how="left",
    )
)

pipeline_comparison_df = (
    pipeline_comparison_df.rename(
        columns={
            "Mean_MAE": "Pipeline_Mean_MAE",
            "Mean_RMSE": "Pipeline_Mean_RMSE",
            "Mean_R2": "Pipeline_Mean_R2",
        }
    )
)

pipeline_comparison_df["R2_Change"] = (
    pipeline_comparison_df["Pipeline_Mean_R2"]
    - pipeline_comparison_df["Global_Mean_R2"]
)

pipeline_comparison_df["MAE_Change"] = (
    pipeline_comparison_df["Pipeline_Mean_MAE"]
    - pipeline_comparison_df["Global_Mean_MAE"]
)

print(
    pipeline_comparison_df.to_string(
        index=False
    )
)





# %%
# CELL 13 — Save Day 10 pipeline results

pipeline_cv_results_df.to_csv(
    REPORTS_DIR / "pipeline_group_cv_fold_results.csv",
    index=False,
)

pipeline_cv_summary_df.to_csv(
    REPORTS_DIR / "pipeline_group_cv_summary.csv",
    index=False,
)

pipeline_comparison_df.to_csv(
    REPORTS_DIR / "global_vs_pipeline_preprocessing_comparison.csv",
    index=False,
)

group_fold_diagnostics_df.to_csv(
    REPORTS_DIR / "pipeline_group_cv_fold_diagnostics.csv",
    index=False,
)

print("Saved:")
print("- pipeline_group_cv_fold_results.csv")
print("- pipeline_group_cv_summary.csv")
print("- global_vs_pipeline_preprocessing_comparison.csv")
print("- pipeline_group_cv_fold_diagnostics.csv")




# %%
# CELL 14 — Day 10 result snapshot

rf_pipeline_row = pipeline_cv_summary_df[
    pipeline_cv_summary_df["Model"] == "Random Forest"
].iloc[0]

rf_comparison_row = pipeline_comparison_df[
    pipeline_comparison_df["Model"] == "Random Forest"
].iloc[0]

print("=" * 70)
print("DAY 10 RESULT SNAPSHOT")
print("=" * 70)

print("\nDataset:")
print("Rows:", len(X))
print("Raw predictor columns:", X.shape[1])
print("Unique predictor profiles:", len(unique_profiles))
print("\nGroup-aware validation:")
print("Number of folds:", group_cv.get_n_splits())
print(
    "Maximum profile overlap:",
    int(
        np.max(
            group_fold_diagnostics_df[
                "Profile Overlap"
            ].to_numpy(dtype=int)
        )
    ),
)

print("\nRandom Forest — Train-Only Pipeline:")
print(
    f"Mean MAE: "
    f"{rf_pipeline_row['Mean_MAE']:.4f}"
)
print(
    f"Mean RMSE: "
    f"{rf_pipeline_row['Mean_RMSE']:.4f}"
)
print(
    f"Mean R²: "
    f"{rf_pipeline_row['Mean_R2']:.4f}"
)
print(
    f"Std R²: "
    f"{rf_pipeline_row['Std_R2']:.4f}"
)

print("\nGlobal vs Train-Only Preprocessing:")
print(
    f"Global R²: "
    f"{rf_comparison_row['Global_Mean_R2']:.4f}"
)
print(
    f"Pipeline R²: "
    f"{rf_comparison_row['Pipeline_Mean_R2']:.4f}"
)
print(
    f"R² change: "
    f"{rf_comparison_row['R2_Change']:+.6f}"
)
print(
    f"MAE change: "
    f"{rf_comparison_row['MAE_Change']:+.6f}"
)

print("\nDay 10 analysis completed successfully.")






# %%
# CELL 15 — Plot global vs train-only preprocessing performance

import matplotlib.pyplot as plt

plot_data = pipeline_comparison_df[
    [
        "Model",
        "Global_Mean_R2",
        "Pipeline_Mean_R2",
    ]
].set_index(
    "Model"
)

ax = plot_data.plot(
    kind="bar",
    figsize=(10, 6),
)

ax.set_xlabel("Model")
ax.set_ylabel("Mean GroupKFold R²")

ax.set_title(
    "Global vs Train-Only Preprocessing"
)

plt.xticks(
    rotation=20,
    ha="right",
)

plt.tight_layout()

figure_path = (
    FIGURES_DIR
    / "global_vs_pipeline_group_cv_r2.png"
)

plt.savefig(
    figure_path,
    dpi=300,
    bbox_inches="tight",
)

plt.show()

print("Saved:", figure_path)






# %%
# CELL 16 — Day 10 interpretation summary

print("=" * 70)
print("DAY 10 INTERPRETATION")
print("=" * 70)

print(
    "\nTrain-only preprocessing was successfully implemented "
    "inside a scikit-learn Pipeline."
)

print(
    "\nAll preprocessing was fit separately within each "
    "GroupKFold training fold."
)

print(
    "\nExact predictor-profile overlap across validation folds remained:",
    int(
        np.max(
            group_fold_diagnostics_df[
                "Profile Overlap"
            ].to_numpy(dtype=int)
        )
    ),
)

print(
    "\nRandom Forest mean GroupKFold R² with global preprocessing:",
    f"{rf_comparison_row['Global_Mean_R2']:.4f}",
)

print(
    "Random Forest mean GroupKFold R² with train-only preprocessing:",
    f"{rf_comparison_row['Pipeline_Mean_R2']:.4f}",
)

print(
    "Difference:",
    f"{rf_comparison_row['R2_Change']:+.6f}",
)

print(
    "\nConclusion:"
)

print(
    "The previous global one-hot encoding approach was "
    "methodologically less rigorous, but it did not materially "
    "inflate group-aware validation performance on these same folds."
)

print(
    "\nDay 10 interpretation completed."
)




# %%
# CELL 17 — Generate Day 10 research journal entry

journal_entry = f"""
## Day 10 — Train-Only Preprocessing Pipeline

### Objective

Rebuild the AthleteIQ preprocessing workflow using a scikit-learn Pipeline and ColumnTransformer so that learned preprocessing is fit only on training data within each validation fold.

### Motivation

Earlier stages encoded categorical variables before cross-validation.

Although one-hot encoding does not use the target variable directly, fitting preprocessing globally is methodologically less rigorous because information about the complete set of categories is available before validation.

Day 10 therefore tested whether a strict train-only preprocessing design changes group-aware model performance.

### Work Completed

- Loaded the processed dataset before one-hot encoding.
- Separated the target variable from the 12 raw predictor columns.
- Reconstructed exact predictor-profile groups.
- Confirmed 132 unique predictor profiles.
- Identified 8 numerical features and 4 categorical features.
- Built a ColumnTransformer.
- Passed numerical variables through unchanged.
- Applied OneHotEncoder with `handle_unknown="ignore"` to categorical variables.
- Wrapped preprocessing and each regression model inside a scikit-learn Pipeline.
- Evaluated Dummy Regressor, Linear Regression, Ridge Regression, and Random Forest using 5-fold GroupKFold validation.
- Verified zero exact predictor-profile overlap across all validation folds.
- Constructed a globally encoded comparison dataset.
- Evaluated globally encoded models using the exact same group-aware folds.
- Compared global preprocessing with strict train-only preprocessing.

### Dataset

- Rows: {len(X)}
- Raw predictor columns: {X.shape[1]}
- Numerical features: {len(numerical_features)}
- Categorical features: {len(categorical_features)}
- Unique predictor profiles: {len(unique_profiles)}
- Missing predictor values: {int(X.isna().sum().sum())}

### Group-Aware Validation

GroupKFold used 5 folds.

Maximum exact predictor-profile overlap between training and validation data:

- **0 profiles**

This confirms that identical predictor profiles never appeared in both the training and validation portion of the same fold.

### Train-Only Pipeline Results

| Model | Mean MAE | Mean RMSE | Mean R² | Std R² |
|---|---:|---:|---:|---:|
| Random Forest | 0.0677 | 0.2080 | 0.9612 | 0.0314 |
| Ridge Regression | 0.1731 | 0.2856 | 0.9325 | 0.0300 |
| Linear Regression | 0.1753 | 0.3061 | 0.9214 | 0.0368 |
| Dummy Regressor | 1.0673 | 1.2210 | -0.1565 | 0.0946 |

Random Forest remained the strongest model under the strict train-only preprocessing design.

### Global vs Train-Only Preprocessing

Using the exact same GroupKFold splits:

Random Forest with globally fitted one-hot encoding:

- Mean R²: {rf_comparison_row['Global_Mean_R2']:.4f}

Random Forest with train-only pipeline preprocessing:

- Mean R²: {rf_comparison_row['Pipeline_Mean_R2']:.4f}

Difference:

- R² change: {rf_comparison_row['R2_Change']:+.6f}
- MAE change: {rf_comparison_row['MAE_Change']:+.6f}

Linear Regression, Ridge Regression, and Dummy Regression showed effectively no measurable difference between global one-hot encoding and train-only pipeline preprocessing on the same folds.

### Interpretation

The previous preprocessing workflow was methodologically less rigorous because categorical encoding was fit before validation.

However, the fair same-fold comparison showed that global one-hot encoding did not materially inflate group-aware validation performance in this dataset.

For Random Forest, the difference in mean R² was only {rf_comparison_row['R2_Change']:+.6f}.

The small difference indicates that the strong model performance observed in Day 9 was not primarily caused by globally fitting the one-hot encoder.

The more important methodological issue remains the repeated predictor-profile structure identified during Days 8 and 9.

### Why the Pipeline Is Better

Even though the numerical performance changed very little, the pipeline design is preferable because:

- preprocessing is fit only on training data,
- validation data remains isolated,
- unseen categories can be handled safely using `handle_unknown="ignore"`,
- preprocessing and modeling are bundled into one reproducible workflow,
- future scaling, imputation, or feature transformations can be added without leaking validation information,
- the evaluation design is easier to reproduce and extend.

### Limitations

- The dataset remains small, with only {len(unique_profiles)} unique predictor profiles.
- Group-aware validation prevents exact predictor-profile overlap but does not establish external generalization.
- Missing Sleep Disorder values remain present in the raw dataset and are handled by OneHotEncoder as a categorical level.
- Repeated observations within the same predictor profile can still give some profiles greater influence during training.
- No external independent dataset was available for validation.
- High predictive performance should not be interpreted as clinical readiness.

### Files Generated

Reports:

- `pipeline_group_cv_fold_results.csv`
- `pipeline_group_cv_summary.csv`
- `global_vs_pipeline_preprocessing_comparison.csv`
- `pipeline_group_cv_fold_diagnostics.csv`

Figure:

- `global_vs_pipeline_group_cv_r2.png`

### Key Conclusion

A strict train-only preprocessing pipeline was successfully implemented.

The new pipeline improved methodological rigor but produced almost the same group-aware model performance as global one-hot encoding.

For Random Forest:

- Global preprocessing R²: {rf_comparison_row['Global_Mean_R2']:.4f}
- Train-only pipeline R²: {rf_comparison_row['Pipeline_Mean_R2']:.4f}
- Difference: {rf_comparison_row['R2_Change']:+.6f}

This indicates that global one-hot encoding was not a major source of optimistic performance in this experiment.

### Next Steps

- Define a realistic prediction-time feature scenario.
- Evaluate whether same-night Sleep Duration should remain available as a predictor.
- Compare full-feature and deployment-realistic feature sets.
- Consider repeated group-aware validation for the pipeline-based workflow.
- Delay hyperparameter tuning until the final prediction scenario and feature set are defined.
"""

print(journal_entry)
