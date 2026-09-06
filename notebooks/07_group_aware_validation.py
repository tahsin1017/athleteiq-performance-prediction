# %%
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.base import clone
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
    KFold,
)

PROJECT_ROOT = Path(
    "/Users/home/Desktop/AthleteIQ/athleteiq-performance-prediction"
)

DATA_DIR = PROJECT_ROOT / "data" / "processed"
REPORTS_DIR = PROJECT_ROOT / "reports"
FIGURES_DIR = PROJECT_ROOT / "figures"

REPORTS_DIR.mkdir(parents=True, exist_ok=True)
FIGURES_DIR.mkdir(parents=True, exist_ok=True)

print("Project root:", PROJECT_ROOT)
print("Day 9 environment ready.")


# %%
# CELL 2 — Load Day 5 train/test datasets

X_train_original = pd.read_csv(DATA_DIR / "X_train.csv")
X_test_original = pd.read_csv(DATA_DIR / "X_test.csv")

y_train_original = pd.read_csv(DATA_DIR / "y_train.csv")["Quality of Sleep"]
y_test_original = pd.read_csv(DATA_DIR / "y_test.csv")["Quality of Sleep"]

print("X_train:", X_train_original.shape)
print("X_test:", X_test_original.shape)
print("y_train:", y_train_original.shape)
print("y_test:", y_test_original.shape)

print("\ny_train type:", type(y_train_original))
print("y_test type:", type(y_test_original))



# %%
# CELL 3 — Reconstruct the complete encoded dataset

X_all = pd.concat(
    [X_train_original, X_test_original],
    ignore_index=True,
)

y_all = pd.concat(
    [y_train_original, y_test_original],
    ignore_index=True,
)

print("Combined X shape:", X_all.shape)
print("Combined y shape:", y_all.shape)

print("\nMissing values in X_all:", int(X_all.isna().sum().sum()))
print("Target accidentally present in X_all:", "Quality of Sleep" in X_all.columns)



# %%
# CELL 4 — Create exact predictor-profile groups

profile_tuples = pd.Series(
    list(map(tuple, X_all.to_numpy()))
)

groups, unique_profiles = pd.factorize(
    profile_tuples
)

profile_check = X_all.copy()
profile_check["Quality of Sleep"] = y_all.to_numpy()
profile_check["Profile Group"] = groups

target_values_per_profile = (
    profile_check
    .groupby("Profile Group")["Quality of Sleep"]
    .nunique()
)

conflicting_profiles = int(
    (target_values_per_profile > 1).sum()
)

print("Total observations:", len(X_all))
print("Unique predictor profiles:", len(unique_profiles))
print("Conflicting profile targets:", conflicting_profiles)
# %%
# CELL 5 — Define models and validation strategies

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

random_cv = KFold(
    n_splits=5,
    shuffle=True,
    random_state=42,
)

group_cv = GroupKFold(
    n_splits=5
)

print("Models loaded:")
for model_name in models:
    print("-", model_name)

print("\nRandom CV:", random_cv)




# %%
# CELL 6 — Verify GroupKFold has zero profile overlap

group_fold_diagnostics = []

for fold_number, (train_idx, val_idx) in enumerate(
    group_cv.split(X_all, y_all, groups=groups),
    start=1,
):
    train_groups = set(groups[train_idx])
    val_groups = set(groups[val_idx])

    overlapping_groups = train_groups.intersection(val_groups)

    group_fold_diagnostics.append(
        {
            "Fold": fold_number,
            "Training Rows": len(train_idx),
            "Validation Rows": len(val_idx),
            "Training Profiles": len(train_groups),
            "Validation Profiles": len(val_groups),
            "Profile Overlap": len(overlapping_groups),
        }
    )

group_fold_diagnostics_df = pd.DataFrame(group_fold_diagnostics)

print(group_fold_diagnostics_df.to_string(index=False))

print(
    "\nMaximum profile overlap:",
    group_fold_diagnostics_df["Profile Overlap"].max()
)


# %%
# CELL 7 — Create helper function for model evaluation

def evaluate_model_cv(
    model_name,
    model,
    splitter,
    strategy_name,
    use_groups=False,
):
    results = []

    if use_groups:
        split_iterator = splitter.split(
            X_all,
            y_all,
            groups=groups,
        )
    else:
        split_iterator = splitter.split(
            X_all,
            y_all,
        )

    for fold_number, (train_idx, val_idx) in enumerate(
        split_iterator,
        start=1,
    ):
        estimator = clone(model)

        X_train_fold = X_all.iloc[train_idx]
        X_val_fold = X_all.iloc[val_idx]

        y_train_fold = y_all.iloc[train_idx]
        y_val_fold = y_all.iloc[val_idx]

        estimator.fit(
            X_train_fold,
            y_train_fold,
        )

        predictions = estimator.predict(
            X_val_fold
        )

        results.append(
            {
                "Model": model_name,
                "Validation Strategy": strategy_name,
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

    return pd.DataFrame(results)


print("Evaluation helper function ready.")



# %%
# CELL 8 — Run all models under random CV and group-aware CV

all_cv_results = []

for model_name, model in models.items():

    random_results = evaluate_model_cv(
        model_name=model_name,
        model=model,
        splitter=random_cv,
        strategy_name="Random 5-Fold CV",
        use_groups=False,
    )

    group_results = evaluate_model_cv(
        model_name=model_name,
        model=model,
        splitter=group_cv,
        strategy_name="Group-Aware 5-Fold CV",
        use_groups=True,
    )

    all_cv_results.append(random_results)
    all_cv_results.append(group_results)

cv_fold_results_df = pd.concat(
    all_cv_results,
    ignore_index=True,
)

print(cv_fold_results_df.head(10).to_string(index=False))
print("\nTotal result rows:", len(cv_fold_results_df))



# %%
# CELL 9 — Summarize CV performance

cv_summary_df = (
    cv_fold_results_df
    .groupby(
        ["Model", "Validation Strategy"]
    )
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

print(
    cv_summary_df.to_string(
        index=False
    )
)



# %%
# CELL 10 — Quantify R² change from random to group-aware CV

r2_comparison_df = (
    cv_summary_df
    .pivot(
        index="Model",
        columns="Validation Strategy",
        values="Mean_R2",
    )
    .reset_index()
)

r2_comparison_df["R2 Change (Group - Random)"] = (
    r2_comparison_df["Group-Aware 5-Fold CV"]
    - r2_comparison_df["Random 5-Fold CV"]
)

print(
    r2_comparison_df.to_string(
        index=False
    )
)



# %%
# CELL 11 — Create a profile-separated holdout split

group_holdout_splitter = GroupShuffleSplit(
    n_splits=1,
    test_size=0.20,
    random_state=42,
)

group_train_idx, group_test_idx = next(
    group_holdout_splitter.split(
        X_all,
        y_all,
        groups=groups,
    )
)

X_group_train = X_all.iloc[
    group_train_idx
].reset_index(drop=True)

X_group_test = X_all.iloc[
    group_test_idx
].reset_index(drop=True)

y_group_train = y_all.iloc[
    group_train_idx
].reset_index(drop=True)

y_group_test = y_all.iloc[
    group_test_idx
].reset_index(drop=True)

groups_train = groups[group_train_idx]
groups_test = groups[group_test_idx]

train_profiles = set(groups_train)
test_profiles = set(groups_test)

profile_overlap = train_profiles.intersection(
    test_profiles
)

print("Training rows:", len(X_group_train))
print("Test rows:", len(X_group_test))

print(
    "Training unique profiles:",
    len(train_profiles)
)

print(
    "Test unique profiles:",
    len(test_profiles)
)

print(
    "Exact profile overlap:",
    len(profile_overlap)
)



# %%
# CELL 12 — Evaluate all models on the profile-separated holdout

group_holdout_results = []
group_holdout_predictions = {}

for model_name, model in models.items():
    estimator = clone(model)

    estimator.fit(
        X_group_train,
        y_group_train,
    )

    predictions = estimator.predict(
        X_group_test
    )

    group_holdout_predictions[model_name] = predictions

    mae = mean_absolute_error(
        y_group_test,
        predictions,
    )

    rmse = np.sqrt(
        mean_squared_error(
            y_group_test,
            predictions,
        )
    )

    r2 = r2_score(
        y_group_test,
        predictions,
    )

    group_holdout_results.append(
        {
            "Model": model_name,
            "MAE": mae,
            "RMSE": rmse,
            "R2": r2,
        }
    )

group_holdout_results_df = (
    pd.DataFrame(group_holdout_results)
    .sort_values(
        "R2",
        ascending=False,
    )
    .reset_index(drop=True)
)

print(
    group_holdout_results_df.to_string(
        index=False
    )
)



# %%
# CELL 13 — Compare Random Forest with the original Day 7 holdout

original_rf_results = pd.DataFrame(
    {
        "Evaluation": [
            "Original Random Holdout",
            "Profile-Separated Holdout",
        ],
        "MAE": [
            0.039133,
            float(
                group_holdout_results_df.loc[
                    group_holdout_results_df["Model"] == "Random Forest",
                    "MAE",
                ].iloc[0]
            ),
        ],
        "RMSE": [
            0.133767,
            float(
                group_holdout_results_df.loc[
                    group_holdout_results_df["Model"] == "Random Forest",
                    "RMSE",
                ].iloc[0]
            ),
        ],
        "R2": [
            0.988139,
            float(
                group_holdout_results_df.loc[
                    group_holdout_results_df["Model"] == "Random Forest",
                    "R2",
                ].iloc[0]
            ),
        ],
    }
)

print(
    original_rf_results.to_string(
        index=False
    )
)



# %%
# CELL 14 — Save holdout results and diagnostics

group_holdout_results_df.to_csv(
    REPORTS_DIR / "group_holdout_model_results.csv",
    index=False,
)

original_rf_results.to_csv(
    REPORTS_DIR / "random_forest_holdout_comparison.csv",
    index=False,
)

holdout_diagnostics_df = pd.DataFrame(
    {
        "Metric": [
            "Total rows",
            "Total unique profiles",
            "Training rows",
            "Test rows",
            "Training profiles",
            "Test profiles",
            "Profile overlap",
        ],
        "Value": [
            len(X_all),
            len(np.unique(groups)),
            len(X_group_train),
            len(X_group_test),
            len(np.unique(groups_train)),
            len(np.unique(groups_test)),
            len(profile_overlap),
        ],
    }
)

holdout_diagnostics_df.to_csv(
    REPORTS_DIR / "group_holdout_split_diagnostics.csv",
    index=False,
)

print("Saved:")
print("- group_holdout_model_results.csv")
print("- random_forest_holdout_comparison.csv")
print("- group_holdout_split_diagnostics.csv")
# %%
# CELL 15 — Plot Random Forest actual vs predicted on unseen profiles

rf_group_predictions = group_holdout_predictions[
    "Random Forest"
]

plt.figure(
    figsize=(7, 7)
)

plt.scatter(
    y_group_test,
    rf_group_predictions,
    alpha=0.7,
)

minimum_value = min(
    y_group_test.min(),
    rf_group_predictions.min(),
)

maximum_value = max(
    y_group_test.max(),
    rf_group_predictions.max(),
)

plt.plot(
    [minimum_value, maximum_value],
    [minimum_value, maximum_value],
    linestyle="--",
)

plt.xlabel(
    "Actual Quality of Sleep"
)

plt.ylabel(
    "Predicted Quality of Sleep"
)

plt.title(
    "Random Forest: Actual vs Predicted\n"
    "Profile-Separated Holdout"
)

plt.tight_layout()

figure_path = (
    FIGURES_DIR
    / "group_holdout_rf_actual_vs_predicted.png"
)

plt.savefig(
    figure_path,
    dpi=300,
    bbox_inches="tight",
)

plt.show()

print("Saved:", figure_path)




# %%
# CELL 16 — Repeat profile-separated Random Forest evaluation 20 times

repeated_splitter = GroupShuffleSplit(
    n_splits=20,
    test_size=0.20,
    random_state=42,
)

repeated_results = []

for split_number, (train_idx, test_idx) in enumerate(
    repeated_splitter.split(
        X_all,
        y_all,
        groups=groups,
    ),
    start=1,
):
    split_train_groups = set(
        groups[train_idx]
    )

    split_test_groups = set(
        groups[test_idx]
    )

    split_overlap = len(
        split_train_groups.intersection(
            split_test_groups
        )
    )

    rf = RandomForestRegressor(
        n_estimators=200,
        random_state=42,
        n_jobs=-1,
    )

    rf.fit(
        X_all.iloc[train_idx],
        y_all.iloc[train_idx],
    )

    predictions = rf.predict(
        X_all.iloc[test_idx]
    )

    repeated_results.append(
        {
            "Split": split_number,
            "Training Rows": len(train_idx),
            "Test Rows": len(test_idx),
            "Training Profiles": len(split_train_groups),
            "Test Profiles": len(split_test_groups),
            "Profile Overlap": split_overlap,
            "MAE": mean_absolute_error(
                y_all.iloc[test_idx],
                predictions,
            ),
            "RMSE": np.sqrt(
                mean_squared_error(
                    y_all.iloc[test_idx],
                    predictions,
                )
            ),
            "R2": r2_score(
                y_all.iloc[test_idx],
                predictions,
            ),
        }
    )

repeated_group_results_df = pd.DataFrame(
    repeated_results
)

print(
    repeated_group_results_df.to_string(
        index=False
    )
)

print(
    "\nMaximum profile overlap:",
    repeated_group_results_df[
        "Profile Overlap"
    ].max()
)




# %%
# CELL 17 — Summarize repeated profile-separated performance

repeated_group_summary_df = pd.DataFrame(
    {
        "Metric": [
            "Mean MAE",
            "Std MAE",
            "Mean RMSE",
            "Std RMSE",
            "Mean R2",
            "Std R2",
            "Minimum R2",
            "Maximum R2",
        ],
        "Value": [
            repeated_group_results_df["MAE"].mean(),
            repeated_group_results_df["MAE"].std(),
            repeated_group_results_df["RMSE"].mean(),
            repeated_group_results_df["RMSE"].std(),
            repeated_group_results_df["R2"].mean(),
            repeated_group_results_df["R2"].std(),
            repeated_group_results_df["R2"].min(),
            repeated_group_results_df["R2"].max(),
        ],
    }
)

print(
    repeated_group_summary_df.to_string(
        index=False
    )
)





# %%
# CELL 18 — Save repeated holdout results and summary

repeated_group_results_df.to_csv(
    REPORTS_DIR / "random_forest_repeated_group_holdout.csv",
    index=False,
)

repeated_group_summary_df.to_csv(
    REPORTS_DIR / "random_forest_repeated_group_summary.csv",
    index=False,
)

print("Saved:")
print("- random_forest_repeated_group_holdout.csv")
print("- random_forest_repeated_group_summary.csv")



# %%
# CELL 19 — Plot repeated profile-separated R² distribution

plt.figure(
    figsize=(9, 6)
)

plt.hist(
    repeated_group_results_df["R2"],
    bins=10,
)

r2_values = repeated_group_results_df["R2"].to_numpy(
    dtype=float
)
mean_r2 = float(
    np.mean(r2_values)
)

plt.axvline(
    mean_r2,
    linestyle="--",
    label="Mean profile-separated R²",

)

plt.axvline(
    0.988139,
    linestyle=":",
    label="Original random holdout R²",
)

plt.xlabel("Random Forest R²")
plt.ylabel("Number of Splits")

plt.title(
    "Random Forest Performance Across\n"
    "20 Profile-Separated Holdout Splits"
)

plt.legend()
plt.tight_layout()

figure_path = (
    FIGURES_DIR
    / "random_forest_group_holdout_r2_distribution.png"
)

plt.savefig(
    figure_path,
    dpi=300,
    bbox_inches="tight",
)

plt.show()

print("Saved:", figure_path)




# %%
# CELL 20 — Plot random CV vs group-aware CV

plot_data = (
    cv_summary_df
    .pivot(
        index="Model",
        columns="Validation Strategy",
        values="Mean_R2",
    )
)

ax = plot_data.plot(
    kind="bar",
    figsize=(11, 6),
)

ax.set_xlabel("Model")
ax.set_ylabel("Mean 5-Fold CV R²")

ax.set_title(
    "Random vs Group-Aware Cross-Validation"
)

plt.xticks(
    rotation=20,
    ha="right",
)

plt.tight_layout()

figure_path = (
    FIGURES_DIR
    / "random_vs_group_cv_r2.png"
)

plt.savefig(
    figure_path,
    dpi=300,
    bbox_inches="tight",
)

plt.show()

print("Saved:", figure_path)





# %%
# CELL 21 — Save CV summaries and diagnostics

cv_summary_df.to_csv(
    REPORTS_DIR / "random_vs_group_cv_results.csv",
    index=False,
)

r2_comparison_df.to_csv(
    REPORTS_DIR / "random_vs_group_cv_r2_comparison.csv",
    index=False,
)

group_fold_diagnostics_df.to_csv(
    REPORTS_DIR / "group_cv_fold_diagnostics.csv",
    index=False,
)

print("Saved:")
print("- random_vs_group_cv_results.csv")
print("- random_vs_group_cv_r2_comparison.csv")
print("- group_cv_fold_diagnostics.csv")




# %%
# CELL 22 — Day 9 result snapshot

rf_random_cv = cv_summary_df[
    (
        cv_summary_df["Model"] == "Random Forest"
    )
    &
    (
        cv_summary_df["Validation Strategy"]
        == "Random 5-Fold CV"
    )
].iloc[0]

rf_group_cv = cv_summary_df[
    (
        cv_summary_df["Model"] == "Random Forest"
    )
    &
    (
        cv_summary_df["Validation Strategy"]
        == "Group-Aware 5-Fold CV"
    )
].iloc[0]

rf_holdout_row = group_holdout_results_df[
    group_holdout_results_df["Model"]
    == "Random Forest"
].iloc[0]

rf_r2_change = (
    rf_group_cv["Mean_R2"]
    - rf_random_cv["Mean_R2"]
)

print("=" * 70)
print("DAY 9 RESULT SNAPSHOT")
print("=" * 70)

print("\nDataset:")
print("Rows:", len(X_all))
print("Unique predictor profiles:", len(unique_profiles))

print("\nGroup-aware validation check:")
print(
    "Maximum profile overlap across GroupKFold folds:",
    group_fold_diagnostics_df["Profile Overlap"].max()
)

print("\nRandom Forest — Random 5-Fold CV:")
print(f"MAE: {rf_random_cv['Mean_MAE']:.4f}")
print(f"RMSE: {rf_random_cv['Mean_RMSE']:.4f}")
print(f"R²: {rf_random_cv['Mean_R2']:.4f}")

print("\nRandom Forest — Group-Aware 5-Fold CV:")
print(f"MAE: {rf_group_cv['Mean_MAE']:.4f}")
print(f"RMSE: {rf_group_cv['Mean_RMSE']:.4f}")
print(f"R²: {rf_group_cv['Mean_R2']:.4f}")

print(
    "\nR² change (group-aware - random):",
    f"{rf_r2_change:+.4f}"
)

print("\nProfile-Separated Holdout:")
print("Training rows:", len(X_group_train))
print("Test rows:", len(X_group_test))
print("Training profiles:", len(train_profiles))
print("Test profiles:", len(test_profiles))
print("Profile overlap:", len(profile_overlap))

print("\nRandom Forest — Profile-Separated Holdout:")
print(f"MAE: {rf_holdout_row['MAE']:.4f}")
print(f"RMSE: {rf_holdout_row['RMSE']:.4f}")
print(f"R²: {rf_holdout_row['R2']:.4f}")

print("\n20 Repeated Profile-Separated Holdouts:")
print(
    f"Mean MAE: "
    f"{repeated_group_results_df['MAE'].mean():.4f}"
)
print(
    f"Mean RMSE: "
    f"{repeated_group_results_df['RMSE'].mean():.4f}"
)
print(
    f"Mean R²: "
    f"{repeated_group_results_df['R2'].mean():.4f}"
)
print(
    f"Std R²: "
    f"{repeated_group_results_df['R2'].std():.4f}"
)
print(
    f"Minimum R²: "
    f"{repeated_group_results_df['R2'].min():.4f}"
)
print(
    f"Maximum R²: "
    f"{repeated_group_results_df['R2'].max():.4f}"
)

print("\nDay 9 analysis completed successfully.")




# %%
# CELL 23 — Generate Day 9 research journal entry

journal_entry = f"""
## Day 9 — Group-Aware Validation and Unseen-Profile Generalization

### Objective

Evaluate AthleteIQ under a stricter validation design in which identical predictor profiles cannot appear in both training and evaluation data.

Day 8 showed that the original random train/test split contained substantial profile overlap, with 65 of 75 test observations having an exact predictor profile already represented in training.

### Work Completed

- Reconstructed the complete encoded dataset containing {len(X_all)} observations.
- Confirmed {len(unique_profiles)} unique predictor profiles.
- Assigned identical predictor combinations to the same profile group.
- Verified that no profile group was associated with multiple Quality of Sleep target values.
- Compared ordinary 5-fold random cross-validation with 5-fold GroupKFold validation.
- Verified zero predictor-profile overlap in every group-aware validation fold.
- Created a profile-separated holdout split using GroupShuffleSplit.
- Evaluated Dummy Regressor, Linear Regression, Ridge Regression, and Random Forest.
- Repeated profile-separated Random Forest evaluation across 20 different holdout splits.

### Key Results

Random Forest under ordinary random 5-fold cross-validation:

- MAE: {rf_random_cv['Mean_MAE']:.4f}
- RMSE: {rf_random_cv['Mean_RMSE']:.4f}
- R²: {rf_random_cv['Mean_R2']:.4f}

Random Forest under group-aware 5-fold cross-validation:

- MAE: {rf_group_cv['Mean_MAE']:.4f}
- RMSE: {rf_group_cv['Mean_RMSE']:.4f}
- R²: {rf_group_cv['Mean_R2']:.4f}
- R² change relative to random CV: {rf_r2_change:+.4f}

Profile-separated Random Forest holdout:

- Training rows: {len(X_group_train)}
- Test rows: {len(X_group_test)}
- Training profiles: {len(train_profiles)}
- Test profiles: {len(test_profiles)}
- Exact profile overlap: {len(profile_overlap)}
- MAE: {rf_holdout_row['MAE']:.4f}
- RMSE: {rf_holdout_row['RMSE']:.4f}
- R²: {rf_holdout_row['R2']:.4f}

Across 20 repeated profile-separated holdouts:

- Mean MAE: {repeated_group_results_df['MAE'].mean():.4f}
- Mean RMSE: {repeated_group_results_df['RMSE'].mean():.4f}
- Mean R²: {repeated_group_results_df['R2'].mean():.4f}
- R² standard deviation: {repeated_group_results_df['R2'].std():.4f}
- Minimum R²: {repeated_group_results_df['R2'].min():.4f}
- Maximum R²: {repeated_group_results_df['R2'].max():.4f}

### Model Comparison Under Stricter Evaluation

Under the profile-separated holdout, Linear Regression and Ridge Regression slightly outperformed Random Forest:

- Linear Regression R²: 0.9286
- Ridge Regression R²: 0.9282
- Random Forest R²: 0.9165

This shows that model ranking can change when evaluation is performed on genuinely unseen predictor profiles.

### Interpretation

Group-aware validation produced lower performance than ordinary random validation, confirming that repeated predictor profiles had made the original evaluation somewhat optimistic.

However, the Random Forest still achieved strong average performance under stricter evaluation, with a mean R² of {repeated_group_results_df['R2'].mean():.4f} across 20 profile-separated holdouts.

The wide range in R² values, from {repeated_group_results_df['R2'].min():.4f} to {repeated_group_results_df['R2'].max():.4f}, shows that model performance depends strongly on which unseen profiles are selected for testing.

The findings should therefore be interpreted as evidence of train-test profile overlap and non-independent observations under random splitting, rather than direct target leakage.

### Limitations

- Group-aware validation removes exact profile overlap but does not establish external real-world or clinical generalization.
- Repeated observations within the same profile still remain together inside training folds and can give some profiles greater weight.
- The encoded feature representation was created before the original split, so preprocessing is not yet fully isolated inside each training fold.
- The dataset remains small, with only {len(unique_profiles)} unique predictor profiles.
- High predictive performance should not be interpreted as evidence of clinical readiness.

### Files Generated

Reports:

- `group_cv_fold_diagnostics.csv`
- `random_vs_group_cv_results.csv`
- `random_vs_group_cv_r2_comparison.csv`
- `group_holdout_model_results.csv`
- `random_forest_holdout_comparison.csv`
- `group_holdout_split_diagnostics.csv`
- `random_forest_repeated_group_holdout.csv`
- `random_forest_repeated_group_summary.csv`

Figures:

- `random_vs_group_cv_r2.png`
- `group_holdout_rf_actual_vs_predicted.png`
- `random_forest_group_holdout_r2_distribution.png`

### Next Steps

- Rebuild preprocessing using a train-only scikit-learn Pipeline and ColumnTransformer.
- Ensure imputation, categorical encoding, and all learned preprocessing occur only inside training folds.
- Re-evaluate baseline models under the stricter validation framework.
- Compare model performance after removing preprocessing leakage risk.
"""

print(journal_entry)
