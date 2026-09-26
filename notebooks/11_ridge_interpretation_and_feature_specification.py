# %%
# CELL 1 — Day 13 imports and project paths

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import Ridge
from sklearn.model_selection import GroupKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler


def find_project_root() -> Path:
    search_starts = [
        Path.cwd().resolve(),
    ]

    script_path = globals().get("__file__")

    if isinstance(script_path, str):
        search_starts.insert(
            0,
            Path(script_path).resolve().parent,
        )

    for start_path in search_starts:
        for candidate_path in (
            start_path,
            *start_path.parents,
        ):
            if (
                (candidate_path / ".git").exists()
                and (candidate_path / "data").exists()
                and (candidate_path / "notebooks").exists()
            ):
                return candidate_path

    raise FileNotFoundError(
        "Could not locate the AthleteIQ project root."
    )


PROJECT_ROOT = find_project_root()

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
print("Day 13 environment ready.")


# %%
# CELL 2 — Load dataset and define deployment features

processed_data_path = (
    DATA_DIR
    / "processed_sleep_health_dataset.csv"
)

data = pd.read_csv(
    processed_data_path
)

TARGET_COLUMN = "Quality of Sleep"

deployment_features = [
    "Gender",
    "Age",
    "Occupation",
    "Physical Activity Level",
    "Stress Level",
    "BMI Category",
    "Heart Rate",
    "Daily Steps",
]

X_deployment = data[
    deployment_features
].copy()

y = data[
    TARGET_COLUMN
].copy()

print(
    "Dataset shape:",
    data.shape,
)

print(
    "Deployment feature shape:",
    X_deployment.shape,
)

print(
    "Deployment features:",
    X_deployment.columns.tolist(),
)

print(
    "Target present in predictors:",
    TARGET_COLUMN in X_deployment.columns,
)
# %%
# CELL 3 — Rebuild deployment profile groups

deployment_profile_data = X_deployment.fillna(
    "__MISSING__"
)

deployment_profile_tuples = pd.Series(
    list(
        map(
            tuple,
            deployment_profile_data.to_numpy(),
        )
    )
)

deployment_groups, deployment_unique_profiles = pd.factorize(
    deployment_profile_tuples
)

print(
    "Deployment unique profiles:",
    len(deployment_unique_profiles),
)

print(
    "Total rows:",
    len(X_deployment),
)


# %%
# CELL 4 — Identify deployment numeric and categorical features

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
    "\nFeature count:",
    (
        len(deployment_numeric_features)
        + len(deployment_categorical_features)
    ),
)


# %%
# CELL 5 — Build interpretable standardized Ridge pipeline

preprocessor = ColumnTransformer(
    transformers=[
        (
            "numeric",
            StandardScaler(),
            deployment_numeric_features,
        ),
        (
            "categorical",
            OneHotEncoder(
                handle_unknown="ignore",
                drop="first",
            ),
            deployment_categorical_features,
        ),
    ]
)

ridge_pipeline = Pipeline(
    steps=[
        (
            "preprocessor",
            preprocessor,
        ),
        (
            "model",
            Ridge(
                alpha=1.0
            ),
        ),
    ]
)

print(
    "Interpretable standardized Ridge pipeline ready."
)


# %%
# CELL 6 — Build shared deployment-profile folds

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
    "\nMaximum deployment-profile overlap:",
    maximum_profile_overlap,
)


# %%
# CELL 7 — Extract Ridge coefficients across group-aware folds

coefficient_rows = []

for fold_number, (
    train_indices,
    validation_indices,
) in enumerate(
    shared_folds,
    start=1,
):
    X_train_fold = X_deployment.iloc[
        train_indices
    ]

    y_train_fold = y.iloc[
        train_indices
    ]

    fold_pipeline = clone(
        ridge_pipeline
    )

    assert isinstance(
        fold_pipeline,
        Pipeline,
    )

    fold_pipeline.fit(
        X_train_fold,
        y_train_fold,
    )

    fitted_preprocessor = fold_pipeline.named_steps[
        "preprocessor"
    ]

    fitted_ridge = fold_pipeline.named_steps[
        "model"
    ]

    assert isinstance(
        fitted_preprocessor,
        ColumnTransformer,
    )

    assert isinstance(
        fitted_ridge,
        Ridge,
    )

    feature_names = (
        fitted_preprocessor
        .get_feature_names_out()
    )

    coefficients = (
        fitted_ridge.coef_
    )

    for feature_name, coefficient in zip(
        feature_names,
        coefficients,
        strict=True,
    ):
        coefficient_rows.append(
            {
                "Fold": fold_number,
                "Feature": str(
                    feature_name
                ),
                "Coefficient": float(
                    coefficient
                ),
            }
        )

coefficient_df = pd.DataFrame(
    coefficient_rows
)

print(
    coefficient_df
)

print(
    "\nCoefficient rows:",
    len(coefficient_df),
)

print(
    "Unique transformed features:",
    coefficient_df[
        "Feature"
    ].nunique(),
)

# %%
# CELL 8 — Summarize Ridge coefficient stability

coefficient_summary_df = pd.DataFrame(
    coefficient_df
    .groupby(
        "Feature",
        as_index=False,
    )
    .agg(
        Mean_Coefficient=("Coefficient", "mean"),
        Std_Coefficient=("Coefficient", "std"),
        Min_Coefficient=("Coefficient", "min"),
        Max_Coefficient=("Coefficient", "max"),
        Fold_Count=("Coefficient", "count"),
    )
)

coefficient_summary_df[
    "Absolute_Mean_Coefficient"
] = (
    coefficient_summary_df[
        "Mean_Coefficient"
    ].abs()
)

coefficient_summary_df[
    "Sign_Stable"
] = (
    (
        coefficient_summary_df[
            "Min_Coefficient"
        ] >= 0
    )
    |
    (
        coefficient_summary_df[
            "Max_Coefficient"
        ] <= 0
    )
)

coefficient_summary_df = (
    coefficient_summary_df
    .sort_values(
        by="Absolute_Mean_Coefficient",
        ascending=False,
    )
    .reset_index(
        drop=True
    )
)

print(
    coefficient_summary_df
)

stable_sign_values = (
    coefficient_summary_df[
        "Sign_Stable"
    ]
    .to_numpy(
        dtype=bool
    )
)

stable_sign_count = int(
    np.count_nonzero(
        stable_sign_values
    )
)

print(
    "\nFeatures with stable coefficient sign:",
    stable_sign_count,
)

print(
    "Total summarized features:",
    len(
        coefficient_summary_df
    ),
)


# %%
# CELL 9 — Separate stable and unstable Ridge coefficients

stable_coefficients_df = pd.DataFrame(
    coefficient_summary_df[
        coefficient_summary_df[
            "Sign_Stable"
        ]
    ]
).copy()

unstable_coefficients_df = pd.DataFrame(
    coefficient_summary_df[
        ~coefficient_summary_df[
            "Sign_Stable"
        ]
    ]
).copy()

print(
    "Stable coefficients:"
)

print(
    stable_coefficients_df[
        [
            "Feature",
            "Mean_Coefficient",
            "Std_Coefficient",
            "Fold_Count",
        ]
    ]
)

print(
    "\nUnstable coefficients:"
)

print(
    unstable_coefficients_df[
        [
            "Feature",
            "Mean_Coefficient",
            "Std_Coefficient",
            "Min_Coefficient",
            "Max_Coefficient",
            "Fold_Count",
        ]
    ]
)


# %%
# CELL 10 — Fit final deployment Ridge and extract coefficients

final_ridge_pipeline = clone(
    ridge_pipeline
)

assert isinstance(
    final_ridge_pipeline,
    Pipeline,
)

final_ridge_pipeline.fit(
    X_deployment,
    y,
)

final_preprocessor = final_ridge_pipeline.named_steps[
    "preprocessor"
]

final_ridge_model = final_ridge_pipeline.named_steps[
    "model"
]

assert isinstance(
    final_preprocessor,
    ColumnTransformer,
)

assert isinstance(
    final_ridge_model,
    Ridge,
)

final_feature_names = (
    final_preprocessor
    .get_feature_names_out()
)

final_coefficients = (
    final_ridge_model.coef_
)

final_coefficient_df = pd.DataFrame(
    {
        "Feature": final_feature_names,
        "Coefficient": final_coefficients,
    }
)

final_coefficient_df[
    "Absolute_Coefficient"
] = (
    final_coefficient_df[
        "Coefficient"
    ].abs()
)

final_coefficient_df = (
    final_coefficient_df
    .sort_values(
        by="Absolute_Coefficient",
        ascending=False,
    )
    .reset_index(
        drop=True
    )
)

print(
    final_coefficient_df
)

print(
    "\nFinal transformed feature count:",
    len(
        final_coefficient_df
    ),
)



# %%
# CELL 11 — Combine final coefficients with fold stability

coefficient_interpretation_df = pd.merge(
    final_coefficient_df,
    coefficient_summary_df[
        [
            "Feature",
            "Mean_Coefficient",
            "Std_Coefficient",
            "Min_Coefficient",
            "Max_Coefficient",
            "Fold_Count",
            "Sign_Stable",
        ]
    ],
    on="Feature",
    how="left",
)

print(
    coefficient_interpretation_df
)

final_stable_values = (
    coefficient_interpretation_df[
        "Sign_Stable"
    ]
    .to_numpy(
        dtype=bool
    )
)

final_stable_count = int(
    np.count_nonzero(
        final_stable_values
    )
)

print(
    "\nSign-stable final features:",
    final_stable_count,
)

print(
    "Total final transformed features:",
    len(
        coefficient_interpretation_df
    ),
)


# %%
# CELL 12 — Plot Ridge coefficient stability

plot_coefficients_df = pd.DataFrame(
    coefficient_summary_df[
        [
            "Feature",
            "Mean_Coefficient",
            "Std_Coefficient",
            "Sign_Stable",
        ]
    ]
).copy()

plot_coefficients_df = (
    plot_coefficients_df
    .sort_values(
        by="Mean_Coefficient",
        ascending=True,
    )
    .reset_index(
        drop=True
    )
)

ax = plot_coefficients_df.plot(
    x="Feature",
    y="Mean_Coefficient",
    kind="barh",
    xerr="Std_Coefficient",
    figsize=(11, 8),
    legend=False,
)

ax.set_title(
    "Day 13: Ridge Coefficient Stability Across Group-Aware Folds"
)

ax.set_xlabel(
    "Mean Ridge Coefficient"
)

ax.set_ylabel(
    "Transformed Feature"
)

ax.axvline(
    0,
    linewidth=1,
)

plt.tight_layout()

coefficient_figure_path = (
    FIGURES_DIR
    / "ridge_coefficient_stability.png"
)

plt.savefig(
    coefficient_figure_path,
    dpi=300,
    bbox_inches="tight",
)

plt.show()

print(
    "Saved figure:",
    coefficient_figure_path.name,
)


# %%
# CELL 13 — Save Day 13 coefficient reports

coefficient_folds_path = (
    REPORTS_DIR
    / "ridge_fold_coefficients.csv"
)

coefficient_summary_path = (
    REPORTS_DIR
    / "ridge_coefficient_stability_summary.csv"
)

coefficient_interpretation_path = (
    REPORTS_DIR
    / "ridge_final_coefficient_interpretation.csv"
)

coefficient_df.to_csv(
    coefficient_folds_path,
    index=False,
)

coefficient_summary_df.to_csv(
    coefficient_summary_path,
    index=False,
)

coefficient_interpretation_df.to_csv(
    coefficient_interpretation_path,
    index=False,
)

print("Saved:")
print(coefficient_folds_path.name)
print(coefficient_summary_path.name)
print(coefficient_interpretation_path.name)


# %%
# CELL 14 — Generate Day 13 interpretation

stable_feature_values = (
    coefficient_interpretation_df[
        "Sign_Stable"
    ]
    .to_numpy(
        dtype=bool
    )
)

stable_feature_count = int(
    np.count_nonzero(
        stable_feature_values
    )
)

total_feature_count = len(
    coefficient_interpretation_df
)

unstable_features = (
    coefficient_interpretation_df[
        ~coefficient_interpretation_df[
            "Sign_Stable"
        ]
    ]["Feature"]
    .tolist()
)

print(
    "DAY 13 INTERPRETATION"
)

print(
    "-" * 50
)

print(
    "Deployment profiles:",
    len(deployment_unique_profiles),
)

print(
    "Maximum profile overlap:",
    maximum_profile_overlap,
)

print(
    "Final transformed features:",
    total_feature_count,
)

print(
    "Sign-stable transformed features:",
    stable_feature_count,
)

print(
    "\nUnstable transformed features:"
)

for feature_name in unstable_features:
    print(
        "-",
        feature_name,
    )

print(
    "\nKey numeric patterns:"
)

print(
    "Stress Level: strong negative and highly stable."
)

print(
    "Age: positive and highly stable."
)

print(
    "Heart Rate: negative and sign-stable."
)

print(
    "Physical Activity Level: positive and sign-stable, but smaller."
)

print(
    "Daily Steps: weak and sign-unstable."
)

print(
    "\nInterpretation:"
)

print(
    "Ridge coefficients show that the deployment model is driven "
    "primarily by a small number of stable signals rather than by "
    "uniformly strong contributions from every predictor."
)

print(
    "Stress Level and Age are the clearest stable numeric predictors "
    "under the current standardized Ridge specification."
)

print(
    "Daily Steps contributes only a small and unstable independent "
    "coefficient once the other deployment predictors are included."
)

print(
    "Categorical coefficients must be interpreted relative to their "
    "omitted reference categories rather than as absolute effects."
)

print(
    "These coefficient results support keeping the current deployment "
    "feature set for now, while flagging Daily Steps and a few "
    "occupation contrasts as weak or unstable contributors."
)


# %%
# CELL 15 — Generate Day 13 research journal entry

journal_entry = f"""
## Day 13 — Ridge Interpretation and Final Feature Specification

### Objective

Interpret the deployment-oriented Ridge Regression model and evaluate the stability of its coefficients across group-aware folds.

### Motivation

Day 12 identified Ridge Regression as the most stable deployment-oriented model under repeated group-separated holdout evaluation.

Before moving to hyperparameter tuning or model packaging, Day 13 examined whether Ridge relies on stable and interpretable predictors.

### Deployment Feature Set

The deployment-oriented predictor set contains 8 original features:

- Gender
- Age
- Occupation
- Physical Activity Level
- Stress Level
- BMI Category
- Heart Rate
- Daily Steps

The deployment feature set excludes:

- Sleep Duration
- Sleep Disorder
- Systolic BP
- Diastolic BP

### Predictor-Profile Structure

- Total rows: {len(X_deployment)}
- Unique deployment profiles: {len(deployment_unique_profiles)}
- Maximum train-validation profile overlap: {maximum_profile_overlap}

Five-fold `GroupKFold` validation was used so identical deployment predictor profiles could not appear in both training and validation data within the same fold.

### Interpretable Ridge Pipeline

The Ridge model was rebuilt using:

- `StandardScaler` for numerical variables
- `OneHotEncoder(handle_unknown="ignore", drop="first")` for categorical variables
- `Ridge(alpha=1.0)` as the regression model

Numerical standardization was added because raw numerical variables such as Age, Heart Rate, and Daily Steps are measured on different scales.

Using `drop="first"` created an omitted reference category for each categorical variable, making categorical Ridge coefficients easier to interpret.

Categorical coefficients therefore represent differences relative to their omitted reference categories.

This interpretability-oriented specification differs from the Day 12 Ridge pipeline because numerical variables are standardized and categorical variables use reference-category encoding. Its predictive performance therefore needs to be evaluated directly before treating it as the final deployment model.

### Coefficient Stability Analysis

A separate Ridge pipeline was fit in each of the five group-aware folds.

The fitted transformed feature names and Ridge coefficients were extracted from every fold.

After reference-category encoding:

- Final transformed features: {total_feature_count}
- Sign-stable transformed features: {stable_feature_count}
- Sign-unstable transformed features: {total_feature_count - stable_feature_count}

### Strongest Stable Numeric Patterns

#### Stress Level

Mean fold coefficient:

- **-0.8330**

Final full-data coefficient:

- **-0.8491**

Stress Level had the strongest numeric coefficient and remained negative in every fold.

Within this standardized Ridge model, higher Stress Level was therefore consistently associated with lower predicted Quality of Sleep, conditional on the other included predictors.

This association should not be interpreted causally.

#### Age

Mean fold coefficient:

- **+0.4751**

Final full-data coefficient:

- **+0.4886**

Age remained positive across all folds and showed low coefficient variability.

This makes Age one of the clearest stable positive numerical signals in the deployment model.

#### Heart Rate

Mean fold coefficient:

- **-0.0990**

Final full-data coefficient:

- **-0.0823**

Heart Rate remained negative across all folds, although its coefficient was considerably smaller than those of Stress Level and Age.

#### Physical Activity Level

Mean fold coefficient:

- **+0.0610**

Final full-data coefficient:

- **+0.0539**

Physical Activity Level remained positive across all folds, but its independent contribution was relatively small.

#### Daily Steps

Mean fold coefficient:

- **+0.0384**

Final full-data coefficient:

- **+0.0398**

Daily Steps had a very small coefficient and crossed zero across folds.

It is therefore considered a weak and unstable independent Ridge contributor after controlling for the other deployment predictors.

### Categorical Coefficient Interpretation

Categorical variables were encoded using reference-category coding.

Therefore, occupation, gender, and BMI coefficients represent contrasts relative to omitted reference categories rather than absolute effects.

Several categorical contrasts were large and sign-stable.

However, coefficient magnitude should not be compared mechanically across all transformed features because standardized numerical variables and binary categorical indicators are not represented on exactly the same scale.

Some categorical levels were absent from individual training folds, causing their coefficient fold count to be smaller than five.

### Unstable Transformed Features

The transformed features whose coefficient signs changed across folds were:

- `categorical__Occupation_Lawyer`
- `categorical__Occupation_Manager`
- `numeric__Daily Steps`

These terms should be interpreted cautiously.

A sign change does not automatically mean the original predictor should be removed, especially for categorical variables represented through multiple one-hot contrasts.

### Final Full-Data Ridge Coefficients

The final Ridge model was fit on the full deployment dataset after preprocessing.

The strongest coefficient magnitudes included:

- Stress Level: approximately **-0.849**
- Sales Representative occupation contrast: approximately **-0.674**
- Overweight BMI contrast: approximately **-0.643**
- Salesperson occupation contrast: approximately **-0.575**
- Teacher occupation contrast: approximately **-0.545**
- Obese BMI contrast: approximately **-0.498**
- Age: approximately **+0.489**
- Male gender contrast: approximately **+0.387**

These values describe model associations within this dataset and should not be interpreted as causal effects.

### Interpretation

Day 13 shows that the deployment-oriented Ridge model is not relying equally on every predictor.

A relatively small group of stable signals drives much of the fitted linear structure.

The clearest numerical signals are:

- Stress Level
- Age
- Heart Rate
- Physical Activity Level

Daily Steps contributes little independent information under the current Ridge specification.

The coefficient analysis also reinforces the need to interpret categorical effects carefully because each one-hot coefficient is defined relative to an omitted reference category.

### Feature Specification Decision

The current 8-feature deployment set will be retained for the next stage.

Daily Steps is flagged as a weak and unstable contributor, but it will not yet be removed solely on the basis of one coefficient-stability analysis.

Occupation will also remain because instability in a small number of occupation contrasts does not imply that the entire categorical feature is uninformative.

The current final pre-tuning deployment feature specification is therefore:

- Gender
- Age
- Occupation
- Physical Activity Level
- Stress Level
- BMI Category
- Heart Rate
- Daily Steps

### Limitations

- The dataset contains only 374 observations.
- Only {len(deployment_unique_profiles)} unique deployment predictor profiles are present.
- Coefficient stability was evaluated across only five GroupKFold partitions.
- Some categorical levels were absent from certain training folds.
- Ridge coefficients describe conditional model associations, not causal effects.
- Categorical coefficients depend on the choice of omitted reference category.
- Numerical coefficients were standardized, while categorical indicators remained binary.
- Exact-profile grouping does not establish external or unseen-person generalization.
- No external validation dataset was available.
- The dataset may not represent athletic populations specifically.

### Files Generated

Reports:

- `reports/ridge_fold_coefficients.csv`
- `reports/ridge_coefficient_stability_summary.csv`
- `reports/ridge_final_coefficient_interpretation.csv`

Figure:

- `figures/ridge_coefficient_stability.png`

Notebook/script:

- `notebooks/11_ridge_interpretation_and_feature_specification.py`

### Key Conclusion

The interpretability-oriented Ridge specification shows stable coefficient signs for most transformed features across profile-separated folds.

Stress Level and Age are the clearest stable numerical signals.

Daily Steps appears weak and unstable, while a small number of occupation contrasts also change sign.

Because this standardized, reference-coded Ridge specification differs from the Day 12 predictive pipeline, its predictive performance must be re-evaluated on the same group-aware folds before final model selection.

The current 8-feature deployment specification will therefore be retained provisionally rather than being reduced on coefficient magnitude alone.

### Next Steps

- Compare the current 8-feature Ridge model with a reduced specification that removes weak or questionable predictors.
- Test whether removing Daily Steps materially changes repeated group-aware performance.
- Reassess whether Stress Level and Heart Rate are truly available at the intended prediction time.
- Freeze the final feature specification.
- Perform modest Ridge hyperparameter tuning only after the feature specification is finalized.
- Package the final preprocessing and model pipeline after tuning.
"""

print(
    journal_entry
)


# %%
# CELL 16 — Final Day 13 verification

expected_report_files = [
    "ridge_fold_coefficients.csv",
    "ridge_coefficient_stability_summary.csv",
    "ridge_final_coefficient_interpretation.csv",
]

expected_figure_files = [
    "ridge_coefficient_stability.png",
]

print(
    "DAY 13 FINAL VERIFICATION"
)

print(
    "-" * 50
)

print(
    "Dataset rows:",
    len(data),
)

print(
    "Deployment profiles:",
    len(deployment_unique_profiles),
)

print(
    "Maximum profile overlap:",
    maximum_profile_overlap,
)

print(
    "Final transformed features:",
    total_feature_count,
)

print(
    "Sign-stable transformed features:",
    stable_feature_count,
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
    "\nDay 13 analysis completed successfully."
)


# %%
# CELL 17 — Update Day 13 research journal entry

journal_path = (
    PROJECT_ROOT
    / "docs"
    / "research_journal.md"
)

existing_journal = journal_path.read_text(
    encoding="utf-8"
)

day_13_heading = (
    "## Day 13 — Ridge Interpretation and Final Feature Specification"
)

if day_13_heading in existing_journal:
    day_13_start = existing_journal.index(
        day_13_heading
    )

    next_section_marker = (
        "\n\n---\n\n## "
    )

    next_section_start = existing_journal.find(
        next_section_marker,
        day_13_start + len(day_13_heading),
    )

    if next_section_start == -1:
        updated_journal = (
            existing_journal[
                :day_13_start
            ].rstrip()
            + "\n\n"
            + journal_entry.strip()
            + "\n"
        )
    else:
        updated_journal = (
            existing_journal[
                :day_13_start
            ]
            + journal_entry.strip()
            + existing_journal[
                next_section_start:
            ]
        )

    journal_path.write_text(
        updated_journal,
        encoding="utf-8",
    )

    print(
        "Day 13 research journal entry updated."
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
        "Day 13 appended to research journal."
    )

print(
    "Journal:",
    journal_path
)
