# AthleteIQ Research Journal

## Day 1

### Objective

Set up the AthleteIQ project structure and development environment.

### Completed

- Created the AthleteIQ workspace.
- Organized folders for Project 1.
- Added standard project files (README, LICENSE, requirements, etc.).
- Prepared VS Code for development.

### What I Learned

A well-organized project structure makes collaboration, maintenance, and reproducibility much easier.

### Questions

- Which dataset should we use?
- What exact problem are we trying to solve?

### Next Steps

Choose a suitable dataset and define the research question.

---

## Day 2

### Objective

Load the dataset into Python and perform an initial exploration.

### Tasks Completed

- Imported pandas.
- Loaded the CSV dataset.
- Displayed the first five rows.
- Checked dataset dimensions.
- Reviewed column names.
- Examined data types.
- Generated summary statistics.
- Checked for missing values.
- Checked for duplicate records.

### Key Observations

- Number of rows: **374**
- Number of columns: **13**
- Missing values were found in the `Sleep Disorder` column.
- `Sleep Disorder` contains **219 missing values**.
- No duplicate records were identified.
- Numerical variables include Age, Sleep Duration, Quality of Sleep, Physical Activity Level, Stress Level, Heart Rate, and Daily Steps.
- The dataset contains both numerical and categorical variables.

### Reflection

Today I learned how pandas reads CSV files into a DataFrame and why understanding the dataset is the first step before building any machine learning model.

### Questions

- Which feature should become the prediction target?
- Which features are likely to be the most influential?

---

## Day 3

### Objective

Perform exploratory data analysis (EDA) to understand the distributions of key variables and identify relationships between sleep, stress, physical activity, and heart rate.

### Tasks Completed

- Created a distribution plot for Sleep Duration.
- Created a distribution plot for Quality of Sleep.
- Compared Sleep Duration across genders using a boxplot.
- Created a distribution plot for Heart Rate.
- Generated a correlation heatmap for numerical variables.
- Generated a pairplot to examine relationships between important variables.
- Added written observations based on the visualizations.
- Saved visualizations in the `figures/` directory.

### Key Findings

- Sleep Duration and Quality of Sleep showed a strong positive correlation (`r = 0.88`).
- Stress Level and Quality of Sleep showed a very strong negative correlation (`r = -0.90`).
- Stress Level and Sleep Duration showed a strong negative correlation (`r = -0.81`).
- Physical Activity Level and Daily Steps showed a strong positive correlation (`r = 0.77`).
- Stress Level and Heart Rate showed a moderately strong positive correlation (`r = 0.67`).
- Most participants reported approximately 6–8 hours of sleep.
- Quality of Sleep was concentrated around scores of 6–8, with scores of 8 and 6 being particularly common.
- No duplicate records were identified.
- The Sleep Disorder variable contains missing values.

### Interpretation

The exploratory analysis suggests that sleep duration, sleep quality, and stress level are closely associated in this dataset. Longer sleep duration is associated with higher reported sleep quality, while higher stress levels are associated with shorter sleep duration and lower sleep quality.

Physical activity level also shows a strong relationship with daily step count. Additionally, higher stress levels are associated with higher heart rates.

These findings represent correlations within the dataset and do not establish causal relationships.

### Limitations Identified

- The dataset contains only 374 observations.
- The dataset is observational, so causal conclusions cannot be drawn.
- Several variables are self-reported and may contain measurement or reporting bias.
- The dataset contains missing values in the Sleep Disorder variable.
- The dataset may not represent athletic populations specifically, which will need to be considered when interpreting findings for AthleteIQ.

### Questions Raised

- Which variable should ultimately be used as the prediction target?
- Should the project predict sleep quality, sleep disorder risk, or another performance-related outcome?
- Which features provide the most useful predictive information?
- How well will classical machine learning models perform on this dataset?

### Next Steps

- Define the machine learning problem and prediction target.
- Perform feature engineering and preprocessing.
- Separate features and target variables.
- Establish a baseline model.
- Compare multiple classical machine learning algorithms.
- Evaluate model performance using appropriate metrics.

---

## Day 4

### Objective

Define the prediction problem and prepare the dataset for machine learning.

### Research Question

Can lifestyle, behavioral, and physiological factors be used to predict sleep quality, and which factors contribute most strongly to the prediction?

### Target Variable

The target variable selected for the project is:

**`Quality of Sleep`**

The target represents the participant's reported sleep quality on a scale from 4 to 9.

### Candidate Features

The candidate predictor variables are:

- Age
- Gender
- Occupation
- Sleep Duration
- Physical Activity Level
- Stress Level
- BMI Category
- Heart Rate
- Daily Steps
- Sleep Disorder
- Systolic BP
- Diastolic BP

`Person ID` will not be used as a predictive feature because it is only an identifier.

### Tasks Completed

- Created a separate preprocessing notebook named `02_data_preprocessing.ipynb`.
- Loaded the raw dataset.
- Created a copy of the raw DataFrame for preprocessing.
- Removed the `Person ID` identifier.
- Standardized BMI categories by combining `Normal Weight` with `Normal`.
- Split the original `Blood Pressure` column into `Systolic BP` and `Diastolic BP`.
- Removed the original text-based `Blood Pressure` column.
- Investigated missing values.
- Replaced missing `Sleep Disorder` values with `None`.
- Checked the dataset for remaining missing values.
- Checked for duplicate records.
- Inspected the final processed dataset.
- Saved the processed dataset to `data/processed/processed_sleep_health_dataset.csv`.
- Verified that the saved processed dataset could be loaded successfully.

### Preprocessing Decisions

#### Person ID

`Person ID` was removed because it is an identifier rather than a meaningful predictive feature.

#### BMI Category

The dataset contained both `Normal` and `Normal Weight` categories. These were standardized into a single `Normal` category to avoid treating equivalent categories as separate groups.

#### Blood Pressure

The original `Blood Pressure` variable was stored as text in the form `systolic/diastolic`, such as `126/83`.

It was separated into two numerical variables:

- `Systolic BP`
- `Diastolic BP`

This makes the information easier to use in machine learning models.

#### Sleep Disorder

The `Sleep Disorder` variable contained missing values. Missing values were replaced with `None`, representing participants without a recorded sleep disorder.

### Final Dataset

The processed dataset contains:

- **374 observations**
- **13 columns**

The final variables are:

- Gender
- Age
- Occupation
- Sleep Duration
- Quality of Sleep
- Physical Activity Level
- Stress Level
- BMI Category
- Heart Rate
- Daily Steps
- Sleep Disorder
- Systolic BP
- Diastolic BP

### Reflection

Today I learned that preprocessing is an important step between exploratory analysis and machine learning. Raw datasets often contain identifiers, inconsistent category names, text-based numerical information, and missing values that need to be handled before models can be trained.

I also learned that preprocessing decisions should be documented rather than performed without explanation, because these decisions can affect the results of the machine learning models.

### Questions

- How should categorical variables be encoded?
- Should numerical features be standardized?
- Which machine learning algorithm should be used as the baseline?
- How much predictive performance can be achieved using these features?

### Next Steps

- Encode categorical variables.
- Separate the features (`X`) from the target (`y`).
- Split the dataset into training and testing sets.
- Apply appropriate feature scaling where necessary.
- Build a baseline machine learning model.
- Compare multiple machine learning algorithms.
- Evaluate model performance using appropriate metrics.

---

## Day 5

### Objective

Transform the cleaned dataset into a machine-learning-ready feature matrix and prepare separate training and testing datasets.

### Research Question

Can lifestyle, behavioral, and physiological factors be used to predict sleep quality, and which factors contribute most strongly to the prediction?

### Tasks Completed

- Created `03_feature_engineering.ipynb`.
- Loaded the processed dataset from Day 4.
- Defined `Quality of Sleep` as the prediction target.
- Separated the target variable (`y`) from the predictor variables (`X`).
- Identified numerical and categorical features.
- Applied one-hot encoding to categorical variables.
- Combined numerical and encoded categorical features into a single feature matrix.
- Verified that all features were numeric.
- Checked for missing values after feature engineering.
- Split the data into 80% training and 20% testing sets.
- Used `random_state=42` to make the train-test split reproducible.
- Saved `X_train`, `X_test`, `y_train`, and `y_test` to the `data/processed/` directory.
- Verified that the saved datasets could be loaded successfully.

### Feature Engineering Decisions

Categorical variables such as `Gender`, `Occupation`, `BMI Category`, and `Sleep Disorder` were converted into numerical representations using one-hot encoding.

One-hot encoding was selected because these categories do not have a natural numerical ordering.

The target variable, `Quality of Sleep`, was kept separate from the feature matrix and was not encoded as a predictor.

### Dataset Split

The dataset was divided into:

- 80% training data
- 20% testing data

A fixed random state of 42 was used to ensure that the split can be reproduced.

### Reflection

Today I learned how raw categorical information must be transformed before it can be used by machine learning algorithms. I also learned the importance of separating the target variable from the predictor variables and keeping a test set completely separate for later evaluation.

Feature engineering increased the number of usable model features because categorical variables were expanded into multiple binary indicator variables.

### Questions

- Which regression model will provide a useful baseline?
- How accurately can the selected features predict Quality of Sleep?
- Which features will contribute most strongly to model predictions?
- Will more complex models outperform a simple baseline?

### Next Steps

- Establish a baseline machine learning model.
- Train the first regression model.
- Generate predictions on the test set.
- Evaluate model performance using appropriate regression metrics.
- Compare additional machine learning algorithms.


## Day 6

### Objective

Build and evaluate baseline regression models for predicting Quality of Sleep using the processed AthleteIQ dataset.

### Tasks Completed

- Loaded the processed training and testing datasets.
- Verified the feature and target shapes.
- Confirmed that all predictor variables are numeric.
- Trained a Dummy Regressor as a baseline.
- Trained a Linear Regression model.
- Trained a Ridge Regression model.
- Trained a Random Forest Regressor.
- Evaluated models using Mean Absolute Error (MAE), Root Mean Squared Error (RMSE), and R².
- Compared model performance.
- Identified the best-performing baseline model.
- Examined actual versus predicted Quality of Sleep values.
- Saved baseline model results to `reports/baseline_model_results.csv`.
- Restarted the notebook kernel and ran the complete notebook from top to bottom without errors.

### Dataset Split

The dataset was divided into training and testing sets using an 80/20 split.

- `X_train`: 299 rows × 27 features
- `X_test`: 75 rows × 27 features
- `y_train`: 299 observations
- `y_test`: 75 observations

### Model Results

| Model | MAE | RMSE | R² |
|---|---:|---:|---:|
| Dummy Regressor | 1.0862 | 1.2550 | -0.0441 |
| Linear Regression | 0.1198 | 0.2251 | 0.9664 |
| Ridge Regression | 0.1244 | 0.2063 | 0.9718 |
| Random Forest | 0.0391 | 0.1338 | 0.9881 |

### Key Findings

- The Dummy Regressor performed poorly, with an R² of -0.0441, providing a useful baseline for comparison.
- Linear Regression performed substantially better than the dummy baseline, achieving an R² of 0.9664.
- Ridge Regression achieved an R² of 0.9718 and a lower RMSE than Linear Regression.
- Random Forest achieved the best overall performance with an R² of 0.9881.
- Random Forest also achieved the lowest MAE (0.0391) and RMSE (0.1338).
- The Random Forest model therefore performed best among the baseline models evaluated.
- The actual-versus-predicted results show that the model predictions were very close to the observed Quality of Sleep values for the displayed test examples.

### Interpretation

The baseline modeling results indicate that the available lifestyle, behavioral, and physiological features contain substantial predictive information about Quality of Sleep in this dataset.

The Random Forest model produced the strongest performance among the tested models, explaining approximately 98.8% of the variance in the test-set target values according to R².

However, these results should not be interpreted as evidence that the model will perform equally well on completely new populations. The dataset is relatively small, and further validation and model analysis are required before drawing strong conclusions about generalization.

### Validation

The notebook was tested from a fresh kernel.

- Kernel restarted successfully.
- The complete notebook was executed from top to bottom.
- All cells ran without errors.
- Model results were reproduced successfully.

### Reflection

Today I learned how baseline machine learning models can be compared using multiple regression metrics. I also learned why a simple Dummy Regressor is useful as a reference point before evaluating more sophisticated models.

The Random Forest model performed best in this initial comparison, but additional analysis is needed to understand which features are driving its predictions and whether the model generalizes well.

### Questions Raised

- Which features contribute most strongly to the Random Forest predictions?
- Does the Random Forest model generalize well beyond this dataset?
- Are there signs of overfitting?
- How does model performance change with cross-validation?
- Can hyperparameter tuning improve the model further?
- Which model should ultimately be selected for the AthleteIQ project?

### Next Steps

- Analyze feature importance.
- Investigate potential model overfitting.
- Perform cross-validation.
- Tune the strongest-performing models.
- Compare tuned models against the baseline results.
- Select and document the final candidate model.


## Day 7

### Objective

Evaluate the baseline machine learning models using cross-validation and held-out test data to determine whether the strong Day 6 results are reliable and whether the models show signs of overfitting.

### Tasks Completed

- Created `05_model_evaluation.ipynb`.
- Loaded the training and testing datasets.
- Recreated the baseline regression models.
- Configured 5-fold cross-validation with shuffling and `random_state=42`.
- Evaluated Dummy Regression, Linear Regression, Ridge Regression, and Random Forest.
- Calculated cross-validation MAE, RMSE, and R².
- Compared training and cross-validation R².
- Evaluated the Random Forest model on the held-out test dataset.
- Compared Random Forest cross-validation performance with test-set performance.
- Investigated potential overfitting using the difference between training and validation performance.
- Created an actual-versus-predicted visualization for Random Forest.
- Saved the evaluation visualization to the `figures/` directory.
- Saved cross-validation results to the `reports/` directory.
- Restarted the notebook kernel and successfully ran the complete notebook from top to bottom without errors.

### Evaluation Strategy

Five-fold cross-validation was used on the training dataset.

The cross-validation configuration was:

- 5 folds
- Shuffling enabled
- `random_state=42`

The held-out test set was kept separate from the cross-validation procedure and used for final evaluation.

### Cross-Validation Results

| Model | CV MAE | CV RMSE | CV R² | Train R² |
|---|---:|---:|---:|---:|
| Dummy Regressor | 1.0327 | 1.1800 | -0.0019 | 0.0000 |
| Linear Regression | 0.1567 | 0.2690 | 0.9463 | 0.9657 |
| Ridge Regression | 0.1596 | 0.2620 | 0.9494 | 0.9643 |
| Random Forest | 0.0513 | 0.1600 | 0.9796 | 0.9956 |

### Best Cross-Validation Model

Random Forest achieved the strongest cross-validation performance.

Its results were:

- CV MAE: 0.0513
- CV RMSE: 0.1600
- CV R²: 0.9796
- Mean Training R²: 0.9956

### Held-Out Test Performance

The Random Forest model achieved:

- MAE: 0.0391
- RMSE: 0.1338
- R²: 0.9881

### Cross-Validation vs Test Performance

| Metric | Cross-Validation | Test Set |
|---|---:|---:|
| MAE | 0.0513 | 0.0391 |
| RMSE | 0.1600 | 0.1338 |
| R² | 0.9796 | 0.9881 |

The held-out test performance was consistent with the strong performance observed during cross-validation.

### Overfitting Analysis

The Random Forest model achieved:

- Mean Training R²: **0.9956**
- Mean Cross-Validation R²: **0.9796**
- Held-Out Test R²: **0.9881**
- Training-CV R² Gap: **0.0160**

The training performance is slightly higher than the cross-validation performance, which is expected for a flexible model such as Random Forest.

However, the relatively small R² gap of 0.0160 does not indicate severe overfitting based on this evaluation.

The strong held-out test result is also consistent with the cross-validation performance.

Further validation is still required because the dataset contains only 374 observations.

### Key Findings

- Random Forest remained the strongest model after 5-fold cross-validation.
- Random Forest achieved a mean cross-validation R² of **0.9796**.
- The held-out test R² was **0.9881**.
- Linear Regression achieved a cross-validation R² of **0.9463**.
- Ridge Regression achieved a cross-validation R² of **0.9494**.
- The Dummy Regressor failed to explain meaningful variance in the target.
- Random Forest achieved substantially lower prediction errors than the other evaluated models.
- The training-CV R² gap of **0.0160** does not suggest severe overfitting.
- Actual-versus-predicted values were generally close to the ideal diagonal reference line.

### Interpretation

The cross-validation results strengthen the evidence that the selected lifestyle, behavioral, and physiological features contain substantial predictive information about Quality of Sleep within this dataset.

Random Forest achieved the strongest overall performance and remained consistent across training, cross-validation, and held-out testing.

The similarity between cross-validation and test performance suggests that the strong Day 6 result was not caused solely by one favorable train-test split.

However, an R² close to 0.99 is unusually high for many real-world behavioral prediction problems. Therefore, model interpretation is now particularly important.

Before considering the model final, the next stage should investigate which features are driving the predictions and whether any variables provide overly direct or redundant information about the target.

### Visualization

An actual-versus-predicted plot was created for the Random Forest model.

Most predictions were located on or close to the diagonal reference line, indicating that predicted Quality of Sleep values were generally close to their observed values in the held-out test dataset.

Some small prediction errors remained, particularly for a few observations with lower or intermediate sleep-quality scores.

The visualization was saved as:

`figures/random_forest_actual_vs_predicted.png`

### Saved Results

The following evaluation files were created:

- `reports/cross_validation_results.csv`
- `reports/random_forest_cv_vs_test.csv`

These files preserve the evaluation results for later comparison with tuned models.

### Validation

The notebook was tested from a fresh kernel.

- Kernel restarted successfully.
- The complete notebook was executed from top to bottom.
- All cells ran without errors.
- Cross-validation results were reproduced successfully.
- Held-out test results were reproduced successfully.
- Evaluation files and figures were generated successfully.

### Reflection

Today I learned why cross-validation is an important part of reliable machine learning evaluation.

A single train-test split provides only one estimate of model performance, while cross-validation evaluates the model across multiple validation subsets.

I also learned how comparing training, cross-validation, and held-out test performance can help identify potential overfitting.

The Random Forest model remains the strongest candidate, but its unusually high predictive performance means that understanding why it performs so well is now more important than immediately attempting to increase its accuracy.

### Questions Raised

- Which features contribute most strongly to Random Forest predictions?
- Does feature importance agree with the relationships identified during exploratory data analysis?
- Is any feature providing unusually direct information about Quality of Sleep?
- Could strongly related variables such as Sleep Duration or Stress Level be responsible for much of the model performance?
- Can a simpler model achieve comparable predictive performance?
- How stable are feature importance estimates?
- Should any predictors be excluded before final model selection?

### Limitations

- The dataset contains only 374 observations.
- Cross-validation improves reliability but does not replace evaluation on an independent external dataset.
- The dataset may not represent athletic populations specifically.
- The target and several predictor variables may be strongly related by construction or measurement.
- Strong predictive performance within this dataset does not guarantee equivalent performance on new populations.

### Next Steps

- Analyze Random Forest feature importance.
- Investigate possible target leakage or overly direct predictors.
- Compare feature importance with the EDA findings.
- Visualize the most influential features.
- Evaluate model performance after removing potentially dominant features if necessary.
- Begin model interpretation.
- Prepare for hyperparameter tuning only after the model's behavior is better understood.



---

## Day 8

### Objective

Interpret the Random Forest model, identify which features contribute most strongly to its predictions, and investigate possible explanations for the unusually high predictive performance observed during baseline evaluation and cross-validation.

### Tasks Completed

- Created `06_feature_importance_and_leakage.ipynb`.
- Loaded the processed training and testing datasets.
- Recreated the Random Forest model used during baseline evaluation.
- Confirmed that `Quality of Sleep` was not present in the predictor matrix.
- Confirmed that the feature matrices contained no missing values.
- Confirmed that all model features were numeric.
- Reproduced the held-out Random Forest performance from Day 7.
- Calculated impurity-based Random Forest feature importance.
- Calculated permutation importance using the held-out test dataset.
- Compared model importance with feature-target correlations.
- Investigated repeated predictor profiles within the training and testing datasets.
- Investigated exact predictor-profile overlap between training and testing data.
- Examined whether identical predictor profiles were associated with different Quality of Sleep values.
- Performed feature-ablation experiments.
- Evaluated model performance after removing Sleep Duration.
- Evaluated model performance after removing Stress Level.
- Evaluated model performance after removing both Sleep Duration and Stress Level.
- Evaluated the effect of removing Sleep Disorder features.
- Saved interpretation and diagnostic results to the `reports/` directory.
- Saved feature-importance and ablation visualizations to the `figures/` directory.
- Restarted the notebook kernel and successfully executed the complete notebook from top to bottom without errors.

### Random Forest Performance

The Random Forest reproduced the strong held-out performance observed during Day 7:

- Test MAE: **0.0391**
- Test RMSE: **0.1338**
- Test R²: **0.9881**

This confirmed that the model used for the Day 8 interpretation analysis was consistent with the model previously evaluated.

### Feature Importance

The impurity-based Random Forest feature importance showed that the model relied most strongly on:

1. `Sleep Duration` — **0.7941**
2. `Stress Level` — approximately **0.1009**
3. `Occupation_Doctor` — approximately **0.0356**
4. `Heart Rate` — approximately **0.0340**
5. `Daily Steps` — approximately **0.0090**

`Sleep Duration` dominated the impurity-based feature importance, accounting for approximately 79% of the total importance assigned by the Random Forest.

### Permutation Importance

Permutation importance produced a similar ranking.

The most important features were:

1. `Sleep Duration` — **1.1514**
2. `Stress Level` — **0.1247**
3. `Occupation_Doctor` — **0.0550**
4. `Heart Rate` — **0.0473**
5. `Daily Steps` — **0.0174**
6. `Age` — **0.0141**

The permutation importance value for Sleep Duration indicates that randomly shuffling this feature caused a very large reduction in model R².

Because permutation importance measures the loss in predictive performance after disrupting a feature, values can exceed 1.0 when shuffling a highly informative feature causes the model to perform substantially worse than its original prediction performance.

The agreement between impurity-based importance and permutation importance strengthens the conclusion that Sleep Duration is the dominant feature used by the Random Forest.

### Feature Importance Interpretation

The importance analysis does not establish that Sleep Duration causes Quality of Sleep.

Instead, it shows that the Random Forest depends heavily on Sleep Duration when making predictions within this dataset.

Other variables such as Stress Level, Heart Rate, occupation, Daily Steps, and Age also contribute information, but their contributions are substantially smaller.

The model therefore appears to obtain most of its predictive power from a relatively small subset of features.

### Relationship to Exploratory Analysis

The feature-importance results were consistent with the relationships identified during exploratory data analysis.

The strongest numerical correlations with Quality of Sleep in the training data were:

- Stress Level: **r = -0.8946**
- Sleep Duration: **r = 0.8842**
- Heart Rate: **r = -0.6408**
- Age: **r = 0.4523**
- `Occupation_Engineer`: **r = 0.3980**

These findings broadly agree with the Day 3 exploratory analysis, where Sleep Duration and Stress Level were identified as the strongest relationships with Quality of Sleep.

An important observation is that Stress Level has a slightly stronger absolute linear correlation with Quality of Sleep than Sleep Duration, while the Random Forest assigns much greater predictive importance to Sleep Duration.

This demonstrates that correlation and model feature importance measure different types of relationships.

### Train-Test Profile Analysis

A major finding from Day 8 was the presence of substantial repetition in the predictor data.

The training dataset contained:

- Training rows: **299**
- Unique training predictor profiles: **122**
- Repeated training rows relative to unique profiles: **177**

The test dataset contained:

- Test rows: **75**
- Unique test predictor profiles: **54**
- Repeated test rows relative to unique profiles: **21**

When the training and test sets were compared directly:

- Unique predictor profiles appearing in both training and testing data: **44**
- Test rows whose exact predictor profile appeared during training: **65 / 75**
- Percentage of test rows with a predictor profile previously seen during training: **86.67%**

This is a major methodological finding.

Although the test set was technically held out from model training, most test observations had an exact predictor profile that had already appeared in the training dataset.

Therefore, the random train-test split does not provide a fully independent evaluation of performance on unseen participant profiles.

### Dataset Profile Structure

Across all 374 observations:

- Total unique predictor profiles: **132**
- Repeated predictor profiles: **78**
- Predictor profiles associated with multiple Quality of Sleep values: **0**
- Rows belonging to repeated predictor profiles: **320 / 374**
- Percentage of rows belonging to repeated predictor profiles: **85.56%**

This means that only 132 unique predictor combinations exist across 374 observations.

Furthermore, 320 observations belong to profiles that occur more than once.

Most importantly, none of the repeated predictor profiles were associated with more than one Quality of Sleep value.

In other words, within this dataset, identical predictor profiles always mapped to the same target value.

This structure makes the prediction task substantially easier because repeated profiles provide extremely consistent relationships between the predictor variables and Quality of Sleep.

### Leakage Assessment

No direct target leakage was detected.

`Quality of Sleep` was confirmed to be absent from both `X_train` and `X_test`.

Therefore, the model is not directly using the target variable as an input.

However, the profile analysis revealed an important evaluation issue.

The presence of exact predictor profiles in both training and testing data means that the model is frequently being evaluated on combinations of predictors that it has effectively already encountered during training.

This is better described as **train-test profile overlap** or **non-independent observations** rather than direct target leakage.

The overlap may cause the random train-test evaluation to overestimate the model's ability to generalize to genuinely new participant profiles.

### Feature Ablation Results

Random Forest performance was evaluated after removing selected predictors.

| Feature Set | Number of Features | Mean CV R² | Std CV R² | Change vs All Features |
|---|---:|---:|---:|---:|
| All Features | 27 | 0.9796 | 0.0135 | 0.0000 |
| Without Sleep Disorder | 24 | 0.9784 | 0.0150 | -0.0012 |
| Without Sleep Duration | 26 | 0.9757 | 0.0150 | -0.0040 |
| Without Stress Level | 26 | 0.9728 | 0.0177 | -0.0068 |
| Without Sleep Duration + Stress Level + Sleep Disorder | 22 | 0.9116 | 0.0857 | -0.0680 |
| Without Sleep Duration + Stress Level | 25 | 0.9097 | 0.0877 | -0.0699 |

### Ablation Interpretation

Removing Sleep Disorder had almost no effect on predictive performance.

Mean cross-validation R² decreased from **0.9796** to **0.9784**, a change of only **-0.0012**.

This suggests that Sleep Disorder contributes very little additional predictive information once the other features are available.

Removing Sleep Duration alone reduced R² only slightly, from **0.9796** to **0.9757**.

Removing Stress Level alone also caused only a small decrease, producing an R² of **0.9728**.

These relatively small individual reductions indicate that the model can compensate for the loss of one highly informative feature by relying on correlated or redundant information from other variables.

However, removing both Sleep Duration and Stress Level caused a much larger decrease:

- With all features: **R² = 0.9796**
- Without Sleep Duration and Stress Level: **R² = 0.9097**

This corresponds to a decrease of approximately **0.0699 R²**.

Removing Sleep Disorder in addition to those two features produced an R² of **0.9116**, which was nearly identical.

This further confirms that Sleep Disorder is not a major source of the model's predictive performance.

### Redundant Predictive Information

The ablation results suggest that Sleep Duration and Stress Level contain overlapping predictive information.

Removing either variable individually has only a modest effect because the remaining variables can partially compensate.

Removing both simultaneously causes a substantially larger performance decline.

This is consistent with the strong relationship previously observed between Sleep Duration and Stress Level.

It also explains why feature importance should not be interpreted in isolation when predictors are correlated.

### Feature Availability

Another important issue is whether every predictor would actually be available at the intended time of prediction.

Sleep Duration is a valid predictor if the model is intended to estimate sleep quality using information collected after a participant sleeps.

However, if the intended goal were to predict the upcoming night's sleep quality before the participant goes to sleep, the same night's Sleep Duration would not yet be available.

Similarly, Sleep Disorder may represent clinical information that would not necessarily be available in every deployment setting.

Therefore, the final feature set should depend on a clearly defined prediction scenario.

### Preprocessing Consideration

The current feature-engineering workflow performed categorical encoding before the train-test split.

Because one-hot encoding does not use the target variable, this represents much less risk than target-based preprocessing.

However, best machine-learning practice is to learn preprocessing transformations from the training data only.

A later stage of the project should therefore rebuild preprocessing and model training using a scikit-learn `Pipeline` or `ColumnTransformer`.

This will ensure that the complete machine-learning workflow follows a stricter train-only preprocessing design.

### Visualizations

Three new visualizations were created:

- `figures/random_forest_feature_importance.png`
- `figures/random_forest_permutation_importance.png`
- `figures/random_forest_ablation_analysis.png`

The feature-importance plots show that Sleep Duration is the dominant predictor under both importance methods.

The ablation plot shows that model performance remains extremely strong when individual features are removed but decreases more noticeably when both Sleep Duration and Stress Level are removed together.

### Saved Results

The following Day 8 reports were created:

- `reports/random_forest_feature_importance.csv`
- `reports/random_forest_permutation_importance.csv`
- `reports/feature_target_correlations.csv`
- `reports/data_leakage_profile_checks.csv`
- `reports/random_forest_ablation_results.csv`

These reports preserve the interpretation, profile-overlap, correlation, and feature-ablation results for future analysis.

### Key Findings

- Sleep Duration was the dominant Random Forest predictor.
- Sleep Duration had an impurity-based importance of **0.7941**.
- Sleep Duration had a permutation importance of **1.1514**.
- Stress Level was the second-most important predictor.
- Feature-importance results broadly agreed with the relationships found during exploratory data analysis.
- No direct inclusion of the target variable was detected.
- The dataset contains substantial repetition in predictor profiles.
- Only **132 unique predictor profiles** exist across **374 observations**.
- **320 of 374 observations (85.56%)** belong to repeated predictor profiles.
- **65 of 75 test rows (86.67%)** had an exact predictor profile already present in the training data.
- **44 unique feature profiles** appeared in both the training and test sets.
- No identical predictor profile was associated with multiple Quality of Sleep values.
- Removing Sleep Disorder had almost no effect on model performance.
- Removing either Sleep Duration or Stress Level individually caused only a small decline.
- Removing both Sleep Duration and Stress Level reduced mean CV R² from **0.9796** to approximately **0.9097**.
- The high baseline and test scores are therefore likely influenced by both strong feature-target relationships and substantial repetition within the dataset.

### Interpretation

Day 8 significantly changed the interpretation of the earlier model results.

The Random Forest remains highly accurate under the existing random-split evaluation, but its R² of approximately 0.99 should not be interpreted as evidence that the model would achieve the same accuracy on completely new participant profiles.

The dataset contains extensive repetition, and most held-out test observations have predictor profiles that are already represented in the training set.

This means that the current test set is not fully independent at the profile level.

The model also relies heavily on Sleep Duration and Stress Level, although correlated predictors provide enough redundant information that removing either one individually has only a small effect.

Therefore, the next stage should focus on obtaining a more conservative estimate of generalization rather than attempting to increase model accuracy further.

### Limitations

- The dataset contains only 374 observations.
- Only 132 unique predictor profiles are present.
- Approximately 85.56% of all rows belong to repeated predictor profiles.
- Approximately 86.67% of test rows have an exact profile already represented in training.
- Random train-test splitting does not guarantee independence between repeated profiles.
- Feature importance measures model dependence rather than causation.
- Impurity-based Random Forest importance can be affected by correlated predictors.
- Permutation importance was calculated on a relatively small test set of 75 observations.
- Predictor availability depends on the intended real-world prediction scenario.
- The current preprocessing workflow was not implemented as a train-only pipeline.
- Performance on this dataset should not automatically be generalized to external populations.

### Validation

The Day 8 notebook was tested from a fresh kernel.

- Kernel restarted successfully.
- The complete notebook executed from top to bottom.
- All cells ran without errors.
- Random Forest performance was reproduced successfully.
- Feature importance was generated successfully.
- Permutation importance was generated successfully.
- Profile-overlap diagnostics were generated successfully.
- Feature-ablation experiments completed successfully.
- All expected reports were saved.
- All expected figures were saved.
- The final notebook output reported:

`Day 8 analysis completed successfully.`

### Reflection

Today I learned that unusually strong model performance should be investigated rather than automatically treated as a successful final result.

I learned the distinction between direct target leakage and train-test profile overlap.

The target itself was not present in the predictor matrix, so no obvious direct target leakage was found. However, the discovery that 86.67% of test rows shared an exact predictor profile with training data revealed that the existing test set is much less independent than initially assumed.

I also learned that feature importance and correlation answer different questions. Sleep Duration was overwhelmingly dominant in the Random Forest even though Stress Level had a slightly stronger absolute linear correlation with Quality of Sleep.

Feature ablation also demonstrated how correlated variables can compensate for one another. Removing either Sleep Duration or Stress Level individually caused relatively little performance loss, while removing both caused a much larger decline.

This analysis showed that understanding the structure of the data and the evaluation design is just as important as obtaining a high accuracy score.

### Questions Raised

- How much will performance decrease when identical predictor profiles are prevented from appearing across training and validation data?
- What is the best way to create a group-aware validation strategy for repeated profiles?
- How much of the original R² is explained by train-test profile overlap?
- Can Random Forest still generalize well to genuinely unseen predictor combinations?
- Should the project define a prediction scenario that excludes same-night Sleep Duration?
- Should Sleep Disorder remain in the final feature set given its very small additional contribution?
- How should preprocessing be rebuilt using a train-only Pipeline or ColumnTransformer?
- Will simpler models remain competitive under a stricter validation strategy?
- Is hyperparameter tuning still useful after correcting the evaluation methodology?

### Next Steps

- Create a group identifier for identical predictor profiles.
- Use a group-aware train-validation strategy so identical profiles cannot appear in both training and validation data.
- Re-evaluate Random Forest under this stricter validation design.
- Compare group-aware performance with the original random-split performance.
- Quantify how much the original evaluation was affected by repeated profile overlap.
- Define the intended prediction timing and determine which features would realistically be available.
- Rebuild preprocessing using a train-only scikit-learn Pipeline or ColumnTransformer.
- Compare Random Forest with simpler models under the stricter evaluation strategy.
- Delay hyperparameter tuning until the generalization methodology has been strengthened.


## Day 9 — Group-Aware Validation and Unseen-Profile Generalization

### Objective

Evaluate AthleteIQ under a stricter validation design in which identical predictor profiles cannot appear in both training and evaluation data.

Day 8 showed that the original random train/test split contained substantial profile overlap, with 65 of 75 test observations having an exact predictor profile already represented in training.

### Work Completed

- Reconstructed the complete encoded dataset containing 374 observations.
- Confirmed 132 unique predictor profiles.
- Assigned identical predictor combinations to the same profile group.
- Verified that no profile group was associated with multiple Quality of Sleep target values.
- Compared ordinary 5-fold random cross-validation with 5-fold GroupKFold validation.
- Verified zero predictor-profile overlap in every group-aware validation fold.
- Created a profile-separated holdout split using GroupShuffleSplit.
- Evaluated Dummy Regressor, Linear Regression, Ridge Regression, and Random Forest.
- Repeated profile-separated Random Forest evaluation across 20 different holdout splits.

### Key Results

Random Forest under ordinary random 5-fold cross-validation:

- MAE: 0.0432
- RMSE: 0.1478
- R²: 0.9820

Random Forest under group-aware 5-fold cross-validation:

- MAE: 0.0688
- RMSE: 0.2176
- R²: 0.9628
- R² change relative to random CV: -0.0193

Profile-separated Random Forest holdout:

- Training rows: 316
- Test rows: 58
- Training profiles: 105
- Test profiles: 27
- Exact profile overlap: 0
- MAE: 0.2030
- RMSE: 0.4204
- R²: 0.9165

Across 20 repeated profile-separated holdouts:

- Mean MAE: 0.0892
- Mean RMSE: 0.2383
- Mean R²: 0.9519
- R² standard deviation: 0.0343
- Minimum R²: 0.8798
- Maximum R²: 0.9918

### Model Comparison Under Stricter Evaluation

Under the profile-separated holdout, Linear Regression and Ridge Regression slightly outperformed Random Forest:

- Linear Regression R²: 0.9286
- Ridge Regression R²: 0.9282
- Random Forest R²: 0.9165

This shows that model ranking can change when evaluation is performed on genuinely unseen predictor profiles.

### Interpretation

Group-aware validation produced lower performance than ordinary random validation, confirming that repeated predictor profiles had made the original evaluation somewhat optimistic.

However, the Random Forest still achieved strong average performance under stricter evaluation, with a mean R² of 0.9519 across 20 profile-separated holdouts.

The range in R² values, from 0.8798 to 0.9918, shows that model performance depends on which unseen profiles are selected for testing.

The findings should therefore be interpreted as evidence of train-test profile overlap and non-independent observations under random splitting, rather than direct target leakage.

### Limitations

- Group-aware validation removes exact profile overlap but does not establish external real-world or clinical generalization.
- Repeated observations within the same profile remain together inside training folds and can give some profiles greater weight.
- The encoded feature representation was created before the original split, so preprocessing is not yet fully isolated inside each training fold.
- The dataset remains small, with only 132 unique predictor profiles.
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
