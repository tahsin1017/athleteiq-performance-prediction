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