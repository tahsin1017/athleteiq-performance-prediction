# Data provenance

This project uses the **Sleep Health and Lifestyle Dataset** created by
**Laksika Tharmalingam** and distributed through Kaggle.

Original dataset page:

https://www.kaggle.com/datasets/uom190346a/sleep-health-and-lifestyle-dataset

## Dataset used in this repository

The raw file used in the analysis is:

`data/raw/Sleep_health_and_lifestyle_dataset.csv`

The version used here contains 374 observations and includes demographic,
sleep, lifestyle, cardiovascular, and sleep-disorder variables.

The prediction target in this project is:

`Quality of Sleep`

## Important scope note

The repository name **AthleteIQ** is a project name. The dataset itself does
not establish that the observations represent athletes, so results from this
project should not be described as athlete-specific evidence.

## Participant identity limitation

Repeated rows or identical predictor combinations are treated in this project
as **repeated predictor profiles**.

These groups are not verified participant identities. Group-aware evaluation
therefore tests generalization to unseen predictor profiles, not necessarily
to unseen people.

## Licensing

No license claim is made here for the third-party dataset.

The dataset remains subject to the terms and rights associated with its
original source. Any license applied to original code in this repository
should not be interpreted as relicensing the dataset.
