# Air Quality Index Prediction Using XGBoost

**Name:** Arif alam  
**Enrollment:** 24111590742

## Overview

This project estimates the Air Quality Index (AQI) from pollutant concentration values using an XGBoost regression model. The dataset includes pollution-related measurements such as PM2.5, PM10, NO, NO2, NOx, NH3, CO, SO2, O3, Benzene, and Toluene.

The workflow covers data inspection, exploratory visualization, model training, regression evaluation, feature-importance analysis, model saving, and AQI prediction from user-entered pollutant values.

## Aim

- Predict AQI from air-pollution parameters.
- Study how different pollutants relate to AQI.
- Apply XGBoost to a regression-based machine learning problem.
- Measure model performance using standard regression metrics.
- Accept new pollutant values from the user and generate an AQI estimate.
- Convert the predicted AQI into the correct AQI category.

## Dataset

The project uses `air_quality_data.csv`, which contains 29,531 records and 18 columns.

Main columns:

- `City`
- `Date`
- `PM2.5`
- `PM10`
- `NO`
- `NO2`
- `NOx`
- `NH3`
- `CO`
- `SO2`
- `O3`
- `Benzene`
- `Toluene`
- `AQI`
- `AQI_Bucket`
- `Year`
- `Month`
- `Season`

Target variable:

```text
AQI
```

Features used for training:

```text
PM2.5, PM10, NO, NO2, NOx, NH3, CO, SO2, O3, Benzene, Toluene
```

`AQI_Bucket` is not included as an input feature because it is derived from AQI. Using it would create target leakage.

## Method

The program loads the dataset with Pandas, checks structure, missing values, duplicates, and summary statistics, then creates visualizations such as AQI distribution and a correlation heatmap. Selected pollutant columns are separated from the target column, and the data is split into training and testing sets.

An `XGBRegressor` model is trained on the training data and evaluated with MAE, MSE, RMSE, and R2 score. The project also displays feature importance and an actual-versus-predicted AQI graph. After training, the model is saved with Joblib and used for prediction from user input.

## Algorithm

The project uses XGBoost, a gradient-boosting algorithm based on decision trees. It is useful here because AQI depends on non-linear relationships between several pollutants.

```python
XGBRegressor(
    n_estimators=300,
    learning_rate=0.05,
    max_depth=6,
    subsample=0.8,
    colsample_bytree=0.8,
    random_state=42,
    objective="reg:squarederror"
)
```

## Evaluation

- MAE: average absolute difference between actual and predicted AQI.
- MSE: average squared prediction error.
- RMSE: square root of MSE, shown in AQI units.
<!-- - R2 Score: explains how much AQI variation is captured by the model. -->

## Visual Output

- AQI distribution graph
- Correlation heatmap
- Feature-importance chart
- Actual vs predicted AQI plot

## Prediction Example

Example input:

```text
PM2.5: 45
PM10: 80
NO: 10
NO2: 30
NOx: 40
NH3: 20
CO: 0.8
SO2: 15
O3: 40
Benzene: 1.5
Toluene: 5
```

Example output:

```text
Predicted AQI : 92.37
AQI Category  : Satisfactory
```

Actual values may vary based on the trained model.

## AQI Categories

| AQI Range | Category |
| --------: | -------- |
| 0-50 | Good |
| 51-100 | Satisfactory |
| 101-200 | Moderate |
| 201-300 | Poor |
| 301-400 | Very Poor |
| 401+ | Severe |

## Technologies

- Python
- Pandas
- NumPy
- Matplotlib
- Seaborn
- Scikit-learn
- XGBoost
- Joblib

## Project Files

```text
AQI-Prediction/
|-- air_quality_data.csv
|-- AQI-Prediction.py
|-- requirements.txt
|-- README.md
`-- .gitignore
```

## Setup and Run

```bash
git clone <your-repository-link>
cd AQI-Prediction
pip install -r requirements.txt
python AQI-Prediction.py
```

## Possible Enhancements

- Add weather features such as temperature and humidity.
- Encode `City` and use it as a feature.
- Include historical AQI values for time-series prediction.
- Tune XGBoost hyperparameters.
- Add cross-validation.
- Build a GUI or web application.
- Connect real-time air-quality API data.

## Learning Outcomes

This project demonstrates data preprocessing, EDA, regression modeling, XGBoost training, model evaluation, feature importance, visualization, Joblib model saving, user-input prediction, and AQI category classification.
