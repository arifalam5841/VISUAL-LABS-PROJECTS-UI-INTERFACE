# Crop Yield Prediction Using Random Forest

**Name:** Arif alam  
**Enrollment:** 24111590742

## Overview

This project predicts crop yield from agricultural data using Random Forest Regression. It uses a real-world Indian crop-yield dataset containing crop type, year, season, state, cultivated area, rainfall, fertilizer usage, pesticide usage, and yield.

The purpose is to train a model that learns the relationship between these agricultural factors and crop yield so it can estimate yield for new conditions.

`Production` is intentionally excluded from training because yield is directly related to production and area. Including it would cause target leakage.

## Objectives

- Analyze agricultural records.
- Preprocess numerical and categorical data.
- Apply One-Hot Encoding to categorical columns.
- Train a Random Forest regression model.
- Predict yield from user-entered farming conditions.
- Evaluate performance with regression metrics.
- Identify the most important yield-related features.

## Dataset

The dataset contains 19,689 records.

| Column | Description |
| ------ | ----------- |
| `Crop` | Crop name |
| `Crop_Year` | Year of cultivation |
| `Season` | Agricultural season |
| `State` | Indian state |
| `Area` | Cultivated area |
| `Production` | Total production, not used for training |
| `Annual_Rainfall` | Annual rainfall |
| `Fertilizer` | Fertilizer used |
| `Pesticide` | Pesticide used |
| `Yield` | Target variable |

Training features:

```text
Crop, Crop_Year, Season, State, Area, Annual_Rainfall, Fertilizer, Pesticide
```

Target:

```text
Yield
```

## Algorithm

The project uses `RandomForestRegressor` from Scikit-learn. Random Forest combines predictions from many decision trees to produce a stronger and more stable regression result.

```python
RandomForestRegressor(
    n_estimators=200,
    random_state=42,
    n_jobs=-1
)
```

## Workflow

The program loads the crop-yield CSV file, displays dataset details, checks missing values, removes missing rows, separates input features and target values, one-hot encodes categorical columns, keeps numerical columns unchanged, splits the data in an 80:20 ratio, trains a 200-tree Random Forest model, evaluates predictions, visualizes actual vs predicted yield, displays feature importance, and accepts new agricultural input for yield prediction.

## Evaluation

- MAE: average absolute prediction error.
- MSE: average squared error.
- RMSE: prediction error in the target scale.
- R2 Score: model explanation of yield variation.

## Visualizations

- Actual vs predicted crop yield scatter plot.
- Feature-importance chart.

## Technologies

- Python
- Pandas
- NumPy
- Matplotlib
- Scikit-learn

## Project Files

```text
Crop Yield Prediction/
|-- Crop_yield_prediction.py
|-- crop_yield.csv
|-- requirements.txt
|-- README.md
`-- .gitignore
```

## Setup and Run

```bash
git clone <your-repository-url>
cd Crop-Yield-Prediction
pip install -r requirements.txt
python Crop_yield_prediction.py
```

## Sample Input

```text
Crop: Rice
Crop Year: 2015
Season: Kharif
State: Maharashtra
Area: 500
Annual Rainfall: 1200
Fertilizer: 150000
Pesticide: 2500
```

The model returns an estimated crop yield based on the trained Random Forest model.

## Future Scope

- Tune hyperparameters with GridSearchCV or RandomizedSearchCV.
- Add cross-validation.
- Compare Random Forest with XGBoost and other regressors.
- Include soil and weather-related features.
- Create a Streamlit dashboard.
- Save and reload the model with Joblib.
- Add SHAP-based interpretability.
- Deploy the project as a web app.
