# Electricity Consumption Prediction Using XGBoost

**Name:** Arif alam  
**Enrollment:** 24111590742

## Overview

This project predicts electricity consumption in kWh using an XGBoost regression model. It uses smart meter data with environmental, historical-consumption, and timestamp-based features.

<!-- The project demonstrates how machine learning can learn electricity usage patterns from temperature, humidity, wind speed, previous consumption, and time-related values. -->

## Objective

The goal is to build a model that estimates electricity consumption from historical smart meter readings and related features.

## Dataset

The project uses the Smart Meter Electricity Consumption Dataset from Kaggle. It contains 5,000 records collected at 30-minute intervals.

| Feature | Description |
| ------- | ----------- |
| `Timestamp` | Date and time of measurement |
| `Electricity_Consumed` | Electricity consumed in kWh, target variable |
| `Temperature` | Temperature at measurement time |
| `Humidity` | Humidity level |
| `Wind_Speed` | Wind speed |
| `Avg_Past_Consumption` | Average previous consumption |
| `Anomaly_Label` | Anomaly indicator, not used for prediction |

## Workflow

The program loads the smart meter dataset, checks missing values, converts `Timestamp` to datetime format, extracts year, month, day, hour, minute, and day-of-week features, and splits the data chronologically to respect the time-series order.

An XGBoost Regressor is trained on environmental, past-consumption, and time-based inputs. The trained model predicts test-set consumption and is evaluated using MAE, MSE, RMSE, and R2 score. The project also shows actual vs predicted consumption, feature importance, and prediction from user-entered values.

## Algorithm

XGBoost is a decision-tree-based gradient boosting algorithm. It builds trees sequentially so each new tree improves on the errors of the previous trees. Here it is used because electricity consumption is a continuous numerical target.

## Features Used

- Temperature
- Humidity
- Wind Speed
- Average Past Consumption
- Year
- Month
- Day
- Hour
- Minute
- Day of Week

Target:

```text
Electricity_Consumed
```

## Evaluation

- MAE
- MSE
- RMSE
- R2 Score

## Visualizations

- Actual vs predicted consumption.
- Feature-importance chart.

## Sample Prediction Input

```text
Temperature: 32.4
Humidity: 72
Wind Speed: 8.5
Average Past Consumption: 3.10
Year: 2024
Month: 8
Day: 10
Hour: 20
Minute: 30
Day of Week: 5
```

The model returns the expected electricity consumption in kWh.

## Technologies

- Python
- Pandas
- NumPy
- Scikit-learn
- XGBoost
- Matplotlib

## Project Files

```text
Electricity Consumption Prediction/
|-- smart_meter_data.csv
|-- Electricity_prediction.py
|-- README.md
|-- requirements.txt
`-- .gitignore
```

## How to Run

```bash
git clone <your-github-repository-link>
cd Electricity-Consumption-Prediction
pip install -r requirements.txt
python Electricity_prediction.py
```

## Key Points

- Smart meter consumption prediction.
- XGBoost regression.
- Timestamp feature extraction.
- Chronological train-test split.
- Regression model evaluation.
- Feature importance.
- Interactive prediction.

## Learning Outcomes

This project practices time-based feature engineering, preprocessing with Pandas, XGBoost regression, regression metrics, feature-importance analysis, Matplotlib visualization, and prediction using a trained model.
