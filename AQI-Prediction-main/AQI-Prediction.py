# AIR QUALITY INDEX (AQI) PREDICTION
# USING XGBOOST REGRESSION


# ============================================================
# Part 1 : MODEL TRAINING AND EVALUATION
# ============================================================


# 1. IMPORT LIBRARIES
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns


from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error, 
    r2_score
)

from xgboost import XGBRegressor
import joblib

# 2. LOAD DATASET
df = pd.read_csv("air_quality_data.csv")

# 3. DISPLAY BASIC INFORMATION
print("\nFirst 5 Records : ")
print(df.head())

print("\nDataset Shape : ")
print(df.shape)

print("\nColumn Names : ")
print(df.columns.tolist())

print("\nDataset Information : ")
print(df.info())

print("\nMissing Values : ")
print(df.isnull().sum())

print("\nDuplicate Records : ")
print(df.duplicated().sum())


# 4. STATISTICAL INFORMATION
print("\nStatistical Summary : ")
print(df.describe())


# 5. AQI DISTRIBUTION
plt.figure(figsize = (10, 6))

sns.histplot(df["AQI"], bins = 40, kde = True)

plt.title("Distribution of AQI")
plt.xlabel("AQI")
plt.ylabel("Frequency")

plt.show()


# 6. CORRELATION HEATMAP
numeric_columns = [
    "PM2.5",
    "PM10",
    "NO",
    "NO2",
    "NOx",
    "NH3",
    "CO",
    "SO2",
    "O3",
    "Benzene",
    "Toluene",
    "AQI"
]

plt.figure(figsize = (12, 8))

sns.heatmap(
    df[numeric_columns].corr(),
    annot = True,
    cmap = "coolwarm",
    fmt=".2f"
)

plt.title("Correlation Between Pollutants and AQI")

plt.show()


# 7. SELECT FEATURES AND TARGET
features = [
    "PM2.5",
    "PM10",
    "NO",
    "NO2",
    "NOx",
    "NH3",
    "CO",
    "SO2",
    "O3",
    "Benzene",
    "Toluene"
]

X = df[features]

y = df["AQI"]


print("\nSelected Features : ")
print(features)

print("\nTarget Variable : ")
print("AQI")


# 8. SPLIT DATASET
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size = 0.20,
    random_state = 42
)


print("\nTraining Records : ", len(X_train))
print("Testing Records : ", len(X_test))


# 9. CREATE XGBOOST MODEL
model = XGBRegressor(
    n_estimators = 300,
    learning_rate = 0.05,
    max_depth = 6,
    subsample = 0.8,
    colsample_bytree = 0.8,
    random_state = 42,
    objective = "reg:squarederror"
)


# 10. TRAIN MODEL
print("\nTraining XGBoost Model...")

model.fit(X_train, y_train)

print("Model Training Completed!")


# 11. MAKE PREDICTIONS
y_pred = model.predict(X_test)


# 12. MODEL EVALUATION
mae = mean_absolute_error(y_test, y_pred)

mse = mean_squared_error(y_test, y_pred)

rmse = np.sqrt(mse)

r2 = r2_score(y_test, y_pred)


print("\n====================================")
print("       MODEL EVALUATION")
print("====================================")

print(f"Mean Absolute Error (MAE) : {mae:.2f}")
print(f"Mean Squared Error (MSE) : {mse:.2f}")
print(f"Root Mean Squared Error (RMSE) : {rmse:.2f}")
print(f"R² Score : {r2:.4f}")


# 13. FEATURE IMPORTANCE
feature_importance = pd.DataFrame({
    "Feature": features,
    "Importance": model.feature_importances_
})

feature_importance = feature_importance.sort_values(
    by = "Importance",
    ascending = False
)

print("\nFeature Importance : ")
print(feature_importance)


# 14. FEATURE IMPORTANCE CHART
plt.figure(figsize = (10, 6))

sns.barplot(data = feature_importance, x = "Importance",
    y = "Feature"
)

plt.title("XGBoost Feature Importance")
plt.xlabel("Importance")
plt.ylabel("Feature")

plt.show()


# 15. ACTUAL VS PREDICTED AQI
plt.figure(figsize = (10, 6))

plt.scatter(y_test, y_pred, alpha = 0.5)

plt.xlabel("Actual AQI")
plt.ylabel("Predicted AQI")

plt.title("Actual vs Predicted AQI")

# Reference line
min_value = min(y_test.min(), y_pred.min())
max_value = max(y_test.max(), y_pred.max())

plt.plot(
    [min_value, max_value],
    [min_value, max_value],
    linestyle="--"
)

plt.show()


# 16. SAVE TRAINED MODEL
joblib.dump(model, "aqi_xgboost_model.pkl")

print("\nModel saved successfully as:")
print("aqi_xgboost_model.pkl")

# ============================================================
# PART 2 : USER INPUT AND AQI PREDICTION
# ============================================================


# 17. LOAD TRAINED MODEL
loaded_model = joblib.load("aqi_xgboost_model.pkl")


# 18. FUNCTION TO DETERMINE AQI CATEGORY
def get_aqi_category(aqi):

    if aqi <= 50:
        return "Good"

    elif aqi <= 100:
        return "Satisfactory"

    elif aqi <= 200:
        return "Moderate"

    elif aqi <= 300:
        return "Poor"

    elif aqi <= 400:
        return "Very Poor"

    else:
        return "Severe"


# 19. TAKE INPUT FROM USER
print("\n====================================")
print("       AQI PREDICTION SYSTEM")
print("====================================")

print("\nEnter the following pollutant values:")

pm25 = float(input("PM2.5 : "))
pm10 = float(input("PM10 : "))
no = float(input("NO : "))
no2 = float(input("NO2 : "))
nox = float(input("NOx : "))
nh3 = float(input("NH3 : "))
co = float(input("CO : "))
so2 = float(input("SO2 : "))
o3 = float(input("O3 : "))
benzene = float(input("Benzene : "))
toluene = float(input("Toluene : "))


# 20. CREATE INPUT DATAFRAME
user_data = pd.DataFrame({

    "PM2.5": [pm25],
    "PM10": [pm10],
    "NO": [no],
    "NO2": [no2],
    "NOx": [nox],
    "NH3": [nh3],
    "CO": [co],
    "SO2": [so2],
    "O3": [o3],
    "Benzene": [benzene],
    "Toluene": [toluene]

})


# 21. PREDICT AQI
predicted_aqi = loaded_model.predict(user_data)[0]


# Prevent negative AQI values
predicted_aqi = max(0, predicted_aqi)


# 22. GET AQI CATEGORY
aqi_category = get_aqi_category(predicted_aqi)


# 23. DISPLAY RESULT
print("\n====================================")
print("          AQI PREDICTION")
print("====================================")

print(f"Predicted AQI : {predicted_aqi:.2f}")
print(f"AQI Category : {aqi_category}")

print("====================================")