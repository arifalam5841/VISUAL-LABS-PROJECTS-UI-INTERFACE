# ELECTRICITY CONSUMPTION PREDICTION
# USING XGBOOST

# =============================================
# PART 1: DATA PREPROCESSING AND MODEL TRAINING
# =============================================

# Import Libraries
import pandas as pd
import numpy as np

from xgboost import XGBRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


# 1. LOAD DATASET
df = pd.read_csv("smart_meter_data.csv")


# 2. DISPLAY DATASET INFORMATION
print("First 5 Records : ")
print(df.head())

print("\nDataset Shape : ")
print(df.shape)

print("\nDataset Columns : ")
print(df.columns.tolist())

print("\nMissing Values : ")
print(df.isnull().sum())


# 3. CONVERT TIMESTAMP
df["Timestamp"] = pd.to_datetime(df["Timestamp"])


# 4. FEATURE ENGINEERING

# Extract useful information from Timestamp

df["Year"] = df["Timestamp"].dt.year
df["Month"] = df["Timestamp"].dt.month
df["Day"] = df["Timestamp"].dt.day
df["Hour"] = df["Timestamp"].dt.hour
df["Minute"] = df["Timestamp"].dt.minute
df["DayOfWeek"] = df["Timestamp"].dt.dayofweek


# 5. SELECT FEATURES AND TARGET

# Features used for prediction

features = [
    "Temperature",
    "Humidity",
    "Wind_Speed",
    "Avg_Past_Consumption",
    "Year",
    "Month",
    "Day",
    "Hour",
    "Minute",
    "DayOfWeek"
]

X = df[features]

# Target variable
y = df["Electricity_Consumed"]


# 6. CHRONOLOGICAL TRAIN-TEST SPLIT

# Since this is time-series data,
# we do not randomly shuffle the records.

split_index = int(len(df) * 0.8)

X_train = X.iloc[:split_index]
X_test = X.iloc[split_index:]

y_train = y.iloc[:split_index]
y_test = y.iloc[split_index:]


print("\nTraining Records:", len(X_train))
print("Testing Records:", len(X_test))


# 7. CREATE XGBOOST MODEL
model = XGBRegressor(
    n_estimators=200,
    learning_rate=0.05,
    max_depth=6,
    subsample=0.8,
    colsample_bytree=0.8,
    random_state=42
)


# 8. TRAIN THE MODEL
print("\nTraining XGBoost Model...")

model.fit(X_train, y_train)

print("Model Training Completed!")


# 9. MAKE PREDICTIONS
y_pred = model.predict(X_test)


# 10. MODEL EVALUATION
mae = mean_absolute_error(y_test, y_pred)

mse = mean_squared_error(y_test, y_pred)

rmse = np.sqrt(mse)

r2 = r2_score(y_test, y_pred)


print("\n========== MODEL PERFORMANCE ==========")

print("Mean Absolute Error (MAE) : ", round(mae, 4))

print("Mean Squared Error (MSE) : ", round(mse, 4))

print("Root Mean Squared Error (RMSE) : ", round(rmse, 4))

print("R2 Score : ", round(r2, 4))

print("=======================================")

# ===============================================
# PART 2: PREDICTION ANALYSIS AND USER PREDICTION
# ===============================================

import matplotlib.pyplot as plt


# 11. DISPLAY ACTUAL VS PREDICTED
results = pd.DataFrame({
    "Actual Consumption": y_test.values,
    "Predicted Consumption": y_pred
})

print("\n========== ACTUAL VS PREDICTED ==========")
print(results.head(10))


# 12. ACTUAL VS PREDICTED CHART
plt.figure(figsize = (10, 5))

plt.plot(y_test.values[:100],label = "Actual Consumption")

plt.plot(y_pred[:100],label = "Predicted Consumption")

plt.title("Actual vs Predicted Electricity Consumption")

plt.xlabel("Test Records")

plt.ylabel("Electricity Consumed (kWh)")

plt.legend()

plt.tight_layout()

plt.show()


# 13. FEATURE IMPORTANCE
importance = model.feature_importances_

feature_importance = pd.DataFrame({
    "Feature": features,
    "Importance": importance
})

feature_importance = feature_importance.sort_values(
    by="Importance",
    ascending=False
)


print("\n========== FEATURE IMPORTANCE ==========")
print(feature_importance)


# 14. FEATURE IMPORTANCE CHART
plt.figure(figsize = (10, 6))

plt.bar(
    feature_importance["Feature"],
    feature_importance["Importance"]
)

plt.title("XGBoost Feature Importance")

plt.xlabel("Features")

plt.ylabel("Importance")

plt.xticks(rotation=45)

plt.tight_layout()

plt.show()


# 15. USER INPUT PREDICTION
print("\n======================================")
print(" ELECTRICITY CONSUMPTION PREDICTION")
print("======================================")


temperature = float(input("Enter Temperature : "))

humidity = float(input("Enter Humidity : "))

wind_speed = float(input("Enter Wind Speed : "))

avg_past_consumption = float(input("Enter Average Past Consumption : "))

year = int(input("Enter Year : "))

month = int(input("Enter Month (1-12) : "))

day = int(input("Enter Day (1-31) : "))

hour = int(input("Enter Hour (0-23) : "))

minute = int(input("Enter Minute (0 or 30) : "))

day_of_week = int(input("Enter Day of Week (0=Monday, 6=Sunday) : "))


# 16. CREATE INPUT DATAFRAME
new_data = pd.DataFrame({
    "Temperature": [temperature],
    "Humidity": [humidity],
    "Wind_Speed": [wind_speed],
    "Avg_Past_Consumption": [avg_past_consumption],
    "Year": [year],
    "Month": [month],
    "Day": [day],
    "Hour": [hour],
    "Minute": [minute],
    "DayOfWeek": [day_of_week]
})


# 17. MAKE FINAL PREDICTION
prediction = model.predict(new_data)


print("\n======================================")
print("Predicted Electricity Consumption : ")
print(round(prediction[0], 4), "kWh")
print("======================================")
