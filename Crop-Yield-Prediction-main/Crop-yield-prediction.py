# CROP YIELD PREDICTION 
# USING RANDOM FOREST

# 1. Import Libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    r2_score
)


# 2. Load Dataset
df = pd.read_csv("crop_yield.csv")

# 3. Display Dataset Information
print("=" * 60)
print("CROP YIELD PREDICTION USING RANDOM FOREST")
print("=" * 60)

print("\nFirst 5 records : ")
print(df.head())

print("\nDataset Shape : ")
print(df.shape)

print("\nColumn Names : ")
print(df.columns.tolist())


# 4. Check Missing Values
print("\nMissing Values : ")
print(df.isnull().sum())
# print(df.isnull().sum())


# 5. Remove Missing Values
df = df.dropna()

print("\nDataset Shape After Removing Missing Values : ")
print(df.shape)


# 6. Define Features and Target
X = df[
    [
        "Crop",
        "Crop_Year",
        "Season",
        "State",
        "Area",
        "Annual_Rainfall",
        "Fertilizer",
        "Pesticide"
    ]
]

y = df["Yield"]


# 7. Define Feature Types
categorical_features = [
    "Crop",
    "Season",
    "State"
]

numerical_features = [
    "Crop_Year",
    "Area",
    "Annual_Rainfall",
    "Fertilizer",
    "Pesticide"
]


# 8. Preprocessing
preprocessor = ColumnTransformer(
    transformers = [
        (
            "categorical",
            OneHotEncoder(handle_unknown = "ignore"),
            categorical_features
        ),
        (
            "numerical",
            "passthrough",
            numerical_features
        )
    ]
)


# 9. Create Random Forest
random_forest = RandomForestRegressor(
    n_estimators = 200,
    random_state = 42,
    n_jobs = -1
)


# 10. Create Pipeline
model = Pipeline(
    steps=[
        ("preprocessor", preprocessor),
        ("random_forest", random_forest)
    ]
)


# 11. Train-Test Split
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size = 0.20,
    random_state = 42
)

print("\nTraining Records : ", len(X_train))
print("Testing Records : ", len(X_test))


# 12. Train Model
print("\nTraining Random Forest Model...")

model.fit(X_train, y_train)

print("Model training completed.")


# 13. Make Predictions
y_pred = model.predict(X_test)


# 14. Model Evaluation
mae = mean_absolute_error(y_test, y_pred)

mse = mean_squared_error(y_test, y_pred)

rmse = np.sqrt(mse)

r2 = r2_score(y_test, y_pred)


print("\n" + "=" * 60)
print("MODEL EVALUATION")
print("=" * 60)

print(f"Mean Absolute Error (MAE) : {mae:.4f}")
print(f"Mean Squared Error (MSE) : {mse:.4f}")
print(f"Root Mean Squared Error (RMSE) : {rmse:.4f}")
print(f"R² Score : {r2:.4f}")


# 15. Sample Predictions
results = pd.DataFrame({
    "Actual Yield": y_test.values,
    "Predicted Yield": y_pred
})

print("\nSample Predictions : ")

print(results.head(10).to_string(index=False))


# 16. Actual vs Predicted Chart
plt.figure(figsize=(8, 6))

plt.scatter(
    y_test,
    y_pred,
    alpha = 0.5
)

min_value = min(
    y_test.min(),
    y_pred.min()
)

max_value = max(
    y_test.max(),
    y_pred.max()
)

plt.plot(
    [min_value, max_value],
    [min_value, max_value],
    linestyle="--"
)

plt.xlabel("Actual Yield")
plt.ylabel("Predicted Yield")

plt.title("Actual vs Predicted Crop Yield")

plt.tight_layout()

plt.show()


# 17. Feature Importance
# Get fitted preprocessing step

fitted_preprocessor = model.named_steps[
    "preprocessor"
]

# Get encoded feature names

feature_names = (
    fitted_preprocessor
    .get_feature_names_out()
)

# Get Random Forest feature importance

importance = (
    model
    .named_steps["random_forest"]
    .feature_importances_
)


feature_importance = pd.DataFrame({
    "Feature": feature_names,
    "Importance": importance
})


feature_importance = (
    feature_importance
    .sort_values(
        by = "Importance",
        ascending = False
    )
)


# 18. Display Feature Importance
print("\n" + "=" * 60)
print("FEATURE IMPORTANCE")
print("=" * 60)

print(
    feature_importance
    .head(15)
    .to_string(index=False)
)


# 19. Feature Importance Chart
top_features = feature_importance.head(10)

plt.figure(figsize=(10, 7))

plt.barh(
    top_features["Feature"][::-1],
    top_features["Importance"][::-1]
)

plt.xlabel("Importance")

plt.ylabel("Feature")

plt.title("Top 10 Features Affecting Crop Yield")

plt.tight_layout()

plt.show()


# 20. User Input Prediction
print("\n" + "=" * 60)
print("CROP YIELD PREDICTION")
print("=" * 60)

print("\nEnter agricultural information:")

try:

    crop = input(
        "Crop: "
    )

    crop_year = int(
        input("Crop Year: ")
    )

    season = input(
        "Season: "
    )

    state = input(
        "State: "
    )

    area = float(
        input("Area: ")
    )

    rainfall = float(
        input("Annual Rainfall: ")
    )

    fertilizer = float(
        input("Fertilizer: ")
    )

    pesticide = float(
        input("Pesticide: ")
    )


    # Create DataFrame for user input
    user_data = pd.DataFrame({
        "Crop": [crop],
        "Crop_Year": [crop_year],
        "Season": [season],
        "State": [state],
        "Area": [area],
        "Annual_Rainfall": [rainfall],
        "Fertilizer": [fertilizer],
        "Pesticide": [pesticide]
    })


    # Make prediction
    prediction = model.predict(user_data)

    print("\n" + "=" * 60)
    print("PREDICTION RESULT")
    print("=" * 60)

    print(f"Predicted Crop Yield : {prediction[0]:.4f}")


except ValueError:

    print("\nInvalid numerical input.")

    print("Please enter valid numbers.")

print("\nProgram completed.")