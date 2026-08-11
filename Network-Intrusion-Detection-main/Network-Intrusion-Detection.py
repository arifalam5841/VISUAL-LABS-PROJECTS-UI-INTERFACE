# NETWORK INTRUSION DETECTION
# USING RANDOM FOREST


# Import Libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import joblib

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report


# ==========================================
# PART 1: MODEL TRAINING
# ==========================================


# 1. Load datasets
normal_data = pd.read_csv(
    "Monday-WorkingHours.pcap_ISCX.csv"
)

intrusion_data = pd.read_csv(
    "Friday-WorkingHours-Afternoon-DDos.pcap_ISCX.csv"
)


# 2. Clean column names
normal_data.columns = normal_data.columns.str.strip()
intrusion_data.columns = intrusion_data.columns.str.strip()


# 3. Set labels
normal_data["Label"] = 0
intrusion_data["Label"] = 1


# 4. Combine datasets
data = pd.concat(
    [normal_data, intrusion_data],
    ignore_index=True
)


print("Total records : ", len(data))

print("Normal traffic : ", len(normal_data))
print("Intrusion traffic : ", len(intrusion_data))


# 5. Remove infinite values
data.replace(
    [np.inf, -np.inf],
    np.nan,
    inplace=True
)


# 6. Remove missing values
data.dropna(inplace=True)


# 7. Remove duplicate records
data.drop_duplicates(inplace=True)


print("\nRecords after cleaning : ", len(data))


# 8. Separate features and target
X = data.drop("Label", axis=1)

y = data["Label"]


# 9. Keep numerical columns
X = X.select_dtypes(
    include=["number"]
)


# 10. Split data
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size = 0.20,
    random_state = 42,
    stratify = y
)


print("\nTraining records : ", len(X_train))
print("Testing records : ", len(X_test))


# 11. Create Random Forest
model = RandomForestClassifier(
    n_estimators = 100,
    random_state = 42,
    n_jobs = -1
)


# 12. Train model
print("\nTraining Random Forest model...")

model.fit(X_train, y_train)

print("Training completed!")


# 13. Predictions
y_pred = model.predict(X_test)


# 14. Accuracy
accuracy = accuracy_score(y_test, y_pred)

print("\nModel Accuracy : ", accuracy)


# 15. Confusion Matrix
cm = confusion_matrix(y_test, y_pred)

print("\nConfusion Matrix : ")
print(cm)


# 16. Classification Report
print("\nClassification Report : ")

print(classification_report(
        y_test,
        y_pred,
        target_names=[
            "Normal",
            "Intrusion"
        ]
    )
)


# 17. Prediction Analysis
normal_predictions = sum(y_pred == 0)

intrusion_predictions = sum(y_pred == 1)

print("\n-------------------")
print("Prediction Analysis")
print("-------------------")

print("Normal traffic : ",normal_predictions)

print("Intrusions : ",intrusion_predictions)


# 18. Feature Importance
importance = model.feature_importances_

feature_importance = pd.DataFrame({
    "Feature": X.columns,
    "Importance": importance
})

feature_importance = feature_importance.sort_values(
    by="Importance",
    ascending=False
)


print("\nTop 10 Important Features : ")

print(feature_importance.head(10))


# 19. Feature Importance Chart
top_features = feature_importance.head(10)

plt.figure(figsize=(10, 6))

plt.barh(
    top_features["Feature"],
    top_features["Importance"]
)

plt.xlabel("Importance")
plt.ylabel("Feature")

plt.title(
    "Top 10 Important Network Features"
)

plt.gca().invert_yaxis()

plt.show()


# 20. Save Model
joblib.dump(model,"random_forest_model.pkl")

print("\nModel saved successfully!")


# ==========================================
# PART 2: INTRUSION PREDICTION
# ==========================================


# Use the test data for prediction
predictions = model.predict(X_test)


# Display first 20 predictions
print("\n------------------")
print("Prediction Results")
print("------------------")


for i, prediction in enumerate(
    predictions[:20],start = 1) :

    if prediction == 0:

        result = "Normal Traffic"

    else:

        result = "Intrusion Detected"

    print(
        f"Record {i}: {result}"
    )


# Count predictions
normal_count = sum(predictions == 0)

intrusion_count = sum(predictions == 1)

print("\n-------------------------")
print("Final Prediction Analysis")
print("-------------------------")

print("Normal Traffic : ",normal_count)

print("Intrusions Detected : ",intrusion_count)

print("Total Records : ",len(predictions))

# 21. Final Prediction Chart
labels = ["Normal Traffic", "Intrusion Detected"]

counts = [normal_count, intrusion_count]

plt.figure(figsize=(7, 5))

plt.bar(labels, counts, color=["green", "red"])

plt.xlabel("Traffic Type")
plt.ylabel("Number of Records")

plt.title("Network Traffic Prediction Analysis")

plt.show()
