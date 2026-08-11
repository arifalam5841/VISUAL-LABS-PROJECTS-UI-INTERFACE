import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

data = {
    "Transaction_Amount": [100, 250, 5000, 150, 8000, 300, 12000, 450, 7000, 200],
    "Transaction_Time": [10, 14, 2, 18, 1, 12, 3, 20, 4, 16],
    "Location_Change": [0, 0, 1, 0, 1, 0, 1, 0, 1, 0],
    "International": [0, 0, 1, 0, 1, 0, 1, 0, 1, 0],
    "Fraud": [0, 0, 1, 0, 1, 0, 1, 0, 1, 0]
}

df = pd.DataFrame(data)

X = df.drop("Fraud", axis=1)
y = df["Fraud"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

model = LogisticRegression()
model.fit(X_train, y_train)

y_pred = model.predict(X_test)
print("Model Accuracy:", accuracy_score(y_test, y_pred))

print("\n====== Credit Card Fraud Detection ======\n")

amount = float(input("Enter Transaction Amount (₹): "))
time = int(input("Enter Transaction Time (0-23 hours): "))
location_change = int(input("Location Changed? (0 = No, 1 = Yes): "))
international = int(input("International Transaction? (0 = No, 1 = Yes): "))

user_data = pd.DataFrame({
    "Transaction_Amount": [amount],
    "Transaction_Time": [time],
    "Location_Change": [location_change],
    "International": [international]
})

prediction = model.predict(user_data)

print("\n========== Result ===========")

if prediction[0] == 1:
    print("Fraudulent Transaction Detected!")
else:
    print("Legitimate Transaction")