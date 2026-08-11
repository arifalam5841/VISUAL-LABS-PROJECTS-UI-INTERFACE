import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

data = {
    "Age": [22, 35, 45, 25, 52, 23, 40, 36, 28, 50],
    "MonthlyCharges": [500, 1200, 1500, 700, 2000, 650, 1800, 1300, 800, 2100],
    "Tenure": [2, 24, 36, 5, 48, 3, 40, 30, 8, 60],
    "SupportCalls": [5, 1, 0, 4, 0, 6, 1, 2, 3, 0],
    "Churn": [1, 0, 0, 1, 0, 1, 0, 0, 1, 0]
}

df = pd.DataFrame(data)
# df = pd.DataFramde(data)

X = df[["Age", "MonthlyCharges", "Tenure", "SupportCalls"]]
y = df["Churn"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

model = LogisticRegression()
model.fit(X_train, y_train)

y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print(f"\nModel Accuracy: {accuracy * 100:.2f}%")


print("\nEnter Customer Details")

age = int(input("Age: "))
monthly_charges = float(input("Monthly Charges: "))
tenure = int(input("Tenure (months): "))
support_calls = int(input("Number of Support Calls: "))

new_customer = pd.DataFrame({
    "Age": [age],
    "MonthlyCharges": [monthly_charges],
    "Tenure": [tenure],
    "SupportCalls": [support_calls]
})

prediction = model.predict(new_customer)

print("\nPrediction Result")
if prediction[0] == 1:
    print("The customer is likely to CHURN.")
else:
    print("The customer is likely to STAY.")
