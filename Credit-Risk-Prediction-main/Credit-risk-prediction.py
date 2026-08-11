import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

data = {
    "Income": [30000, 45000, 50000, 60000, 25000, 70000, 80000, 35000, 90000, 40000],
    "CreditScore": [550, 620, 700, 750, 500, 780, 820, 580, 850, 650],
    "LoanAmount": [200000, 150000, 120000, 100000, 250000, 90000, 80000, 180000, 70000, 140000],
    "Risk": [1, 1, 0, 0, 1, 0, 0, 1, 0, 0]  # 0 = Low Risk, 1 = High Risk
}

df = pd.DataFrame(data)

X = df[["Income", "CreditScore", "LoanAmount"]]
y = df["Risk"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size = 0.2, random_state = 42
)

model = LogisticRegression(max_iter = 1000)
model.fit(X_train, y_train)

y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)

print(f"Model Accuracy : {accuracy * 100:.2f}%")

print("\nEnter Applicant Details")
income = float(input("Income: "))
credit_score = int(input("Credit Score: "))
loan_amount = float(input("Loan Amount: "))

new_data = pd.DataFrame({
    "Income": [income],
    "CreditScore": [credit_score],
    "LoanAmount": [loan_amount]
})

prediction = model.predict(new_data)

print("\nPrediction Result :- ")
if prediction[0] == 0:
    print("Low Credit Risk (Loan is likely to be approved)")
else:
    print("High Credit Risk (Loan approval may be risky!)")