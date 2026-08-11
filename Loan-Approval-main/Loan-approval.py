from sklearn.linear_model import LogisticRegression

# Training data
# Features: [Income, Credit Score]
X = [
    [30000, 650],
    [40000, 700],
    [50000, 750],
    [60000, 800],
    [25000, 600],
    [70000, 820]
]

# Target
# 0 = Loan Rejected
# 1 = Loan Approved
y = [0, 0, 1, 1, 0, 1]
# y = [0, 0, 1, 1, 0, 1]

# Create the model
model = LogisticRegression()

# Train the model
model.fit(X, y)

# Predict for a new applicant
income = int(input("Enter Income: "))
credit_score = int(input("Enter Credit Score: "))

prediction = model.predict([[income, credit_score]])

if prediction[0] == 1:
    print("Loan Approved")
else:
    print("Loan Rejected")