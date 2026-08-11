import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from sklearn import tree
import matplotlib.pyplot as plt

data = pd.read_csv("data.csv")
print("Dataset :- ")
print(data.head())

le = LabelEncoder()

data["Employment_Status"] = le.fit_transform(data["Employment_Status"])
data["Existing_Loan"] = le.fit_transform(data["Existing_Loan"])
data["Loan_Approved"] = le.fit_transform(data["Loan_Approved"])
# data["Loan_Approved"] = le.fit_transform(data["Loan_Approved"])


X = data[["Age",
          "Monthly_Income",
          "Credit_Score",
          "Employment_Status",
          "Existing_Loan"]]

y = data["Loan_Approved"]


X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size = 0.20, random_state = 42
)

model = DecisionTreeClassifier(
    criterion = "gini",
    random_state = 42
)

model.fit(X_train, y_train)

y_pred = model.predict(X_test)

print("\nAccuracy : ", accuracy_score(y_test, y_pred))

print("\nConfusion Matrix : ")
print(confusion_matrix(y_test, y_pred))

print("\nClassification Report:")
print(classification_report(y_test, y_pred))


print("\n=====Enter Customer Details=====")
age = int(input("Enter Age : "))
income = float(input("Enter Monthly Income : "))
credit = int(input("Enter Credit Score : "))

print("\nEmployment Status")
print("1 = Employed")
print("0 = Unemployed")
employment = int(input("Enter Employment Status : "))

print("\nExisting Loan")
print("1 = Yes")
print("0 = No")
existing_loan = int(input("Enter Existing Loan Status : "))

new_applicant = pd.DataFrame({
    "Age": [age],
    "Monthly_Income": [income],
    "Credit_Score": [credit],
    "Employment_Status": [employment],
    "Existing_Loan": [existing_loan]
})

prediction = model.predict(new_applicant)


if prediction[0] == 1:
    print("\nResult : Loan Approved")
else:
    print("\nResult : Loan Rejected")


plt.figure(figsize=(10, 6)) 

tree.plot_tree(
    model,
    feature_names = X.columns,
    class_names = ["No", "Yes"],
    filled = True
)

plt.show()
