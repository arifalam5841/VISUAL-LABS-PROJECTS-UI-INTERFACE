import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from sklearn import tree
import matplotlib.pyplot as plt


data = pd.read_csv("car_data.csv")


le = LabelEncoder()

data["Gender"] = le.fit_transform(data["Gender"])
data["Marital_Status"] = le.fit_transform(data["Marital_Status"])
data["Has_Driving_License"] = le.fit_transform(data["Has_Driving_License"])
data["Owns_House"] = le.fit_transform(data["Owns_House"])
data["Buy_Car"] = le.fit_transform(data["Buy_Car"])


X = data[[
    "Age",
    "Gender",
    "Annual_Income",
    "Marital_Status",
    "Has_Driving_License",
    "Owns_House"
]]


y = data["Buy_Car"]


X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size = 0.20, random_state = 42
)


model = DecisionTreeClassifier(
    criterion = "gini",
    random_state = 42
)


model.fit(X_train, y_train)


y_pred = model.predict(X_test)


print("Accuracy : ", accuracy_score(y_test, y_pred))


print("\nConfusion Matrix : ")
print(confusion_matrix(y_test, y_pred))


print("\nClassification Report : ")
print(classification_report(y_test, y_pred))



print("\n=====Enter Customer Details=====")

age = int(input("Enter Age : "))

print("\nGender")
print("0 = Female")
print("1 = Male") 
gender = int(input("Enter Gender (0/1) : "))

income = float(input("Enter Annual Income : "))

print("\nMarital Status")
print("0 = Married")
print("1 = Single")
marital_status = int(input("Enter Marital Status (0/1) : "))

print("\nDriving License")
print("0 = No")
print("1 = Yes")
driving_license = int(input("Has Driving License? (0/1) : "))

print("\nOwn House")
print("0 = No")
print("1 = Yes")
owns_house = int(input("Owns House? (0/1) : "))


new_customer = pd.DataFrame({
    "Age": [age],
    "Gender": [gender],
    "Annual_Income": [income],
    "Marital_Status": [marital_status],
    "Has_Driving_License": [driving_license],
    "Owns_House": [owns_house]
})

prediction = model.predict(new_customer)


if prediction[0] == 1:
    print("\nCustomer is likely to BUY a car.")
else:
    print("\nCustomer is NOT likely to BUY a car.")


plt.figure(figsize=(10, 6)) 

tree.plot_tree(
    model,
    feature_names=X.columns,
    class_names=["No","Yes"],
    filled=True
)

plt.show()