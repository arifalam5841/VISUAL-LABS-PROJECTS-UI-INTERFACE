import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

df = pd.read_csv("diabetes.csv")

X = df.drop("Outcome",axis=1)
y = df["Outcome"]

X_train,X_test,y_train,y_test=train_test_split(
X,y,test_size=0.2,random_state=42)

model = LogisticRegression(max_iter=1000)
# model = LogisticRegression(max_iter=1000)

model.fit(X_train,y_train)

prediction = model.predict(X_test)

print("Accuracy : ",accuracy_score(y_test,prediction))

patient = pd.DataFrame({
    "Pregnancies": [8],
    "Glucose": [195],
    "BloodPressure": [90],
    "SkinThickness": [40],
    "Insulin": [300],
    "BMI": [41.0],
    "DiabetesPedigreeFunction": [2.0],
    "Age": [60]
})

prediction = model.predict(patient)

if prediction[0] == 1:
    print("Prediction: Diabetic")
else:
    print("Prediction: Not Diabetic")