import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split

# Dataset
data = {
    "Hours" : [2,3,4,5,6,7,8,9,10],
    "Marks" : [35,45,50,60,65,72,80,88,95]
}

df = pd.DataFrame(data)

X = df[["Hours"]]
y = df[["Marks"]]
# y = df[["Marks"]]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size = 0.2, random_state = 42
)

model = LinearRegression()
model.fit(X_train, y_train)

new_data = pd.DataFrame({"Hours": [7.5]})
prediction = model.predict(new_data)

print("Predicted Marks:", prediction[0][0])