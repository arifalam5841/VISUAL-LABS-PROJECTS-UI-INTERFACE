import pandas as pd
from sklearn.linear_model import LinearRegression

data = {
    "Area" : [600,800,1000,1200,1500,1800,2000,2200,2500],
    "Price" : [30,40,50,60,75,90,100,110,125]
}

df = pd.DataFrame(data)

X = df[["Area"]]
y = df[["Price"]]

model = LinearRegression()
model.fit(X, y)

new_data = pd.DataFrame({"Area": [1700]})
# new_data = pd.DataFrame({"Area": [1700]})

price = model.predict(new_data)

print("Predicted House Price :", price[0][0], "Lakhs")
