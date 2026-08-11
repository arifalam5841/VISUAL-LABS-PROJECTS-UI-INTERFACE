from sklearn.linear_model import LogisticRegression
import numpy as np

# Training data (Marks)
X = np.array([[20], [30], [35], [40], [50], [60], [70], [80]])

# Target (0 = Fail, 1 = Pass)
y = np.array([0, 0, 0, 0, 1, 1, 1, 1])

# Create and train the model
model = LogisticRegression()
model.fit(X, y)

# Take input from the user
marks = int(input("Enter student's marks: "))

# Convert to 2D array
prediction = model.predict([[marks]])
# prediction = model.predict([[marks]])

# Display result
if prediction[0] == 1:
    print("Student will Pass")
else:
    print("Student will Fail")