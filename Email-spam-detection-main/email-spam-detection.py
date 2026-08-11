import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

# Load dataset
df = pd.read_csv("completeSpamAssassin.csv")

# Remove unnecessary column
df = df.drop("Unnamed: 0", axis=1)

# Remove missing email bodies
df = df.dropna(subset=["Body"])

# Convert Body to string
df["Body"] = df["Body"].astype(str)

# Create Features and Target AFTER cleaning
X = df["Body"]
y = df["Label"]

# Convert text to numerical features
vectorizer = CountVectorizer(stop_words="english")

X = vectorizer.fit_transform(X)

# Split dataset
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42,
    stratify=y
)

# Train model
model = LogisticRegression(max_iter=1000)

model.fit(X_train, y_train)

prediction = model.predict(X_test)

print("Accuracy :", accuracy_score(y_test, prediction))

print(classification_report(y_test, prediction))

print(confusion_matrix(y_test, prediction))

# Test a new email
email = [
"""
Hi Danish,

Congratulations! You have won a free iPhone.
Click the link below to claim your prize.

http://free-prize.com

"""
]

email = vectorizer.transform(email)

result = model.predict(email)

print("Prediction : ", "Spam" if result[0] == 1 else "Not Spam")
