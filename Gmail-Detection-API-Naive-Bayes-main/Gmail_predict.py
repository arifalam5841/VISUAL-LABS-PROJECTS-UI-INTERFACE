# EMAIL SPAM DETECTION USING NAIVE BAYES
# PART 1 : MODEL TRAINING & EVALUATION


# Import Libraries
import joblib
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB

from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix
)

# Load Dataset
print("="*60)
print("LOADING DATASET")
print("="*60)

df = pd.read_csv("spam.csv", encoding="latin-1")

df = df[['v1','v2']]

df.columns = ['Label','Email']

print("\nFirst 5 Records\n")

print(df.head())

print("\nDataset Shape :", df.shape)


# Convert Labels
df["Label"] = df["Label"].map({
    "ham":0,
    "spam":1
})


# Features and Target
X = df["Email"]

y = df["Label"]


# Convert Text to Numbers
vectorizer = CountVectorizer(
    stop_words="english"
)

X = vectorizer.fit_transform(X)


# Split Dataset
X_train, X_test, y_train, y_test = train_test_split(

    X,
    y,

    test_size=0.20,

    random_state=42

)


# Train Model
print("\nTraining Naive Bayes Model...\n")

model = MultinomialNB()

model.fit(
    X_train,
    y_train
)


# Save Model
joblib.dump(
    model,
    "model.pkl"
)

joblib.dump(
    vectorizer,
    "vectorizer.pkl"
)

print("Model Saved Successfully!")


# Prediction
y_pred = model.predict(X_test)


# Evaluation
accuracy = accuracy_score(
    y_test,
    y_pred
)

print("\n")
print("="*60)
print("MODEL EVALUATION")
print("="*60)

print("\nAccuracy : {:.2f}%".format(
    accuracy*100
))

print("\nClassification Report\n")

print(
    classification_report(
        y_test,
        y_pred
    )
)

cm = confusion_matrix(
    y_test,
    y_pred
)

print("Confusion Matrix\n")

print(cm)

print("\n")
print("="*60)
print("MODEL TRAINING COMPLETED")
print("="*60)


# PART 2 : GMAIL API INTEGRATION
import os
import base64
import matplotlib.pyplot as plt

from email import message_from_bytes

from google.auth.transport.requests import Request
from google.oauth2.credentials import Credentials
from google_auth_oauthlib.flow import InstalledAppFlow
from googleapiclient.discovery import build


# Gmail Authentication
print("\n")
print("="*60)
print("CONNECTING TO GMAIL")
print("="*60)

SCOPES = ["https://www.googleapis.com/auth/gmail.readonly"]

creds = None

if os.path.exists("token.json"):
    creds = Credentials.from_authorized_user_file(
        "token.json",
        SCOPES
    )

if not creds or not creds.valid:

    if creds and creds.expired and creds.refresh_token:
        creds.refresh(Request())

    else:

        flow = InstalledAppFlow.from_client_secrets_file(
            "credentials.json",
            SCOPES
        )

        creds = flow.run_local_server(port=0)

    with open("token.json", "w") as token:
        token.write(creds.to_json())

service = build(
    "gmail",
    "v1",
    credentials=creds
)

print("Connected Successfully!")


# Load Saved Model
model = joblib.load("model.pkl")
vectorizer = joblib.load("vectorizer.pkl")


# Read Latest Emails
results = service.users().messages().list(
    userId="me",
    maxResults=10
).execute()

messages = results.get("messages", [])

if not messages:
    print("No Emails Found")
    exit()

spam_count = 0
ham_count = 0

print("\n")
print("="*60)
print("EMAIL PREDICTIONS")
print("="*60)


# Process Emails
for i, message in enumerate(messages, start=1):

    msg = service.users().messages().get(
        userId="me",
        id=message["id"],
        format="raw"
    ).execute()

    raw = base64.urlsafe_b64decode(msg["raw"])

    email_message = message_from_bytes(raw)

    subject = email_message["Subject"] or ""
    sender = email_message["From"] or "Unknown"

    body = ""

    try:

        if email_message.is_multipart():

            for part in email_message.walk():

                if part.get_content_type() == "text/plain":

                    body += part.get_payload(
                        decode=True
                    ).decode(errors="ignore")

        else:

            body = email_message.get_payload(
                decode=True
            ).decode(errors="ignore")

    except:
        body = ""

    email_text = subject + " " + body

    vector = vectorizer.transform([email_text])

    prediction = model.predict(vector)[0]

    probability = model.predict_proba(vector)[0]

    spam_probability = probability[1] * 100
    ham_probability = probability[0] * 100

    if prediction == 1:
        result = "SPAM"
        spam_count += 1
    else:
        result = "HAM"
        ham_count += 1

    print("\n" + "-"*60)

    print("Email :", i)

    print("\nFrom :")
    print(sender)

    print("\nSubject :")
    print(subject)

    print("\nPrediction :", result)

    print("Spam Probability : {:.2f}%".format(spam_probability))
    print("Ham Probability  : {:.2f}%".format(ham_probability))

print("\n")
print("="*60)
print("SUMMARY")
print("="*60)

print("Total Emails :", len(messages))
print("Spam Emails  :", spam_count)
print("Ham Emails   :", ham_count)

print("Model Accuracy : {:.2f}%".format(accuracy * 100))


# Bar Chart
labels = ["Spam", "Ham"]
counts = [spam_count, ham_count]

plt.figure(figsize=(6,5))

bars = plt.bar(labels, counts)

plt.title("Spam vs Ham Email Analysis")
plt.xlabel("Email Category")
plt.ylabel("Number of Emails")

for bar in bars:

    height = bar.get_height()

    plt.text(
        bar.get_x() + bar.get_width()/2,
        height,
        str(int(height)),
        ha="center",
        va="bottom"
    )

plt.grid(axis="y", linestyle="--", alpha=0.5)

plt.show()

print("\nProject Completed Successfully!")