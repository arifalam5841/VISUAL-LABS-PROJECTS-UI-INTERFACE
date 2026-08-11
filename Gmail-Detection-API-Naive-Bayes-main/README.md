# Email Spam Detection With Naive Bayes and Gmail API

**Name:** Arif alam  
**Enrollment:** 24111590742

## Overview

This project detects whether emails are spam or ham using the Multinomial Naive Bayes classification algorithm. It first trains a spam detection model on a labeled email dataset, then connects to Gmail through the Gmail API to classify recent inbox messages.

<!-- ## Project Summary -->

Email spam is a common security and communication problem. This project uses text classification to identify spam from email content. After training, the application authenticates with Gmail, reads recent emails, predicts each message as spam or ham, shows prediction probabilities, and summarizes the results with a chart.

## How It Works

The dataset is loaded and labels are converted into numerical values. Email text is transformed with CountVectorizer into a Bag-of-Words representation. A Multinomial Naive Bayes classifier is trained and evaluated with accuracy, classification report, and confusion matrix. The trained model and vectorizer are saved with Joblib.

In the Gmail phase, the program uses OAuth 2.0 authentication, retrieves recent messages, extracts subject and body text, vectorizes the text, predicts spam or ham, displays probability scores, and generates a spam-versus-ham bar chart.

## Technologies

- Python
- Pandas
- Scikit-learn
- CountVectorizer
- Multinomial Naive Bayes
- Joblib
- Gmail API
- Google OAuth 2.0
- Matplotlib

## Features

- Trains a Naive Bayes spam detection model.
- Converts email text into numerical features.
- Saves the trained model and vectorizer.
- Evaluates model performance.
- Authenticates Gmail securely with OAuth 2.0.
- Reads latest Gmail messages.
- Predicts spam or ham for each email.
- Displays spam and ham probabilities.
- Summarizes classified email counts.
- Visualizes results using a bar chart.

## Project Files

```text
Gmail-Detection-API-Naive-Bayes/
|-- Gmail_predict.py
|-- spam.csv
|-- README.md
|-- requirements.txt
`-- .gitignore
```

## Model Evaluation

- Accuracy Score
- Classification Report
- Confusion Matrix

These metrics are used before applying the model to Gmail messages.

## Output

The application displays dataset preview, dataset information, model accuracy, classification report, confusion matrix, Gmail authentication status, latest email details, sender, subject, prediction, spam probability, ham probability, total spam count, total ham count, and a spam-vs-ham chart.

## Installation

```bash
git clone https://github.com/your-username/Gmail-Detection-API-Naive-Bayes.git
cd Gmail-Detection-API-Naive-Bayes
pip install -r requirements.txt
```

## Run

```bash
python Gmail_predict.py
```

## Gmail API Setup

1. Create a project in Google Cloud Console.
2. Enable the Gmail API.
3. Create OAuth client credentials.
4. Download the credentials file.
5. Rename it to `credentials.json`.
6. Place it in the project folder.
7. Run the project.
8. Sign in with Gmail when prompted.
9. A `token.json` file will be generated for later use.

## Concepts Covered

- NLP
- Text vectorization
- Bag of Words
- CountVectorizer
- Naive Bayes classification
- Model training and evaluation
- Confusion matrix
- Classification report
- Gmail API integration
- OAuth authentication
- Data visualization

## Future Scope

- Use TF-IDF vectorization.
- Compare multiple machine learning algorithms.
- Try deep learning for spam detection.
- Add a Flask or Django web interface.
- Monitor emails in real time.
- Analyze attachments.
- Highlight spam keywords.
- Compare model performance.
