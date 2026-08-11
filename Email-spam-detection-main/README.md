# Email Spam Detection Using Logistic Regression

**Name:** Arif alam  
**Enrollment:** 24111590742

## Overview

This project detects whether an email is spam or not spam using Logistic Regression. The model is trained on the SpamAssassin email dataset. Email text is cleaned and converted into numerical form with CountVectorizer so the classifier can process it.

## Features

<!-- - Loads and cleans the SpamAssassin dataset. -->
- Removes unnecessary columns.
- Handles missing email records.
- Converts email text to numerical vectors with CountVectorizer.
- Splits data into training and testing sets.
- Trains a Logistic Regression spam classifier.
- Evaluates the model with accuracy score, classification report, and confusion matrix.
- Accepts a new email from the user and predicts spam or not spam.

## Technologies

- Python
- Pandas
- Scikit-learn
- CountVectorizer
- Logistic Regression

## Project Files

```text
Email-Spam-Detection/
|-- email_spam_detection.py
|-- completeSpamAssassin.csv
|-- requirements.txt
|-- .gitignore
`-- README.md
```

## Installation and Run

```bash
git clone https://github.com/your-username/Email-Spam-Detection.git
cd Email-Spam-Detection
pip install -r requirements.txt
python email_spam_detection.py
```

## Program Flow

1. Load the SpamAssassin email dataset.
2. Remove extra columns and missing values.
3. Convert email text into vectors with CountVectorizer.
4. Split the dataset into train and test sets.
5. Train the Logistic Regression model.
6. Evaluate the trained classifier.
7. Ask the user for email content.
8. Classify the email as spam or not spam.

## Evaluation

- Accuracy Score
- Classification Report
- Confusion Matrix

## Example

Input:

```text
Congratulations!

You have won a free iPhone.

Click here to claim your prize.
```

Output:

```text
Prediction: Spam
```

Input:

```text
Hi John,

Please submit the project report before tomorrow's meeting.

Regards,
Danish
```

Output:

```text
Prediction: Not Spam
```

## Learning Outcomes

- Logistic Regression for binary classification.
- Natural Language Processing basics.
- Text preprocessing.
- Feature extraction with CountVectorizer.
- Machine learning workflow.
- Model evaluation.

## License

This project is intended for educational and learning purposes.
