# Credit Card Fraud Detection Using Logistic Regression

**Name:** Arif alam  
**Enrollment:** 24111590742

## Overview

This project classifies credit card transactions as either fraudulent or legitimate using Logistic Regression from Scikit-learn. A sample transaction dataset is used for training, and the user can enter transaction details in the console to get a prediction.

## Key Features

- Builds a sample dataset of transaction records.
- Splits data into training and testing sets with `train_test_split`.
- Trains a Logistic Regression classification model.
- Checks performance using accuracy score.
- Accepts transaction amount, time, location-change status, and international-transaction status.
- Predicts whether the transaction is fraudulent or legitimate.
- Shows the result in a clear console format.

## Technologies

- Python
- Pandas
- Scikit-learn

## Project Files

```text
Credit-Card-Fraud-Detection/
|-- credit_card_fraud_detection.py
|-- README.md
|-- requirements.txt
`-- .gitignore
```

## How to Run

```bash
git clone https://github.com/your-username/Credit-Card-Fraud-Detection.git
cd Credit-Card-Fraud-Detection
pip install -r requirements.txt
python credit_card_fraud_detection.py
```

## Example

```text
====== Credit Card Fraud Detection ======

Enter Transaction Amount (Rs): 9000
Enter Transaction Time (0-23 hours): 2
Location Changed? (0 = No, 1 = Yes): 1
International Transaction? (0 = No, 1 = Yes): 1

<!-- ========== Result ========== -->
Fraudulent Transaction Detected!
```

## Learning Outcomes

This project gives practice with binary classification, Logistic Regression, Pandas-based preprocessing, train-test splitting, Scikit-learn model training, accuracy evaluation, console input handling, and real-time prediction.
