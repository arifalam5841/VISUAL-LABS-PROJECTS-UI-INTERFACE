# Credit Risk Prediction Using Logistic Regression

**Name:** Arif alam  
**Enrollment:** 24111590742

## Overview

This project predicts whether a loan applicant belongs to the low-risk or high-risk credit category. It uses Logistic Regression from Scikit-learn and trains on a sample dataset containing applicant income, credit score, and loan amount.

After training, the program asks the user for applicant details and predicts the credit-risk level.

## Features

- Creates a sample applicant dataset.
- Processes the dataset with Pandas.
- Splits records into training and testing sets.
- Trains a Logistic Regression model.
- Measures prediction accuracy.
- Takes income, credit score, and loan amount from the user.
- Classifies the applicant as low credit risk or high credit risk.

## Program Flow

The program prepares a dataset of loan applicants with credit-risk labels. It separates input features from the target, splits the data, trains a Logistic Regression model, evaluates accuracy, and then uses the trained model to predict credit risk for new user-provided values.

## Technologies

- Python
- Pandas
- Scikit-learn
- Logistic Regression
- VS Code

## Project Files

```text
<!-- Credit-Risk-Prediction/ -->
|-- credit-risk-prediction.py
|-- README.md
|-- requirements.txt
`-- .gitignore
```

## Installation and Execution

```bash
git clone <repository-link>
cd Credit-Risk-Prediction
pip install -r requirements.txt
python credit-risk-prediction.py
```

## Sample Input

```text
Income: 55000
Credit Score: 720
Loan Amount: 120000
```

## Sample Output

```text
Prediction Result:
Low Credit Risk (Loan is likely to be approved)
```

## Concepts Used

- Supervised learning
- Logistic Regression
- Binary classification
- Train-test split
- Model evaluation
- User input handling

## Acknowledgement

Special thanks to mentor Aiman Kazi for guidance and encouragement during the machine learning learning journey.
