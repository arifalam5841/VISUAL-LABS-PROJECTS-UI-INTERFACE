# Customer Churn Prediction Using Logistic Regression

**Name:** Arif alam  
**Enrollment:** 24111590742

## Overview

This project predicts whether a customer is likely to leave a service or continue using it. The model uses Logistic Regression and learns from a sample dataset containing customer age, monthly charges, tenure, and support-call count.

After training, the program asks for customer details and predicts whether the customer is likely to churn or stay.

## Features

- Creates a sample customer dataset.
- Splits the dataset into training and testing sets.
- Trains a Logistic Regression model with Scikit-learn.
- Evaluates the model using accuracy.
- Accepts customer details from the user.
- Predicts customer churn status.
- Displays the result in a simple console format.

## Dataset Columns

| Feature | Description |
| ------- | ----------- |
| Age | Customer age |
| MonthlyCharges | Monthly subscription amount |
| Tenure | Number of months with the company |
| SupportCalls | Customer support-call count |
| Churn | Target value: 1 = Churn, 0 = Stay |

## Workflow

1. Create the customer dataset.
2. Split it into training and testing data.
3. Train a Logistic Regression classifier.
4. Evaluate model accuracy.
5. Read customer details from the user.
6. Predict whether the customer will churn or stay.
7. Print the prediction result.

## Technologies

- Python
- Pandas
- Scikit-learn
- Logistic Regression

## Sample Output

```text
Model Accuracy: 100.00%

Enter Customer Details

Age: 32
Monthly Charges: 950
Tenure (months): 12
Number of Support Calls: 3

Prediction Result
The customer is likely to CHURN.
```

## Installation and Run

```bash
git clone https://github.com/your-username/customer-churn-prediction.git
cd customer-churn-prediction
pip install -r requirements.txt
python customer_churn_prediction.py
```

## Project Files

```text
Customer-Churn-Prediction/
|-- customer_churn_prediction.py
|-- requirements.txt
|-- .gitignore
`-- README.md
```

## Future Scope

- Train with a real customer churn dataset.
- Increase the amount of training data.
- Add visualizations with Matplotlib and Seaborn.
- Build a Flask or Streamlit application.
- Improve accuracy using feature engineering and hyperparameter tuning.
