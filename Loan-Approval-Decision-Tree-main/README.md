# Loan Approval Prediction Using Decision Tree

**Name:** Arif alam  
**Enrollment:** 24111590742

## Overview

This project predicts whether a loan application should be approved or rejected using a Decision Tree classification model. It analyzes applicant details such as age, monthly income, credit score, employment status, and existing loan status.

The project also evaluates classification performance and visualizes the trained decision tree so the decision process is easier to understand.

## Features

- Predicts loan approval from customer details.
- Uses Decision Tree Classification.
- Encodes categorical data with Label Encoding.
- Splits data into training and testing sets.
- Evaluates the model with accuracy score, confusion matrix, and classification report.
- Accepts real-time user input.
- Shows the trained Decision Tree visualization.

## Technologies

- Python
- Pandas
- Scikit-learn
- Matplotlib

## Dataset

The dataset file is `data.csv`.

| Feature | Description |
| ------- | ----------- |
| Age | Applicant age |
| Monthly_Income | Applicant monthly income |
| Credit_Score | Applicant credit score |
| Employment_Status | Employed or Unemployed |
| Existing_Loan | Existing loan status: Yes or No |
| Loan_Approved | Target value: Approved or Rejected |

## Workflow

1. Load the dataset with Pandas.
2. Convert categorical values into numeric values.
3. Select features and target column.
4. Split the data into train and test sets.
5. Train the Decision Tree Classifier with the Gini criterion.
6. Evaluate the model.
7. Read applicant details from the user.
8. Predict approval or rejection.
9. Display the tree visualization.

## Installation and Run

```bash
git clone https://github.com/your-username/Loan-Approval-Decision-Tree.git
cd Loan-Approval-Decision-Tree
pip install -r requirements.txt
python Loan_Approval_Decision_Tree.py
```

## Sample Input

```text
Age: 28
Monthly Income: 50000
Credit Score: 750
Employment Status: 1
Existing Loan: 0
```

## Sample Output

```text
Result : Loan Approved
```

## Evaluation Metrics

- Accuracy Score
- Confusion Matrix
- Classification Report

## Learning Outcomes

- Decision Tree Classification
- Gini Index
- Data preprocessing
- Label Encoding
- Train-test split
- Model training and evaluation
- Classification metrics
- User-input prediction
- Decision Tree visualization
