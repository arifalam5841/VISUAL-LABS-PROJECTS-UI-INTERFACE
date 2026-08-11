# Loan Approval Prediction Using Logistic Regression

**Name:** Arif alam  
**Enrollment:** 24111590742

## Overview

This project predicts whether a loan application will be approved or rejected using Logistic Regression from Scikit-learn. It trains on a sample dataset containing applicant income and credit score, then predicts loan approval for user-entered values.

## Features

- Creates a sample applicant dataset.
- Trains a Logistic Regression model.
- Takes income and credit score as user input.
- Predicts loan approval or rejection.
- Displays the result in a simple format.

## Technologies

- Python
- Scikit-learn
- NumPy

## Project Files

```text
Loan-Approval-Prediction/
|-- loan_approval.py
|-- README.md
|-- requirements.txt
`-- .gitignore
```

## How to Run

```bash
git clone https://github.com/your-username/Loan-Approval-Prediction.git
cd Loan-Approval-Prediction
pip install -r requirements.txt
python loan_approval.py
```

## Example

Input:

```text
Enter Income: 55000
Enter Credit Score: 770
```

Output:

```text
Loan Approved
```

## Program Logic

The program creates a small training dataset with income, credit score, and loan approval status. A Logistic Regression model learns from this data. When a new applicant's income and credit score are entered, the model compares those values with the learned pattern and predicts whether the loan should be approved or rejected.

## Algorithm

- Logistic Regression

## Future Scope

- Train on a larger real-world dataset.
- Add features such as age, employment status, loan amount, and existing debt.
- Display prediction accuracy.
- Build a Tkinter GUI.
- Deploy with Flask or Streamlit.
