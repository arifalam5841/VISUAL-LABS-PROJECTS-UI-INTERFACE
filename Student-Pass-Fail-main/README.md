# Student Pass/Fail Prediction Using Logistic Regression

**Name:** Arif alam  
**Enrollment:** 24111590742

## Overview

This project predicts whether a student will pass or fail based on marks. It uses Logistic Regression from Scikit-learn and trains on a small sample dataset containing marks and pass/fail outcomes.

## Features

- Creates a dataset of student marks and results.
- Trains a Logistic Regression classifier.
- Accepts marks from the user.
- Predicts pass or fail.
- Demonstrates binary classification in a simple way.

## Technologies

- Python
- NumPy
- Scikit-learn

## Project Files

```text
Student-Pass-Fail-Prediction/
|-- student_pass_fail.py
|-- README.md
|-- requirements.txt
`-- .gitignore
```

## How to Run

```bash
git clone https://github.com/your-username/student-pass-fail-prediction.git
cd student-pass-fail-prediction
pip install -r requirements.txt
python student_pass_fail.py
```

Enter marks when the program asks:

```text
Enter student's marks: 65
Student will Pass
```

## Program Logic

The program trains a Logistic Regression model with marks as the input and pass/fail result as the target. When the user enters a mark, the model uses the learned relationship to predict whether the student is likely to pass or fail.

## Algorithm

Logistic Regression is a supervised algorithm used for binary classification.

Class labels:

- `0` = Fail
- `1` = Pass

## Example Output

```text
Enter student's marks: 42
Student will Fail
```

```text
Enter student's marks: 78
Student will Pass
```
