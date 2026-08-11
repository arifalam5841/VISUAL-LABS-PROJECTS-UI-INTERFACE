# Diabetes Prediction Using Logistic Regression

**Name:** Arif alam  
**Enrollment:** 24111590742

## Overview

This project predicts whether a patient is diabetic or non-diabetic using Logistic Regression from Scikit-learn. The model is trained on a diabetes dataset containing medical attributes and then accepts user-entered medical values for prediction.

## Features

- Loads and preprocesses the diabetes dataset.
- Splits the dataset into training and testing sets.
- Trains a Logistic Regression classifier.
- Evaluates the model with accuracy score.
- Displays classification report and confusion matrix.
- Takes medical parameters from the user.
- Predicts diabetic or non-diabetic status.

## Program Flow

The program reads the dataset with Pandas, separates input columns from the target column, splits the data using `train_test_split`, trains a Logistic Regression model, evaluates it using classification metrics, and finally predicts diabetes status from user-provided medical information.

## Dataset

The project uses the Pima Indians Diabetes Dataset.

Input features:

- Pregnancies
- Glucose
- Blood Pressure
- Skin Thickness
- Insulin
- BMI
- Diabetes Pedigree Function
- Age

Target values:

- `0` = Non-Diabetic
- `1` = Diabetic

## Evaluation

- Accuracy Score
- Classification Report
- Confusion Matrix

## Technologies

- Python
- Pandas
- NumPy
- Scikit-learn
- Logistic Regression

## How to Run

1. Clone the repository.
2. Install the required libraries.
3. Place `diabetes.csv` in the project folder.
4. Run the Python script:

```bash
python diabetes_prediction.py
```

5. Enter the requested medical details.
6. View the predicted result.

## Learning Outcomes

- Data preprocessing with Pandas.
- Binary classification with Logistic Regression.
- Train-test splitting.
- Classification model evaluation.
- Prediction using a trained model.
- Building a complete beginner-level machine learning workflow.

## License

This project is created for educational use.
