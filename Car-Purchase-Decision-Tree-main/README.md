# Car Purchase Prediction Using Decision Tree

**Name:** Arif alam  
**Enrollment:** 24111590742

## Overview

This machine learning project predicts whether a customer is likely to buy a car by using a Decision Tree classification model. The prediction is based on customer information such as age, gender, annual income, marital status, driving license status, and house ownership.

## Project Purpose

The project shows how a Decision Tree Classifier can be built with Scikit-learn. It includes preprocessing, label encoding, train-test splitting, model training, evaluation, visualization, and prediction from user input.

## Main Features

- Converts categorical values into numeric form using Label Encoding.
- Trains a Decision Tree Classifier.
- Splits the dataset into training and testing data.
- Evaluates the model with accuracy, confusion matrix, and classification report.
- Accepts customer details through the console.
- Predicts whether the customer may purchase a car.
- Displays a visualization of the decision tree.

## Dataset Columns

| Feature | Meaning |
| ------- | ------- |
| Age | Age of the customer |
| Gender | Male or Female |
| Annual_Income | Customer's yearly income |
| Marital_Status | Married or Single |
| Has_Driving_License | Driving license availability |
| Owns_House | House ownership status |
| Buy_Car | Target value: Yes or No |

## Workflow

1. Load the customer dataset.
2. Encode categorical columns.
3. Select input features and target column.
4. Split the data into training and testing sets.
5. Train the Decision Tree model.
6. Evaluate model performance.
7. Take customer details as input.
8. Predict the car purchase decision.

## Technologies

- Python 3.x
- Pandas
- Scikit-learn
- Matplotlib

## Project Files

```text
Car-Purchase-Prediction/
|-- car_data.csv
|-- Car-Purchase-Decision-Tree.py
|-- streamlit_app.py
|-- README.md
|-- requirements.txt
`-- .gitignore
```

## Run Instructions

### Console Version

```bash
git clone https://github.com/your-username/Car-Purchase-Prediction.git
cd Car-Purchase-Prediction
pip install -r requirements.txt
python Car-Purchase-Decision-Tree.py
```

### Streamlit Dashboard

```bash
pip install -r Requirements.txt
streamlit run streamlit_app.py
```

The dashboard looks for `car_data.csv` in the project folder. If the file is not present, it uses a small demo dataset so the dashboard still opens. You can also upload a CSV from the sidebar.

## Concepts Practiced

- Decision Tree Classification
- Gini Index
- Label Encoding
- Data preprocessing
- Feature selection
- Model training and testing
- Confusion matrix
- Classification report
- Decision tree visualization

## Future Scope

- Use a larger real-world dataset.
- Tune Decision Tree hyperparameters.
- Compare results with Random Forest and Logistic Regression.
- Add a GUI.
- Deploy the project with Flask or Streamlit.
