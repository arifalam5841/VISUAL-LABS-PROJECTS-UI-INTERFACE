# Network Intrusion Detection Using Random Forest

**Name:** Arif alam  
**Enrollment:** 24111590742

## Overview

This project uses machine learning to classify network traffic as normal traffic or intrusion traffic. It applies a Random Forest Classification model trained on network traffic datasets containing normal and DDoS records.

After training, the project evaluates test predictions, analyzes the number of normal and intrusion records, identifies important traffic features, generates visualizations, and saves the trained model for later use.

## Objective

<!-- The aim is to build a model that can detect suspicious traffic and support the basic idea of an Intrusion Detection System (IDS). It demonstrates how supervised machine learning can be applied in network security. -->

## Algorithm

The project uses `RandomForestClassifier`, an ensemble learning method that combines many decision trees for more reliable classification.

Class labels:

```text
0 = Normal Traffic
1 = Intrusion Traffic
```

Model configuration:

```python
RandomForestClassifier(
    n_estimators=100,
    random_state=42,
    n_jobs=-1
)
```

## Dataset

Normal traffic dataset:

```text
Monday-WorkingHours.pcap_ISCX.csv
```

This file is assigned:

```text
Label = 0
```

Intrusion traffic dataset:

```text
Friday-WorkingHours-Afternoon-DDos.pcap_ISCX.csv
```

This file is assigned:

```text
Label = 1
```

## Workflow

```text
Load datasets
Clean column names
Assign labels
Combine datasets
Replace infinite values
Remove missing values
Remove duplicate records
Select numerical features
Split data into train and test sets
Train Random Forest model
Generate predictions
Evaluate model
Analyze predicted classes
Calculate feature importance
Create charts
Save trained model
```

## Program Details

The program loads separate CSV files for normal and intrusion traffic. Column names are cleaned, labels are assigned, and both datasets are merged. The combined dataset is cleaned by replacing infinite values, dropping missing rows, and removing duplicates. Only numerical features are selected before training.

The cleaned data is split in an 80:20 ratio. A Random Forest model with 100 trees is trained and tested. The model is evaluated with accuracy, confusion matrix, and classification report. Prediction counts and the top 10 important network features are visualized. Finally, the trained model is saved as a `.pkl` file.

## Evaluation

- Accuracy Score
- Confusion Matrix
- Classification Report
- Precision
- Recall
- F1-score
- Support

## Prediction Analysis

Example output format:

```text
Final Prediction Analysis
-------------------------
Normal Traffic       : XXXXX
Intrusions Detected  : XXXXX
Total Records        : XXXXX
```

A bar chart is generated to show the final prediction count.

## Feature Importance

Random Forest provides feature-importance values. The project displays the top 10 network features that had the highest effect on intrusion detection.

## Saved Model

The trained model is saved with Joblib:

```text
random_forest_model.pkl
```

It can be loaded later without retraining:

```python
import joblib

model = joblib.load("random_forest_model.pkl")
```

## Technologies

- Python
- Pandas
- NumPy
- Scikit-learn
- Matplotlib
- Joblib
- Random Forest
- Machine Learning
- Network Security

## Project Files

```text
Network-Intrusion-Detection/
|-- Network-Intrusion-Detection.py
|-- requirements.txt
|-- .gitignore
`-- README.md
```

## Requirements

```text
pandas
numpy
matplotlib
scikit-learn
joblib
```

Install dependencies:

```bash
pip install -r requirements.txt
```

## How to Run

```bash
git clone <your-repository-link>
cd Network-Intrusion-Detection
pip install -r requirements.txt
```

Make sure these dataset files are in the project directory:

```text
Monday-WorkingHours.pcap_ISCX.csv
Friday-WorkingHours-Afternoon-DDos.pcap_ISCX.csv
```

Run the program:

```bash
python model-training.py
```

## Main Features

- Loads normal and intrusion traffic data.
- Cleans and preprocesses network records.
- Labels normal and DDoS traffic.
- Removes missing, infinite, and duplicate values.
- Selects numerical features.
- Trains a Random Forest classifier.
- Displays evaluation metrics.
- Performs prediction analysis.
- Shows feature-importance visualization.
- Saves the trained model.

## Learning Outcomes

This project covers supervised learning, Random Forest classification, network intrusion detection, preprocessing, train-test split, model evaluation, confusion matrix, precision, recall, F1-score, feature importance, data visualization, model serialization, and basic network security concepts.

## Future Scope

- Use additional intrusion datasets.
- Detect multiple attack types instead of binary classes.
- Compare Random Forest with XGBoost and Decision Tree.
- Apply feature-selection methods.
- Tune model hyperparameters.
- Build a real-time intrusion detection system.
- Add a web interface.
- Deploy the trained model as an API.

## License

This project is created for educational and internship purposes.
