from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import streamlit as st
from sklearn.cluster import KMeans
from sklearn.compose import ColumnTransformer
from sklearn.decomposition import PCA
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    mean_absolute_error,
    mean_squared_error,
    r2_score,
)
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.tree import DecisionTreeClassifier, plot_tree


PROJECT = Path.cwd().name


CONFIGS = {
    "AQI-Prediction-main": {
        "title": "AQI Prediction Dashboard",
        "task": "regression",
        "data_file": "air_quality_data.csv",
        "target": "AQI",
        "features": ["PM2.5", "PM10", "NO", "NO2", "NOx", "NH3", "CO", "SO2", "O3", "Benzene", "Toluene"],
        "model": "xgb_regressor",
        "prediction_label": "Predicted AQI",
        "unit": "",
        "demo": [
            {"PM2.5": 32, "PM10": 78, "NO": 8, "NO2": 24, "NOx": 33, "NH3": 9, "CO": 0.7, "SO2": 12, "O3": 30, "Benzene": 1.0, "Toluene": 5, "AQI": 92},
            {"PM2.5": 80, "PM10": 160, "NO": 22, "NO2": 44, "NOx": 65, "NH3": 14, "CO": 1.2, "SO2": 18, "O3": 42, "Benzene": 2.5, "Toluene": 10, "AQI": 188},
            {"PM2.5": 18, "PM10": 45, "NO": 5, "NO2": 14, "NOx": 19, "NH3": 6, "CO": 0.4, "SO2": 8, "O3": 24, "Benzene": 0.5, "Toluene": 3, "AQI": 55},
            {"PM2.5": 140, "PM10": 250, "NO": 35, "NO2": 70, "NOx": 95, "NH3": 22, "CO": 2.2, "SO2": 30, "O3": 60, "Benzene": 4.5, "Toluene": 18, "AQI": 315},
            {"PM2.5": 58, "PM10": 120, "NO": 16, "NO2": 31, "NOx": 47, "NH3": 11, "CO": 0.9, "SO2": 15, "O3": 34, "Benzene": 1.7, "Toluene": 7, "AQI": 142},
            {"PM2.5": 210, "PM10": 340, "NO": 45, "NO2": 88, "NOx": 130, "NH3": 30, "CO": 3.1, "SO2": 45, "O3": 75, "Benzene": 6.2, "Toluene": 25, "AQI": 430},
            {"PM2.5": 45, "PM10": 95, "NO": 13, "NO2": 27, "NOx": 40, "NH3": 10, "CO": 0.8, "SO2": 14, "O3": 31, "Benzene": 1.3, "Toluene": 6, "AQI": 118},
            {"PM2.5": 95, "PM10": 190, "NO": 28, "NO2": 55, "NOx": 80, "NH3": 19, "CO": 1.7, "SO2": 25, "O3": 50, "Benzene": 3.4, "Toluene": 14, "AQI": 235},
        ],
    },
    "Credit-card-fraud-detection-main": {
        "title": "Credit Card Fraud Detection Dashboard",
        "task": "classification",
        "target": "Fraud",
        "features": ["Transaction_Amount", "Transaction_Time", "Location_Change", "International"],
        "model": "logistic",
        "label_map": {0: "Legitimate Transaction", 1: "Fraudulent Transaction"},
        "demo": [
            {"Transaction_Amount": 100, "Transaction_Time": 10, "Location_Change": 0, "International": 0, "Fraud": 0},
            {"Transaction_Amount": 250, "Transaction_Time": 14, "Location_Change": 0, "International": 0, "Fraud": 0},
            {"Transaction_Amount": 5000, "Transaction_Time": 2, "Location_Change": 1, "International": 1, "Fraud": 1},
            {"Transaction_Amount": 150, "Transaction_Time": 18, "Location_Change": 0, "International": 0, "Fraud": 0},
            {"Transaction_Amount": 8000, "Transaction_Time": 1, "Location_Change": 1, "International": 1, "Fraud": 1},
            {"Transaction_Amount": 300, "Transaction_Time": 12, "Location_Change": 0, "International": 0, "Fraud": 0},
            {"Transaction_Amount": 12000, "Transaction_Time": 3, "Location_Change": 1, "International": 1, "Fraud": 1},
            {"Transaction_Amount": 450, "Transaction_Time": 20, "Location_Change": 0, "International": 0, "Fraud": 0},
            {"Transaction_Amount": 7000, "Transaction_Time": 4, "Location_Change": 1, "International": 1, "Fraud": 1},
            {"Transaction_Amount": 200, "Transaction_Time": 16, "Location_Change": 0, "International": 0, "Fraud": 0},
        ],
    },
    "Credit-Risk-Prediction-main": {
        "title": "Credit Risk Prediction Dashboard",
        "task": "classification",
        "target": "Risk",
        "features": ["Income", "CreditScore", "LoanAmount"],
        "model": "logistic",
        "label_map": {0: "Low Credit Risk", 1: "High Credit Risk"},
        "demo": [
            {"Income": 30000, "CreditScore": 550, "LoanAmount": 200000, "Risk": 1},
            {"Income": 45000, "CreditScore": 620, "LoanAmount": 150000, "Risk": 1},
            {"Income": 50000, "CreditScore": 700, "LoanAmount": 120000, "Risk": 0},
            {"Income": 60000, "CreditScore": 750, "LoanAmount": 100000, "Risk": 0},
            {"Income": 25000, "CreditScore": 500, "LoanAmount": 250000, "Risk": 1},
            {"Income": 70000, "CreditScore": 780, "LoanAmount": 90000, "Risk": 0},
            {"Income": 80000, "CreditScore": 820, "LoanAmount": 80000, "Risk": 0},
            {"Income": 35000, "CreditScore": 580, "LoanAmount": 180000, "Risk": 1},
            {"Income": 90000, "CreditScore": 850, "LoanAmount": 70000, "Risk": 0},
            {"Income": 40000, "CreditScore": 650, "LoanAmount": 140000, "Risk": 0},
        ],
    },
    "Crop-Yield-Prediction-main": {
        "title": "Crop Yield Prediction Dashboard",
        "task": "regression",
        "data_file": "crop_yield.csv",
        "target": "Yield",
        "features": ["Crop", "Crop_Year", "Season", "State", "Area", "Annual_Rainfall", "Fertilizer", "Pesticide"],
        "categorical": ["Crop", "Season", "State"],
        "model": "rf_regressor",
        "prediction_label": "Predicted Crop Yield",
        "unit": "",
        "demo": [
            {"Crop": "Rice", "Crop_Year": 2020, "Season": "Kharif", "State": "Gujarat", "Area": 1200, "Annual_Rainfall": 850, "Fertilizer": 140, "Pesticide": 18, "Yield": 2.8},
            {"Crop": "Wheat", "Crop_Year": 2020, "Season": "Rabi", "State": "Punjab", "Area": 900, "Annual_Rainfall": 620, "Fertilizer": 125, "Pesticide": 12, "Yield": 3.2},
            {"Crop": "Maize", "Crop_Year": 2021, "Season": "Kharif", "State": "Maharashtra", "Area": 700, "Annual_Rainfall": 760, "Fertilizer": 110, "Pesticide": 10, "Yield": 2.4},
            {"Crop": "Cotton", "Crop_Year": 2021, "Season": "Whole Year", "State": "Gujarat", "Area": 1500, "Annual_Rainfall": 700, "Fertilizer": 160, "Pesticide": 22, "Yield": 1.9},
            {"Crop": "Sugarcane", "Crop_Year": 2022, "Season": "Whole Year", "State": "Uttar Pradesh", "Area": 2000, "Annual_Rainfall": 920, "Fertilizer": 210, "Pesticide": 28, "Yield": 6.4},
            {"Crop": "Rice", "Crop_Year": 2022, "Season": "Kharif", "State": "West Bengal", "Area": 1800, "Annual_Rainfall": 1100, "Fertilizer": 175, "Pesticide": 20, "Yield": 3.9},
            {"Crop": "Wheat", "Crop_Year": 2023, "Season": "Rabi", "State": "Haryana", "Area": 1000, "Annual_Rainfall": 580, "Fertilizer": 135, "Pesticide": 14, "Yield": 3.5},
            {"Crop": "Maize", "Crop_Year": 2023, "Season": "Kharif", "State": "Karnataka", "Area": 850, "Annual_Rainfall": 690, "Fertilizer": 120, "Pesticide": 11, "Yield": 2.6},
        ],
    },
    "Customer-churn-prediction-main": {
        "title": "Customer Churn Prediction Dashboard",
        "task": "classification",
        "target": "Churn",
        "features": ["Age", "MonthlyCharges", "Tenure", "SupportCalls"],
        "model": "logistic",
        "label_map": {0: "Likely to Stay", 1: "Likely to Churn"},
        "demo": [
            {"Age": 22, "MonthlyCharges": 500, "Tenure": 2, "SupportCalls": 5, "Churn": 1},
            {"Age": 35, "MonthlyCharges": 1200, "Tenure": 24, "SupportCalls": 1, "Churn": 0},
            {"Age": 45, "MonthlyCharges": 1500, "Tenure": 36, "SupportCalls": 0, "Churn": 0},
            {"Age": 25, "MonthlyCharges": 700, "Tenure": 5, "SupportCalls": 4, "Churn": 1},
            {"Age": 52, "MonthlyCharges": 2000, "Tenure": 48, "SupportCalls": 0, "Churn": 0},
            {"Age": 23, "MonthlyCharges": 650, "Tenure": 3, "SupportCalls": 6, "Churn": 1},
            {"Age": 40, "MonthlyCharges": 1800, "Tenure": 40, "SupportCalls": 1, "Churn": 0},
            {"Age": 36, "MonthlyCharges": 1300, "Tenure": 30, "SupportCalls": 2, "Churn": 0},
            {"Age": 28, "MonthlyCharges": 800, "Tenure": 8, "SupportCalls": 3, "Churn": 1},
            {"Age": 50, "MonthlyCharges": 2100, "Tenure": 60, "SupportCalls": 0, "Churn": 0},
        ],
    },
    "Customer-Segmentation-Kmeans-main": {
        "title": "Customer Segmentation Dashboard",
        "task": "clustering",
        "data_file": "mall_customers.csv",
        "features": ["Annual Income (k$)", "Spending Score (1-100)"],
        "clusters": 5,
        "demo": [
            {"CustomerID": 1, "Gender": "Male", "Age": 19, "Annual Income (k$)": 15, "Spending Score (1-100)": 39},
            {"CustomerID": 2, "Gender": "Male", "Age": 21, "Annual Income (k$)": 15, "Spending Score (1-100)": 81},
            {"CustomerID": 3, "Gender": "Female", "Age": 20, "Annual Income (k$)": 16, "Spending Score (1-100)": 6},
            {"CustomerID": 4, "Gender": "Female", "Age": 23, "Annual Income (k$)": 16, "Spending Score (1-100)": 77},
            {"CustomerID": 5, "Gender": "Female", "Age": 31, "Annual Income (k$)": 39, "Spending Score (1-100)": 40},
            {"CustomerID": 6, "Gender": "Female", "Age": 35, "Annual Income (k$)": 50, "Spending Score (1-100)": 45},
            {"CustomerID": 7, "Gender": "Male", "Age": 40, "Annual Income (k$)": 70, "Spending Score (1-100)": 20},
            {"CustomerID": 8, "Gender": "Female", "Age": 29, "Annual Income (k$)": 78, "Spending Score (1-100)": 88},
            {"CustomerID": 9, "Gender": "Male", "Age": 45, "Annual Income (k$)": 88, "Spending Score (1-100)": 15},
            {"CustomerID": 10, "Gender": "Female", "Age": 32, "Annual Income (k$)": 86, "Spending Score (1-100)": 95},
        ],
    },
    "Diabetes-Prediction-main": {
        "title": "Diabetes Prediction Dashboard",
        "task": "classification",
        "data_file": "diabetes.csv",
        "target": "Outcome",
        "features": ["Pregnancies", "Glucose", "BloodPressure", "SkinThickness", "Insulin", "BMI", "DiabetesPedigreeFunction", "Age"],
        "model": "logistic",
        "label_map": {0: "Not Diabetic", 1: "Diabetic"},
        "demo": [
            {"Pregnancies": 1, "Glucose": 85, "BloodPressure": 66, "SkinThickness": 29, "Insulin": 0, "BMI": 26.6, "DiabetesPedigreeFunction": 0.351, "Age": 31, "Outcome": 0},
            {"Pregnancies": 8, "Glucose": 183, "BloodPressure": 64, "SkinThickness": 0, "Insulin": 0, "BMI": 23.3, "DiabetesPedigreeFunction": 0.672, "Age": 32, "Outcome": 1},
            {"Pregnancies": 1, "Glucose": 89, "BloodPressure": 66, "SkinThickness": 23, "Insulin": 94, "BMI": 28.1, "DiabetesPedigreeFunction": 0.167, "Age": 21, "Outcome": 0},
            {"Pregnancies": 3, "Glucose": 78, "BloodPressure": 50, "SkinThickness": 32, "Insulin": 88, "BMI": 31.0, "DiabetesPedigreeFunction": 0.248, "Age": 26, "Outcome": 1},
            {"Pregnancies": 5, "Glucose": 116, "BloodPressure": 74, "SkinThickness": 0, "Insulin": 0, "BMI": 25.6, "DiabetesPedigreeFunction": 0.201, "Age": 30, "Outcome": 0},
            {"Pregnancies": 10, "Glucose": 168, "BloodPressure": 74, "SkinThickness": 0, "Insulin": 0, "BMI": 38.0, "DiabetesPedigreeFunction": 0.537, "Age": 34, "Outcome": 1},
            {"Pregnancies": 2, "Glucose": 107, "BloodPressure": 74, "SkinThickness": 30, "Insulin": 100, "BMI": 33.6, "DiabetesPedigreeFunction": 0.404, "Age": 23, "Outcome": 0},
            {"Pregnancies": 7, "Glucose": 196, "BloodPressure": 90, "SkinThickness": 0, "Insulin": 0, "BMI": 39.8, "DiabetesPedigreeFunction": 0.451, "Age": 41, "Outcome": 1},
        ],
    },
    "Electricity-Consumption-Prediction-main": {
        "title": "Electricity Consumption Prediction Dashboard",
        "task": "regression",
        "data_file": "smart_meter_data.csv",
        "target": "Electricity_Consumed",
        "features": ["Temperature", "Humidity", "Wind_Speed", "Avg_Past_Consumption", "Year", "Month", "Day", "Hour", "Minute", "DayOfWeek"],
        "timestamp": "Timestamp",
        "model": "xgb_regressor",
        "prediction_label": "Predicted Electricity Consumption",
        "unit": "kWh",
        "demo": [
            {"Timestamp": "2024-01-01 00:00:00", "Temperature": 18, "Humidity": 62, "Wind_Speed": 8, "Avg_Past_Consumption": 2.1, "Electricity_Consumed": 2.3},
            {"Timestamp": "2024-01-01 01:00:00", "Temperature": 17, "Humidity": 64, "Wind_Speed": 7, "Avg_Past_Consumption": 2.0, "Electricity_Consumed": 2.1},
            {"Timestamp": "2024-01-01 08:00:00", "Temperature": 22, "Humidity": 58, "Wind_Speed": 9, "Avg_Past_Consumption": 3.0, "Electricity_Consumed": 3.4},
            {"Timestamp": "2024-01-01 12:00:00", "Temperature": 30, "Humidity": 45, "Wind_Speed": 12, "Avg_Past_Consumption": 4.0, "Electricity_Consumed": 4.8},
            {"Timestamp": "2024-01-01 18:00:00", "Temperature": 27, "Humidity": 50, "Wind_Speed": 10, "Avg_Past_Consumption": 5.5, "Electricity_Consumed": 6.1},
            {"Timestamp": "2024-01-02 00:00:00", "Temperature": 19, "Humidity": 61, "Wind_Speed": 8, "Avg_Past_Consumption": 2.4, "Electricity_Consumed": 2.6},
            {"Timestamp": "2024-01-02 12:00:00", "Temperature": 32, "Humidity": 42, "Wind_Speed": 14, "Avg_Past_Consumption": 4.3, "Electricity_Consumed": 5.0},
            {"Timestamp": "2024-01-02 19:00:00", "Temperature": 28, "Humidity": 49, "Wind_Speed": 11, "Avg_Past_Consumption": 5.8, "Electricity_Consumed": 6.5},
        ],
    },
    "Email-spam-detection-main": {
        "title": "Email Spam Detection Dashboard",
        "task": "text_classification",
        "data_file": "completeSpamAssassin.csv",
        "text_column": "Body",
        "target": "Label",
        "model": "logistic",
        "label_map": {0: "Not Spam", 1: "Spam"},
        "demo": [
            {"Body": "Meeting agenda attached for tomorrow morning", "Label": 0},
            {"Body": "Congratulations you won a free phone claim now", "Label": 1},
            {"Body": "Please review the quarterly sales report", "Label": 0},
            {"Body": "Win cash prize click this link immediately", "Label": 1},
            {"Body": "Your invoice has been processed successfully", "Label": 0},
            {"Body": "Limited offer free lottery reward waiting", "Label": 1},
        ],
    },
    "Gmail-Detection-API-Naive-Bayes-main": {
        "title": "Gmail Spam Detection Dashboard",
        "task": "text_classification",
        "data_file": "spam.csv",
        "text_column": "Email",
        "target": "Label",
        "model": "naive_bayes",
        "spam_csv": True,
        "label_map": {0: "Ham", 1: "Spam"},
        "demo": [
            {"Label": 0, "Email": "Are we still meeting at the office today"},
            {"Label": 1, "Email": "Free entry in a weekly prize draw text now"},
            {"Label": 0, "Email": "Please call me when you reach home"},
            {"Label": 1, "Email": "Claim your guaranteed cash reward today"},
            {"Label": 0, "Email": "Can you send the project files"},
            {"Label": 1, "Email": "You have won a vacation click to claim"},
        ],
    },
    "House-Price-Prediction-main": {
        "title": "House Price Prediction Dashboard",
        "task": "regression",
        "target": "Price",
        "features": ["Area"],
        "model": "linear",
        "prediction_label": "Predicted House Price",
        "unit": "Lakhs",
        "demo": [
            {"Area": 600, "Price": 30},
            {"Area": 800, "Price": 40},
            {"Area": 1000, "Price": 50},
            {"Area": 1200, "Price": 60},
            {"Area": 1500, "Price": 75},
            {"Area": 1800, "Price": 90},
            {"Area": 2000, "Price": 100},
            {"Area": 2200, "Price": 110},
            {"Area": 2500, "Price": 125},
        ],
    },
    "Loan-Approval-Decision-Tree-main": {
        "title": "Loan Approval Decision Tree Dashboard",
        "task": "classification",
        "data_file": "data.csv",
        "target": "Loan_Approved",
        "features": ["Age", "Monthly_Income", "Credit_Score", "Employment_Status", "Existing_Loan"],
        "categorical": ["Employment_Status", "Existing_Loan"],
        "model": "decision_tree",
        "label_map": {"No": "Loan Rejected", "Yes": "Loan Approved", 0: "Loan Rejected", 1: "Loan Approved"},
        "demo": [
            {"Age": 25, "Monthly_Income": 30000, "Credit_Score": 610, "Employment_Status": "Unemployed", "Existing_Loan": "Yes", "Loan_Approved": "No"},
            {"Age": 32, "Monthly_Income": 55000, "Credit_Score": 700, "Employment_Status": "Employed", "Existing_Loan": "No", "Loan_Approved": "Yes"},
            {"Age": 45, "Monthly_Income": 80000, "Credit_Score": 760, "Employment_Status": "Employed", "Existing_Loan": "No", "Loan_Approved": "Yes"},
            {"Age": 29, "Monthly_Income": 42000, "Credit_Score": 650, "Employment_Status": "Employed", "Existing_Loan": "Yes", "Loan_Approved": "No"},
            {"Age": 37, "Monthly_Income": 62000, "Credit_Score": 720, "Employment_Status": "Employed", "Existing_Loan": "No", "Loan_Approved": "Yes"},
            {"Age": 23, "Monthly_Income": 25000, "Credit_Score": 580, "Employment_Status": "Unemployed", "Existing_Loan": "Yes", "Loan_Approved": "No"},
            {"Age": 50, "Monthly_Income": 95000, "Credit_Score": 790, "Employment_Status": "Employed", "Existing_Loan": "No", "Loan_Approved": "Yes"},
            {"Age": 41, "Monthly_Income": 48000, "Credit_Score": 630, "Employment_Status": "Unemployed", "Existing_Loan": "No", "Loan_Approved": "No"},
        ],
    },
    "Loan-Approval-main": {
        "title": "Loan Approval Dashboard",
        "task": "classification",
        "target": "Loan_Status",
        "features": ["Income", "Credit_Score"],
        "model": "logistic",
        "label_map": {0: "Loan Rejected", 1: "Loan Approved"},
        "demo": [
            {"Income": 30000, "Credit_Score": 650, "Loan_Status": 0},
            {"Income": 40000, "Credit_Score": 700, "Loan_Status": 0},
            {"Income": 50000, "Credit_Score": 750, "Loan_Status": 1},
            {"Income": 60000, "Credit_Score": 800, "Loan_Status": 1},
            {"Income": 25000, "Credit_Score": 600, "Loan_Status": 0},
            {"Income": 70000, "Credit_Score": 820, "Loan_Status": 1},
            {"Income": 45000, "Credit_Score": 710, "Loan_Status": 0},
            {"Income": 80000, "Credit_Score": 840, "Loan_Status": 1},
        ],
    },
    "Movies-Recommendation-Kmeans-main": {
        "title": "Movie Recommendation Clustering Dashboard",
        "task": "clustering",
        "data_file": "movies.csv",
        "features": ["Action", "Comedy", "Drama", "Horror", "Romance", "SciFi"],
        "clusters": 3,
        "id_column": "UserID",
        "demo": [
            {"UserID": 1, "Age": 21, "Action": 5, "Comedy": 1, "Drama": 2, "Horror": 4, "Romance": 1, "SciFi": 5},
            {"UserID": 2, "Age": 25, "Action": 4, "Comedy": 2, "Drama": 2, "Horror": 5, "Romance": 1, "SciFi": 4},
            {"UserID": 3, "Age": 30, "Action": 1, "Comedy": 5, "Drama": 4, "Horror": 1, "Romance": 5, "SciFi": 2},
            {"UserID": 4, "Age": 34, "Action": 2, "Comedy": 4, "Drama": 5, "Horror": 1, "Romance": 4, "SciFi": 1},
            {"UserID": 5, "Age": 28, "Action": 5, "Comedy": 2, "Drama": 1, "Horror": 5, "Romance": 1, "SciFi": 5},
            {"UserID": 6, "Age": 40, "Action": 1, "Comedy": 2, "Drama": 5, "Horror": 1, "Romance": 4, "SciFi": 2},
            {"UserID": 7, "Age": 23, "Action": 4, "Comedy": 3, "Drama": 2, "Horror": 4, "Romance": 2, "SciFi": 5},
            {"UserID": 8, "Age": 37, "Action": 2, "Comedy": 5, "Drama": 4, "Horror": 1, "Romance": 5, "SciFi": 1},
        ],
    },
    "Network-Intrusion-Detection-main": {
        "title": "Network Intrusion Detection Dashboard",
        "task": "network",
        "normal_file": "Monday-WorkingHours.pcap_ISCX.csv",
        "intrusion_file": "Friday-WorkingHours-Afternoon-DDos.pcap_ISCX.csv",
        "label_map": {0: "Normal Traffic", 1: "Intrusion Detected"},
    },
    "Student-Marks-Prediction-main": {
        "title": "Student Marks Prediction Dashboard",
        "task": "regression",
        "target": "Marks",
        "features": ["Hours"],
        "model": "linear",
        "prediction_label": "Predicted Marks",
        "unit": "",
        "demo": [
            {"Hours": 2, "Marks": 35},
            {"Hours": 3, "Marks": 45},
            {"Hours": 4, "Marks": 50},
            {"Hours": 5, "Marks": 60},
            {"Hours": 6, "Marks": 65},
            {"Hours": 7, "Marks": 72},
            {"Hours": 8, "Marks": 80},
            {"Hours": 9, "Marks": 88},
            {"Hours": 10, "Marks": 95},
        ],
    },
    "Student-Pass-Fail-main": {
        "title": "Student Pass/Fail Dashboard",
        "task": "classification",
        "target": "Result",
        "features": ["Marks"],
        "model": "logistic",
        "label_map": {0: "Fail", 1: "Pass"},
        "demo": [
            {"Marks": 20, "Result": 0},
            {"Marks": 30, "Result": 0},
            {"Marks": 35, "Result": 0},
            {"Marks": 40, "Result": 0},
            {"Marks": 50, "Result": 1},
            {"Marks": 60, "Result": 1},
            {"Marks": 70, "Result": 1},
            {"Marks": 80, "Result": 1},
        ],
    },
}


def dataframe_from_demo(config):
    return pd.DataFrame(config.get("demo", []))


def load_table(config):
    uploaded = st.sidebar.file_uploader("Upload CSV", type=["csv"])
    if uploaded is not None:
        return pd.read_csv(uploaded), "Uploaded CSV"

    data_file = config.get("data_file")
    if data_file and Path(data_file).exists():
        if config.get("spam_csv"):
            df = pd.read_csv(data_file, encoding="latin-1")
        else:
            df = pd.read_csv(data_file)
        return normalize_loaded_data(df, config), data_file

    return dataframe_from_demo(config), "Demo data"


def normalize_loaded_data(df, config):
    cleaned = df.copy()
    if config.get("spam_csv"):
        if {"v1", "v2"}.issubset(cleaned.columns):
            cleaned = cleaned[["v1", "v2"]]
            cleaned.columns = ["Label", "Email"]
        cleaned["Label"] = cleaned["Label"].map({"ham": 0, "spam": 1}).fillna(cleaned["Label"])

    if config.get("data_file") == "completeSpamAssassin.csv":
        unnamed = [column for column in cleaned.columns if column.lower().startswith("unnamed")]
        if unnamed:
            cleaned = cleaned.drop(columns=unnamed)
        if "Body" in cleaned.columns:
            cleaned = cleaned.dropna(subset=["Body"])
            cleaned["Body"] = cleaned["Body"].astype(str)

    return cleaned


def validate_columns(df, columns):
    missing = [column for column in columns if column not in df.columns]
    if missing:
        st.error("Missing columns: " + ", ".join(missing))
        st.stop()


def add_time_features(df, timestamp_column):
    data = df.copy()
    data[timestamp_column] = pd.to_datetime(data[timestamp_column], errors="coerce")
    data = data.dropna(subset=[timestamp_column])
    data["Year"] = data[timestamp_column].dt.year
    data["Month"] = data[timestamp_column].dt.month
    data["Day"] = data[timestamp_column].dt.day
    data["Hour"] = data[timestamp_column].dt.hour
    data["Minute"] = data[timestamp_column].dt.minute
    data["DayOfWeek"] = data[timestamp_column].dt.dayofweek
    return data


def label_text(value, config):
    mapping = config.get("label_map", {})
    return str(mapping.get(value, mapping.get(str(value), value)))


def make_estimator(config, task):
    name = config.get("model")
    if name == "linear":
        return LinearRegression()
    if name == "decision_tree":
        return DecisionTreeClassifier(criterion="gini", random_state=42)
    if name == "rf_regressor":
        return RandomForestRegressor(n_estimators=200, random_state=42, n_jobs=-1)
    if name == "xgb_regressor":
        try:
            from xgboost import XGBRegressor

            return XGBRegressor(
                n_estimators=200,
                learning_rate=0.05,
                max_depth=6,
                subsample=0.8,
                colsample_bytree=0.8,
                random_state=42,
                objective="reg:squarederror",
            )
        except Exception:
            return RandomForestRegressor(n_estimators=200, random_state=42, n_jobs=-1)
    if task == "classification":
        return LogisticRegression(max_iter=1000)
    return RandomForestRegressor(n_estimators=200, random_state=42, n_jobs=-1)


def make_pipeline(config, task):
    features = config["features"]
    categorical = config.get("categorical", [])
    numeric = [feature for feature in features if feature not in categorical]
    estimator = make_estimator(config, task)

    if categorical:
        preprocessor = ColumnTransformer(
            transformers=[
                ("categorical", OneHotEncoder(handle_unknown="ignore"), categorical),
                ("numeric", "passthrough", numeric),
            ]
        )
        return Pipeline([("preprocessor", preprocessor), ("model", estimator)])

    return estimator


def split_data(x, y):
    stratify = None
    if y.nunique() > 1 and y.value_counts().min() >= 2:
        stratify = y
    return train_test_split(x, y, test_size=0.2, random_state=42, stratify=stratify)


def render_metrics_header(source, row_count, test_count, score_label, score_value):
    cols = st.columns(4)
    cols[0].metric("Dataset Source", source)
    cols[1].metric("Rows", f"{row_count:,}")
    cols[2].metric("Test Rows", f"{test_count:,}")
    cols[3].metric(score_label, score_value)


def numeric_input_for(df, column):
    series = pd.to_numeric(df[column], errors="coerce").dropna()
    if series.empty:
        return st.number_input(column, value=0.0)
    min_value = float(series.min())
    max_value = float(series.max())
    mean_value = float(series.mean())
    step = max((max_value - min_value) / 100, 1.0)
    return st.number_input(column, value=mean_value, step=step)


def customer_form(df, features, categorical):
    values = {}
    columns = st.columns(2)
    for index, feature in enumerate(features):
        with columns[index % 2]:
            if feature in categorical or not pd.api.types.is_numeric_dtype(df[feature]):
                options = sorted(df[feature].dropna().astype(str).unique().tolist())
                values[feature] = st.selectbox(feature, options)
            else:
                values[feature] = numeric_input_for(df, feature)
    return pd.DataFrame([values], columns=features)


def render_regression(config):
    df, source = load_table(config)
    if config.get("timestamp"):
        validate_columns(df, [config["timestamp"], config["target"]])
        df = add_time_features(df, config["timestamp"])

    validate_columns(df, config["features"] + [config["target"]])
    df = df.dropna(subset=config["features"] + [config["target"]])
    if len(df) < 5:
        st.error("The dataset needs at least five usable rows.")
        st.stop()

    x = df[config["features"]]
    y = pd.to_numeric(df[config["target"]], errors="coerce")
    valid = y.notna()
    x = x.loc[valid]
    y = y.loc[valid]
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)
    model = make_pipeline(config, "regression")
    model.fit(x_train, y_train)
    y_pred = model.predict(x_test)

    mae = mean_absolute_error(y_test, y_pred)
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    r2 = r2_score(y_test, y_pred) if len(y_test) > 1 else 0
    render_metrics_header(source, len(df), len(x_test), "R2 Score", f"{r2:.3f}")

    tab_predict, tab_metrics, tab_data = st.tabs(["Prediction", "Model Metrics", "Dataset"])
    with tab_predict:
        st.subheader("Prediction Input")
        user_input = customer_form(df, config["features"], config.get("categorical", []))
        prediction = float(model.predict(user_input)[0])
        if config["title"].startswith("AQI"):
            prediction = max(0, prediction)
            st.metric(config.get("prediction_label", "Prediction"), f"{prediction:.2f}", get_aqi_category(prediction))
        else:
            unit = config.get("unit", "")
            suffix = f" {unit}" if unit else ""
            st.metric(config.get("prediction_label", "Prediction"), f"{prediction:.2f}{suffix}")

    with tab_metrics:
        c1, c2, c3 = st.columns(3)
        c1.metric("MAE", f"{mae:.3f}")
        c2.metric("RMSE", f"{rmse:.3f}")
        c3.metric("R2", f"{r2:.3f}")
        chart_data = pd.DataFrame({"Actual": y_test.values, "Predicted": y_pred})
        st.line_chart(chart_data.reset_index(drop=True))
        fig, ax = plt.subplots(figsize=(7, 5))
        ax.scatter(y_test, y_pred, alpha=0.7)
        low = min(y_test.min(), y_pred.min())
        high = max(y_test.max(), y_pred.max())
        ax.plot([low, high], [low, high], linestyle="--")
        ax.set_xlabel("Actual")
        ax.set_ylabel("Predicted")
        ax.set_title("Actual vs Predicted")
        st.pyplot(fig)

    with tab_data:
        st.dataframe(df, use_container_width=True)


def get_aqi_category(aqi):
    if aqi <= 50:
        return "Good"
    if aqi <= 100:
        return "Satisfactory"
    if aqi <= 200:
        return "Moderate"
    if aqi <= 300:
        return "Poor"
    if aqi <= 400:
        return "Very Poor"
    return "Severe"


def render_classification(config):
    df, source = load_table(config)
    validate_columns(df, config["features"] + [config["target"]])
    df = df.dropna(subset=config["features"] + [config["target"]])
    if len(df) < 5:
        st.error("The dataset needs at least five usable rows.")
        st.stop()

    x = df[config["features"]]
    y = df[config["target"]]
    x_train, x_test, y_train, y_test = split_data(x, y)
    model = make_pipeline(config, "classification")
    model.fit(x_train, y_train)
    y_pred = model.predict(x_test)
    accuracy = accuracy_score(y_test, y_pred)
    render_metrics_header(source, len(df), len(x_test), "Accuracy", f"{accuracy:.2%}")

    tab_predict, tab_metrics, tab_data = st.tabs(["Prediction", "Model Metrics", "Dataset"])
    with tab_predict:
        st.subheader("Prediction Input")
        user_input = customer_form(df, config["features"], config.get("categorical", []))
        prediction = model.predict(user_input)[0]
        st.metric("Prediction", label_text(prediction, config))
        if hasattr(model, "predict_proba"):
            probability = model.predict_proba(user_input)[0].max()
            st.caption(f"Model confidence: {probability:.2%}")

    with tab_metrics:
        labels = sorted(pd.Series(list(y_test) + list(y_pred)).dropna().unique().tolist())
        matrix = pd.DataFrame(
            confusion_matrix(y_test, y_pred, labels=labels),
            index=[f"Actual {label_text(label, config)}" for label in labels],
            columns=[f"Predicted {label_text(label, config)}" for label in labels],
        )
        st.subheader("Confusion Matrix")
        st.dataframe(matrix, use_container_width=True)
        report = pd.DataFrame(classification_report(y_test, y_pred, output_dict=True, zero_division=0)).transpose()
        st.subheader("Classification Report")
        st.dataframe(report.style.format("{:.2f}"), use_container_width=True)

        if config.get("model") == "decision_tree":
            tree_model = model.named_steps["model"] if isinstance(model, Pipeline) else model
            fig, ax = plt.subplots(figsize=(14, 7))
            plot_tree(tree_model, filled=True, rounded=True, ax=ax)
            st.pyplot(fig)

    with tab_data:
        st.dataframe(df, use_container_width=True)
        st.bar_chart(df[config["target"]].astype(str).value_counts())


def render_clustering(config):
    df, source = load_table(config)
    validate_columns(df, config["features"])
    df = df.dropna(subset=config["features"]).copy()
    k = min(config.get("clusters", 3), len(df))
    scaler = StandardScaler()
    x_scaled = scaler.fit_transform(df[config["features"]])
    model = KMeans(n_clusters=k, random_state=42, n_init=10)
    df["Cluster"] = model.fit_predict(x_scaled)
    centers = pd.DataFrame(scaler.inverse_transform(model.cluster_centers_), columns=config["features"])

    render_metrics_header(source, len(df), k, "Clusters", str(k))
    tab_predict, tab_chart, tab_data = st.tabs(["Cluster Lookup", "Charts", "Dataset"])

    with tab_predict:
        if config.get("id_column") and config["id_column"] in df.columns:
            selected_id = st.selectbox("User ID", sorted(df[config["id_column"]].unique().tolist()))
            cluster = int(df.loc[df[config["id_column"]] == selected_id, "Cluster"].iloc[0])
            st.metric("Cluster", cluster)
            st.subheader("Similar Users")
            st.dataframe(df[df["Cluster"] == cluster], use_container_width=True)
        else:
            user_input = customer_form(df, config["features"], [])
            cluster = int(model.predict(scaler.transform(user_input))[0])
            st.metric("Predicted Cluster", cluster)
        st.subheader("Cluster Centers")
        st.dataframe(centers, use_container_width=True)

    with tab_chart:
        wcss = []
        max_k = min(10, len(df))
        for cluster_count in range(1, max_k + 1):
            temp = KMeans(n_clusters=cluster_count, random_state=42, n_init=10)
            temp.fit(x_scaled)
            wcss.append(temp.inertia_)
        st.line_chart(pd.DataFrame({"K": range(1, max_k + 1), "WCSS": wcss}).set_index("K"))

        if len(config["features"]) == 2:
            fig, ax = plt.subplots(figsize=(8, 5))
            scatter = ax.scatter(df[config["features"][0]], df[config["features"][1]], c=df["Cluster"], cmap="viridis", s=70)
            ax.scatter(centers[config["features"][0]], centers[config["features"][1]], marker="X", c="red", s=180)
            ax.set_xlabel(config["features"][0])
            ax.set_ylabel(config["features"][1])
            ax.set_title("Cluster Visualization")
            st.pyplot(fig)
        else:
            pca = PCA(n_components=2)
            projected = pca.fit_transform(x_scaled)
            fig, ax = plt.subplots(figsize=(8, 5))
            ax.scatter(projected[:, 0], projected[:, 1], c=df["Cluster"], cmap="viridis", s=70)
            ax.set_xlabel("Principal Component 1")
            ax.set_ylabel("Principal Component 2")
            ax.set_title("Cluster Visualization")
            st.pyplot(fig)

    with tab_data:
        st.dataframe(df, use_container_width=True)


def render_text_classification(config):
    df, source = load_table(config)
    validate_columns(df, [config["text_column"], config["target"]])
    df = df.dropna(subset=[config["text_column"], config["target"]]).copy()
    df[config["text_column"]] = df[config["text_column"]].astype(str)
    x = df[config["text_column"]]
    y = df[config["target"]]
    x_train, x_test, y_train, y_test = split_data(x, y)

    estimator = MultinomialNB() if config.get("model") == "naive_bayes" else LogisticRegression(max_iter=1000)
    model = Pipeline([
        ("vectorizer", CountVectorizer(stop_words="english")),
        ("model", estimator),
    ])
    model.fit(x_train, y_train)
    y_pred = model.predict(x_test)
    accuracy = accuracy_score(y_test, y_pred)
    render_metrics_header(source, len(df), len(x_test), "Accuracy", f"{accuracy:.2%}")

    tab_predict, tab_metrics, tab_data = st.tabs(["Prediction", "Model Metrics", "Dataset"])
    with tab_predict:
        text = st.text_area("Email Text", height=180, value="Congratulations! You have won a free prize. Click now to claim.")
        prediction = model.predict([text])[0]
        st.metric("Prediction", label_text(prediction, config))
        if hasattr(model.named_steps["model"], "predict_proba"):
            probability = model.predict_proba([text])[0].max()
            st.caption(f"Model confidence: {probability:.2%}")

    with tab_metrics:
        labels = sorted(pd.Series(list(y_test) + list(y_pred)).dropna().unique().tolist())
        matrix = pd.DataFrame(
            confusion_matrix(y_test, y_pred, labels=labels),
            index=[f"Actual {label_text(label, config)}" for label in labels],
            columns=[f"Predicted {label_text(label, config)}" for label in labels],
        )
        st.dataframe(matrix, use_container_width=True)
        report = pd.DataFrame(classification_report(y_test, y_pred, output_dict=True, zero_division=0)).transpose()
        st.dataframe(report.style.format("{:.2f}"), use_container_width=True)

    with tab_data:
        st.dataframe(df, use_container_width=True)
        st.bar_chart(df[config["target"]].astype(str).value_counts())


def network_demo_data():
    normal = pd.DataFrame([
        {"Flow Duration": 1000, "Total Fwd Packets": 8, "Total Backward Packets": 7, "Flow Bytes/s": 1500, "Flow Packets/s": 12},
        {"Flow Duration": 1200, "Total Fwd Packets": 7, "Total Backward Packets": 8, "Flow Bytes/s": 1400, "Flow Packets/s": 10},
        {"Flow Duration": 980, "Total Fwd Packets": 9, "Total Backward Packets": 6, "Flow Bytes/s": 1600, "Flow Packets/s": 13},
        {"Flow Duration": 1500, "Total Fwd Packets": 6, "Total Backward Packets": 6, "Flow Bytes/s": 1200, "Flow Packets/s": 9},
    ])
    intrusion = pd.DataFrame([
        {"Flow Duration": 90, "Total Fwd Packets": 80, "Total Backward Packets": 2, "Flow Bytes/s": 90000, "Flow Packets/s": 900},
        {"Flow Duration": 110, "Total Fwd Packets": 95, "Total Backward Packets": 3, "Flow Bytes/s": 120000, "Flow Packets/s": 850},
        {"Flow Duration": 75, "Total Fwd Packets": 100, "Total Backward Packets": 1, "Flow Bytes/s": 150000, "Flow Packets/s": 1100},
        {"Flow Duration": 130, "Total Fwd Packets": 88, "Total Backward Packets": 2, "Flow Bytes/s": 98000, "Flow Packets/s": 760},
    ])
    return normal, intrusion, "Demo data"


def load_network_data(config):
    normal_upload = st.sidebar.file_uploader("Upload normal traffic CSV", type=["csv"])
    intrusion_upload = st.sidebar.file_uploader("Upload intrusion traffic CSV", type=["csv"])
    if normal_upload is not None and intrusion_upload is not None:
        return pd.read_csv(normal_upload), pd.read_csv(intrusion_upload), "Uploaded CSVs"

    if Path(config["normal_file"]).exists() and Path(config["intrusion_file"]).exists():
        return pd.read_csv(config["normal_file"]), pd.read_csv(config["intrusion_file"]), "Project CSVs"

    return network_demo_data()


def render_network(config):
    normal, intrusion, source = load_network_data(config)
    normal.columns = normal.columns.str.strip()
    intrusion.columns = intrusion.columns.str.strip()
    normal["Label"] = 0
    intrusion["Label"] = 1
    df = pd.concat([normal, intrusion], ignore_index=True)
    df = df.replace([np.inf, -np.inf], np.nan).dropna().drop_duplicates()
    x = df.drop(columns=["Label"]).select_dtypes(include=["number"])
    y = df["Label"]
    if x.empty or y.nunique() < 2:
        st.error("The dataset needs numerical traffic features and both normal and intrusion rows.")
        st.stop()

    x_train, x_test, y_train, y_test = split_data(x, y)
    model = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1)
    model.fit(x_train, y_train)
    y_pred = model.predict(x_test)
    accuracy = accuracy_score(y_test, y_pred)
    render_metrics_header(source, len(df), len(x_test), "Accuracy", f"{accuracy:.2%}")

    tab_predict, tab_metrics, tab_data = st.tabs(["Prediction", "Model Metrics", "Dataset"])
    with tab_predict:
        st.subheader("Traffic Record")
        input_values = {}
        columns = st.columns(2)
        for index, feature in enumerate(x.columns[:8]):
            with columns[index % 2]:
                input_values[feature] = numeric_input_for(x, feature)
        for feature in x.columns[8:]:
            input_values[feature] = float(x[feature].median())
        user_input = pd.DataFrame([input_values], columns=x.columns)
        prediction = int(model.predict(user_input)[0])
        st.metric("Prediction", label_text(prediction, config))

    with tab_metrics:
        labels = [0, 1]
        matrix = pd.DataFrame(
            confusion_matrix(y_test, y_pred, labels=labels),
            index=[f"Actual {label_text(label, config)}" for label in labels],
            columns=[f"Predicted {label_text(label, config)}" for label in labels],
        )
        st.dataframe(matrix, use_container_width=True)
        report = pd.DataFrame(classification_report(y_test, y_pred, output_dict=True, zero_division=0)).transpose()
        st.dataframe(report.style.format("{:.2f}"), use_container_width=True)
        importance = pd.DataFrame({"Feature": x.columns, "Importance": model.feature_importances_}).sort_values("Importance", ascending=False).head(10)
        st.bar_chart(importance.set_index("Feature"))

    with tab_data:
        st.dataframe(df, use_container_width=True)
        st.bar_chart(df["Label"].map(config["label_map"]).value_counts())


def main():
    config = CONFIGS.get(PROJECT)
    if config is None:
        st.error(f"No dashboard configuration found for {PROJECT}.")
        st.stop()

    st.set_page_config(page_title=config["title"], layout="wide")
    st.title(config["title"])
    st.caption("Web dashboard for model training, evaluation, visualization, and prediction.")

    with st.sidebar:
        st.header("Data")
        st.caption("Use the project CSV if available, upload a CSV, or run with demo data.")

    task = config["task"]
    if task == "regression":
        render_regression(config)
    elif task == "classification":
        render_classification(config)
    elif task == "clustering":
        render_clustering(config)
    elif task == "text_classification":
        render_text_classification(config)
    elif task == "network":
        render_network(config)
    else:
        st.error(f"Unsupported task: {task}")


if __name__ == "__main__":
    main()

