# Software Requirements Specification (SRS)

## 1. Project Identification

**Project Name:** Movie Recommendation Clustering System  
**Folder Name:** `Movies-Recommendation-Kmeans-main`  
**Primary Language:** Python  
**User Interface:** Streamlit dashboard  
**Main Script:** `Movies-Recommendation-kmean.py`  
**Dashboard Script:** `streamlit_app.py`

## 2. Purpose

Group users by movie genre preferences and recommend similar users from the same cluster.

## 3. Scope

The system applies K-Means clustering, visualizes user groups, and supports lookup by UserID.

The project is intended for academic, demonstration, and learning use. It is not designed as a production-grade decision system without further validation, larger datasets, security hardening, and deployment controls.

## 4. Intended Users

- Students
- Recommendation system demo users
- Machine learning learners

## 5. System Overview

The application loads data, preprocesses required fields, trains the configured machine learning or workflow model, evaluates the result where applicable, and presents output through a web dashboard. The dashboard replaces terminal-only input with browser-based controls.

## 6. Functional Requirements

- The system shall provide a browser-based dashboard for the project.
- The system shall load project data from the expected dataset file when available.
- The system shall allow CSV upload from the dashboard where supported.
- The system shall use demo data when a required dataset is not present and demo data is configured.
- The system shall train or initialize the project model from the available data.
- The system shall display important evaluation metrics or workflow results.
- The system shall provide input controls for user prediction or analysis.
- The system shall display the final prediction, cluster, recommendation, or generated research output.
- The system shall show the dataset preview where applicable.
- The system shall fail with a clear message when required columns are missing.

## 7. Non-Functional Requirements

- The dashboard should run locally on Windows using Python 3.
- The interface should be simple enough for academic demonstration.
- The application should avoid requiring external services unless the project specifically depends on them.
- The system should use readable error messages for missing files, missing columns, or invalid data.
- The system should keep the original Python script available for console-based execution.
- The system should be maintainable with common Python libraries such as Pandas, Scikit-learn, Matplotlib, and Streamlit.

## 8. Data Requirements

**Required Data Source:** movies.csv or uploaded CSV with genre preference columns.

**Input Features:**

- Action
- Comedy
- Drama
- Horror
- Romance
- SciFi

**Target or Output Variable:** `Cluster`

## 9. Model and Processing Requirements

**Model or Processing Method:** K-Means Clustering with PCA visualization

- The system shall prepare input features in the same logical format used by the original project.
- The system shall handle categorical encoding where required.
- The system shall split data into training and testing sets for supervised learning projects.
- The system shall compute appropriate metrics for classification, regression, clustering, or workflow output.

## 10. Output Requirements

The system shall display:

- User cluster
- Similar users
- Cluster centers
- Elbow chart
- PCA cluster chart

## 11. External Interface Requirements

### 11.1 User Interface

- The application shall use Streamlit as the web dashboard framework.
- The dashboard shall include sidebar controls for data upload where supported.
- The dashboard shall include tabs or sections for prediction, metrics, charts, and dataset preview where applicable.

### 11.2 Software Interface

- Python packages shall be installed from `Requirements.txt`.
- The dashboard shall be started with Streamlit from the project folder.

## 12. Installation and Run Instructions

Open PowerShell in this folder:

```powershell
cd "E:\visual labs project websites\Movies-Recommendation-Kmeans-main"
```

Install dependencies:

```powershell
python -m pip install -r Requirements.txt
```

Run the dashboard:

```powershell
python -m streamlit run streamlit_app.py
```

Then open the local URL shown by Streamlit, usually:

```text
http://localhost:8501
```

## 13. Assumptions and Constraints

- The system assumes the dataset columns match the required names listed in this SRS.
- Demo datasets are small and are intended only for testing the dashboard flow.
- Model accuracy from demo data should not be treated as real-world performance.
- The dashboard is designed for local execution, not public deployment.
- External API features require valid credentials where applicable.

## 14. Acceptance Criteria

- The dashboard starts without syntax errors.
- The dashboard displays the project title and data source.
- The dashboard trains or initializes the configured model.
- The dashboard displays prediction or analysis output.
- The dashboard displays metrics, charts, or workflow results relevant to the project.
- The dashboard provides a clear error message for missing required columns.
- The README provides enough steps for a user to install dependencies and run the dashboard.

## 15. Future Enhancements

- Add larger real-world datasets.
- Add model persistence and versioning.
- Add advanced validation for user input.
- Add deployment support for cloud hosting.
- Add improved visual design and project-specific reports.
- Add automated tests for preprocessing, model training, and dashboard loading.
