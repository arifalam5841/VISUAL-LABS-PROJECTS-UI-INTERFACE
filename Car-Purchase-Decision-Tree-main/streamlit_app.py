from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import streamlit as st
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.tree import DecisionTreeClassifier, plot_tree


DATA_PATH = Path("car_data.csv")
CATEGORICAL_COLUMNS = [
    "Gender",
    "Marital_Status",
    "Has_Driving_License",
    "Owns_House",
    "Buy_Car",
]
FEATURE_COLUMNS = [
    "Age",
    "Gender",
    "Annual_Income",
    "Marital_Status",
    "Has_Driving_License",
    "Owns_House",
]
TARGET_COLUMN = "Buy_Car"


def demo_data() -> pd.DataFrame:
    return pd.DataFrame(
        [
            [22, "Male", 280000, "Single", "Yes", "No", "No"],
            [25, "Female", 360000, "Single", "Yes", "No", "No"],
            [31, "Male", 620000, "Married", "Yes", "Yes", "Yes"],
            [34, "Female", 540000, "Married", "Yes", "Yes", "Yes"],
            [28, "Male", 410000, "Single", "No", "No", "No"],
            [42, "Female", 820000, "Married", "Yes", "Yes", "Yes"],
            [47, "Male", 900000, "Married", "Yes", "Yes", "Yes"],
            [39, "Female", 690000, "Single", "Yes", "No", "Yes"],
            [52, "Male", 760000, "Married", "No", "Yes", "No"],
            [23, "Female", 300000, "Single", "No", "No", "No"],
            [45, "Male", 1200000, "Married", "Yes", "Yes", "Yes"],
            [37, "Female", 580000, "Married", "Yes", "No", "Yes"],
        ],
        columns=[
            "Age",
            "Gender",
            "Annual_Income",
            "Marital_Status",
            "Has_Driving_License",
            "Owns_House",
            "Buy_Car",
        ],
    )


def load_data(uploaded_file) -> tuple[pd.DataFrame, str]:
    if uploaded_file is not None:
        return pd.read_csv(uploaded_file), "Uploaded CSV"

    if DATA_PATH.exists():
        return pd.read_csv(DATA_PATH), str(DATA_PATH)

    return demo_data(), "Demo data"


def validate_data(data: pd.DataFrame) -> list[str]:
    missing_columns = [
        column for column in FEATURE_COLUMNS + [TARGET_COLUMN] if column not in data.columns
    ]
    errors = []

    if missing_columns:
        errors.append(f"Missing columns: {', '.join(missing_columns)}")

    if data.empty:
        errors.append("Dataset is empty.")

    return errors


def encode_data(data: pd.DataFrame) -> tuple[pd.DataFrame, dict[str, LabelEncoder]]:
    encoded = data.copy()
    encoders = {}

    for column in CATEGORICAL_COLUMNS:
        encoder = LabelEncoder()
        encoded[column] = encoder.fit_transform(encoded[column].astype(str))
        encoders[column] = encoder

    return encoded, encoders


@st.cache_data
def train_model(data: pd.DataFrame):
    encoded, encoders = encode_data(data)
    x = encoded[FEATURE_COLUMNS]
    y = encoded[TARGET_COLUMN]

    stratify = y if y.nunique() > 1 and y.value_counts().min() >= 2 else None
    x_train, x_test, y_train, y_test = train_test_split(
        x,
        y,
        test_size=0.20,
        random_state=42,
        stratify=stratify,
    )

    model = DecisionTreeClassifier(criterion="gini", random_state=42)
    model.fit(x_train, y_train)

    y_pred = model.predict(x_test)
    report = classification_report(
        y_test,
        y_pred,
        target_names=encoders[TARGET_COLUMN].classes_,
        output_dict=True,
        zero_division=0,
    )

    return {
        "model": model,
        "encoders": encoders,
        "x_train": x_train,
        "x_test": x_test,
        "y_test": y_test,
        "y_pred": y_pred,
        "accuracy": accuracy_score(y_test, y_pred),
        "confusion_matrix": confusion_matrix(y_test, y_pred),
        "classification_report": pd.DataFrame(report).transpose(),
    }


def encode_customer(customer: dict[str, object], encoders: dict[str, LabelEncoder]) -> pd.DataFrame:
    encoded_customer = customer.copy()

    for column in ["Gender", "Marital_Status", "Has_Driving_License", "Owns_House"]:
        encoded_customer[column] = int(
            encoders[column].transform([str(encoded_customer[column])])[0]
        )

    return pd.DataFrame([encoded_customer], columns=FEATURE_COLUMNS)


def prediction_label(prediction: int, encoder: LabelEncoder) -> str:
    return str(encoder.inverse_transform([prediction])[0])


st.set_page_config(
    page_title="Car Purchase Decision Tree",
    layout="wide",
)

st.title("Car Purchase Decision Dashboard")
st.caption("Decision Tree model output, evaluation metrics, and customer prediction form.")

with st.sidebar:
    st.header("Dataset")
    uploaded_file = st.file_uploader("Upload car_data.csv", type=["csv"])
    st.caption(
        "Required columns: Age, Gender, Annual_Income, Marital_Status, "
        "Has_Driving_License, Owns_House, Buy_Car."
    )

data, source_name = load_data(uploaded_file)
errors = validate_data(data)

if errors:
    st.error("Please fix the dataset before training the dashboard.")
    for error in errors:
        st.write(f"- {error}")
    st.stop()

results = train_model(data)
target_encoder = results["encoders"][TARGET_COLUMN]

metric_1, metric_2, metric_3, metric_4 = st.columns(4)
metric_1.metric("Dataset Source", source_name)
metric_2.metric("Rows", f"{len(data):,}")
metric_3.metric("Test Rows", f"{len(results['x_test']):,}")
metric_4.metric("Accuracy", f"{results['accuracy']:.2%}")

tab_predict, tab_metrics, tab_data, tab_tree = st.tabs(
    ["Prediction", "Model Metrics", "Dataset", "Decision Tree"]
)

with tab_predict:
    st.subheader("Customer Details")

    form_left, form_right = st.columns(2)
    with form_left:
        age = st.number_input("Age", min_value=18, max_value=100, value=30, step=1)
        gender = st.selectbox("Gender", results["encoders"]["Gender"].classes_)
        income = st.number_input(
            "Annual Income",
            min_value=0,
            value=500000,
            step=10000,
            format="%d",
        )

    with form_right:
        marital_status = st.selectbox(
            "Marital Status", results["encoders"]["Marital_Status"].classes_
        )
        driving_license = st.selectbox(
            "Has Driving License", results["encoders"]["Has_Driving_License"].classes_
        )
        owns_house = st.selectbox(
            "Owns House", results["encoders"]["Owns_House"].classes_
        )

    customer = {
        "Age": int(age),
        "Gender": gender,
        "Annual_Income": float(income),
        "Marital_Status": marital_status,
        "Has_Driving_License": driving_license,
        "Owns_House": owns_house,
    }
    encoded_customer = encode_customer(customer, results["encoders"])
    prediction = int(results["model"].predict(encoded_customer)[0])
    label = prediction_label(prediction, target_encoder)

    if hasattr(results["model"], "predict_proba"):
        probability = results["model"].predict_proba(encoded_customer)[0][prediction]
        st.metric("Prediction", label, delta=f"{probability:.2%} confidence")
    else:
        st.metric("Prediction", label)

    if label.lower() == "yes":
        st.success("Customer is likely to buy a car.")
    else:
        st.warning("Customer is not likely to buy a car.")

with tab_metrics:
    left, right = st.columns([1, 2])

    with left:
        st.subheader("Confusion Matrix")
        matrix = pd.DataFrame(
            results["confusion_matrix"],
            index=[f"Actual {label}" for label in target_encoder.classes_],
            columns=[f"Predicted {label}" for label in target_encoder.classes_],
        )
        st.dataframe(matrix, use_container_width=True)

    with right:
        st.subheader("Classification Report")
        st.dataframe(
            results["classification_report"].style.format("{:.2f}"),
            use_container_width=True,
        )

with tab_data:
    st.subheader("Training Data")
    st.dataframe(data, use_container_width=True)

    st.subheader("Target Distribution")
    target_counts = data[TARGET_COLUMN].astype(str).value_counts()
    st.bar_chart(target_counts)

with tab_tree:
    st.subheader("Decision Tree Visualization")
    fig, ax = plt.subplots(figsize=(16, 8))
    plot_tree(
        results["model"],
        feature_names=FEATURE_COLUMNS,
        class_names=target_encoder.classes_,
        filled=True,
        rounded=True,
        ax=ax,
    )
    st.pyplot(fig)

