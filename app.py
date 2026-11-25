
import streamlit as st
import pandas as pd
import numpy as np
import joblib
from pathlib import Path

# -------------------------------------------------
# 1. PAGE CONFIGURATION
# -------------------------------------------------

st.set_page_config(
    page_title="Employee Attrition Prediction",
    page_icon="👨‍💼",
    layout="wide"
)

# -------------------------------------------------
# 2. LOAD MODEL AND DATASET
# -------------------------------------------------

BASE_DIR = Path(__file__).resolve().parent

MODEL_PATH = BASE_DIR / "models" / "best_attrition_hpt_pipeline.pkl"
DATA_PATH = BASE_DIR / "data" / "Employee Attrition.csv"

@st.cache_resource
def load_model():
    return joblib.load(MODEL_PATH)

@st.cache_data
def load_dataset():
    return pd.read_csv(DATA_PATH)

try:
    model = load_model()
    df = load_dataset()

except Exception as e:
    st.error(f"Error loading model or dataset: {e}")
    st.stop()

# -------------------------------------------------
# 3. FEATURE INFORMATION
# -------------------------------------------------

TARGET = "Attrition"
ID_COLUMN = "Employee ID"

feature_columns = [
    col for col in df.columns
    if col not in [TARGET, ID_COLUMN]
]

X_reference = df[feature_columns]

numeric_columns = X_reference.select_dtypes(
    include=np.number
).columns.tolist()

categorical_columns = [
    col for col in feature_columns
    if col not in numeric_columns
]

# -------------------------------------------------
# 4. APPLICATION TITLE
# -------------------------------------------------

st.title("👨‍💼 Employee Attrition Prediction System")

st.write(
    "Predict whether an employee is likely to stay "
    "or leave the organization using Machine Learning."
)

st.divider()

# -------------------------------------------------
# 5. INPUT FORM
# -------------------------------------------------

st.subheader("Enter Employee Details")

input_data = {}

with st.form("employee_prediction_form"):

    st.markdown("### Employee Information")

    col1, col2 = st.columns(2)

    # Numeric inputs
    for i, column in enumerate(numeric_columns):

        selected_column = col1 if i % 2 == 0 else col2

        with selected_column:

            series = X_reference[column].dropna()

            if series.empty:
                st.error(f"No available values for {column}")
                st.stop()

            minimum = float(series.min())
            maximum = float(series.max())
            median = float(series.median())

            if pd.api.types.is_integer_dtype(series.dtype):

                input_data[column] = st.number_input(
                    column,
                    min_value=int(minimum),
                    max_value=int(maximum),
                    value=int(median),
                    step=1
                )

            else:

                input_data[column] = st.number_input(
                    column,
                    min_value=minimum,
                    max_value=maximum,
                    value=median
                )

    st.divider()

    st.markdown("### Job and Company Information")

    col3, col4 = st.columns(2)

    # Categorical inputs
    for i, column in enumerate(categorical_columns):

        selected_column = col3 if i % 2 == 0 else col4

        with selected_column:

            options = sorted(
                X_reference[column].dropna().unique().tolist(),
                key=str
            )

            if len(options) == 0:
                st.error(f"No available categories for {column}")
                st.stop()

            input_data[column] = st.selectbox(
                column,
                options=options
            )

    st.divider()

    submitted = st.form_submit_button(
        "Predict Employee Attrition",
        use_container_width=True
    )

# -------------------------------------------------
# 6. PREDICTION
# -------------------------------------------------

if submitted:

    input_df = pd.DataFrame(
        [input_data],
        columns=feature_columns
    )

    try:

        prediction = model.predict(input_df)[0]

        st.subheader("Prediction Result")

        if prediction == "Left":

            st.error("⚠️ Prediction: Employee is likely to LEAVE")

        else:

            st.success("✅ Prediction: Employee is likely to STAY")

        # -------------------------------------------------
        # 7. PREDICTION PROBABILITY
        # -------------------------------------------------

        if hasattr(model, "predict_proba"):

            probabilities = model.predict_proba(input_df)[0]

            classes = model.classes_

            probability_df = pd.DataFrame({
                "Attrition Status": classes,
                "Probability": probabilities
            })

            st.subheader("Prediction Probability")

            left_probability = probabilities[
                list(classes).index("Left")
            ]

            stayed_probability = probabilities[
                list(classes).index("Stayed")
            ]

            col1, col2 = st.columns(2)

            with col1:
                st.metric(
                    "Probability of Leaving",
                    f"{left_probability * 100:.2f}%"
                )

            with col2:
                st.metric(
                    "Probability of Staying",
                    f"{stayed_probability * 100:.2f}%"
                )

            st.bar_chart(
                probability_df.set_index("Attrition Status")
            )

        st.caption(
            "Note: This prediction is based on the trained "
            "machine learning model and is not a certainty."
        )

    except Exception as e:

        st.error(f"Prediction error: {e}")
