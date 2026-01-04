
import streamlit as st
import numpy as np
import joblib

# -----------------------------
# Load trained model (JOBLIB)
# -----------------------------
model = joblib.load("insurance_model.joblib")

# -----------------------------
# Page config
# -----------------------------
st.set_page_config(
    page_title="Medical Insurance Cost Predictor",
    page_icon="💊",
    layout="centered"
)

st.title("Medical Insurance Cost Predictor")
st.write("Enter the details below to predict medical insurance cost.")

# -----------------------------
# User Inputs
# -----------------------------
age = st.number_input("Age", min_value=1, max_value=100, value=25)

sex = st.selectbox("Sex", ["male", "female"])
sex = 1 if sex == "male" else 0

bmi = st.number_input("BMI", min_value=10.0, max_value=50.0, value=25.0)

children = st.number_input("Number of Children", min_value=0, max_value=5, value=0)

smoker = st.selectbox("Smoker", ["yes", "no"])
smoker = 1 if smoker == "yes" else 0

region = st.selectbox(
    "Region",
    ["northeast", "northwest", "southeast", "southwest"]
)

# One-hot encoding for region
region_northeast = 1 if region == "northeast" else 0
region_northwest = 1 if region == "northwest" else 0
region_southeast = 1 if region == "southeast" else 0
region_southwest = 1 if region == "southwest" else 0

# -----------------------------
# Prediction
# -----------------------------
if st.button("Predict Insurance Cost"):
    input_data = np.array([[ 
        age,
        sex,
        bmi,
        children,
        smoker,
        region_northeast,
        region_northwest,
        region_southeast,
        region_southwest
    ]])

    prediction = model.predict(input_data)

    st.success(f"Estimated Insurance Cost: ₹ {prediction[0]:,.2f}")

