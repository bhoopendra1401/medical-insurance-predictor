import streamlit as st
import pickle
import numpy as np

# Load model
with open("insurance_model.pkl", "rb") as f:
    model = pickle.load(f)

st.set_page_config(page_title="Medical Insurance Cost Predictor")
st.title("Medical Insurance Cost Predictor")

age = st.number_input("Age", 1, 100, 25)

sex = st.selectbox("Sex", ["male", "female"])
sex = 1 if sex == "male" else 0

bmi = st.number_input("BMI", 10.0, 50.0, 25.0)

children = st.number_input("Number of Children", 0, 5, 0)

smoker = st.selectbox("Smoker", ["yes", "no"])
smoker = 1 if smoker == "yes" else 0

region = st.selectbox(
    "Region",
    ["northeast", "northwest", "southeast", "southwest"]
)

region_northeast = 1 if region == "northeast" else 0
region_northwest = 1 if region == "northwest" else 0
region_southeast = 1 if region == "southeast" else 0
region_southwest = 1 if region == "southwest" else 0

if st.button("Predict Insurance Cost"):
    data = np.array([[age, sex, bmi, children, smoker,
                      region_northeast, region_northwest,
                      region_southeast, region_southwest]])

    result = model.predict(data)
    st.success(f"Estimated Insurance Charge: ₹ {result[0]:,.2f}")
