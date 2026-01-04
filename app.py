import streamlit as st
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor

# ---------------------------
# Page config
# ---------------------------
st.set_page_config(page_title="Medical Insurance Cost Predictor")

st.title("Medical Insurance Cost Predictor")

# ---------------------------
# Load Data
# ---------------------------
@st.cache_data
def load_data():
    df = pd.read_csv("Health_Insurance.csv")
    return df

df = load_data()

# ---------------------------
# Preprocessing
# ---------------------------
df["sex"] = df["sex"].map({"male": 1, "female": 0})
df["smoker"] = df["smoker"].map({"yes": 1, "no": 0})
df["region"] = df["region"].astype("category").cat.codes

X = df.drop("charges", axis=1)
y = df["charges"]

# ---------------------------
# Train Model (Cloud-safe)
# ---------------------------
@st.cache_resource
def train_model():
    model = RandomForestRegressor(
        n_estimators=200,
        max_depth=5,
        random_state=42
    )
    model.fit(X, y)
    return model

model = train_model()

# ---------------------------
# User Inputs
# ---------------------------
age = st.number_input("Age", 18, 100, 25)
sex = st.selectbox("Sex", ["male", "female"])
bmi = st.number_input("BMI", 10.0, 60.0, 25.0)
children = st.number_input("Number of Children", 0, 5, 0)
smoker = st.selectbox("Smoker", ["yes", "no"])
region = st.selectbox("Region", ["southwest", "southeast", "northwest", "northeast"])

sex = 1 if sex == "male" else 0
smoker = 1 if smoker == "yes" else 0
region = ["southwest", "southeast", "northwest", "northeast"].index(region)

# ---------------------------
# Prediction
# ---------------------------
if st.button("Predict Insurance Cost"):
    input_data = np.array([[age, sex, bmi, children, smoker, region]])
    prediction = model.predict(input_data)
    st.success(f"Estimated Medical Insurance Cost: ₹ {prediction[0]:,.2f}")



