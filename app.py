import streamlit as st
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split

st.set_page_config(page_title="Medical Insurance Cost Predictor")

st.title("Medical Insurance Cost Predictor")

@st.cache_data
def load_data():
    df = pd.read_csv("Health_Insurance.csv")
    return df

df = load_data()

# Encode categorical columns
df["sex"] = df["sex"].map({"male": 1, "female": 0})
df["smoker"] = df["smoker"].map({"yes": 1, "no": 0})
df = pd.get_dummies(df, columns=["region"], drop_first=True)

X = df.drop("charges", axis=1)
y = df["charges"]

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

# -------- UI --------
age = st.number_input("Age", 18, 100, 25)
sex = st.selectbox("Sex", ["male", "female"])
bmi = st.number_input("BMI", 10.0, 60.0, 25.0)
children = st.number_input("Number of Children", 0, 5, 0)
smoker = st.selectbox("Smoker", ["yes", "no"])
region = st.selectbox("Region", ["southeast", "southwest", "northeast", "northwest"])

sex = 1 if sex == "male" else 0
smoker = 1 if smoker == "yes" else 0

region_data = {
    "region_northwest": 1 if region == "northwest" else 0,
    "region_southeast": 1 if region == "southeast" else 0,
    "region_southwest": 1 if region == "southwest" else 0,
}

input_data = {
    "age": age,
    "sex": sex,
    "bmi": bmi,
    "children": children,
    "smoker": smoker,
    **region_data
}

input_df = pd.DataFrame([input_data])

if st.button("Predict Insurance Cost"):
    prediction = model.predict(input_df)[0]
    st.success(f"Estimated Insurance Cost: ${prediction:,.2f}")



