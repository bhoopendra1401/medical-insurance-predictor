import streamlit as st
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor

# -----------------------------
# Load dataset
# -----------------------------
@st.cache_data
def load_data():
    df = pd.read_csv("Health_Insurance.csv")
    return df

df = load_data()

# -----------------------------
# Preprocessing
# -----------------------------
df["sex"] = df["sex"].map({"male": 1, "female": 0})
df["smoker"] = df["smoker"].map({"yes": 1, "no": 0})

df = pd.get_dummies(df, columns=["region"], drop_first=False)

X = df.drop("charges", axis=1)
y = df["charges"]

# -----------------------------
# Train model (FAST)
# -----------------------------
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

# -----------------------------
# UI
# -----------------------------
st.set_page_config(page_title="Medical Insurance Cost Predictor", page_icon="💊")
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

# -----------------------------
# Prediction
# -----------------------------
if st.button("Predict Insurance Cost"):
    input_data = np.array([[
        age, sex, bmi, children, smoker,
        region_northeast,
        region_northwest,
        region_southeast,
        region_southwest
    ]])

    prediction = model.predict(input_data)
    st.success(f"Estimated Insurance Cost: ₹ {prediction[0]:,.2f}")


