import streamlit as st
import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import LabelEncoder

# -------------------------------
# Page config
# -------------------------------
st.set_page_config(page_title="Insurance Premium Predictor", layout="centered")
st.title("💰 Smart Premium Prediction App")
st.write("Predict insurance premium using Machine Learning")

# -------------------------------
# Load CSV safely from Google Drive
# -------------------------------
@st.cache_data
def load_data():
    url = "https://drive.google.com/uc?export=download&id=1KITP_B3j98cEk8y0dR0K1azc0A_oq4"
    df = pd.read_csv(url)
    # Check for missing columns or errors
    required_cols = ["age", "gender", "marital_status", "annual_income", "dependents", "premium"]
    missing = [c for c in required_cols if c not in df.columns]
    if missing:
        st.error(f"❌ Missing columns in CSV: {missing}")
    return df

df = load_data()

st.subheader("📊 Dataset Preview")
st.dataframe(df.head())

# -------------------------------
# Encode categorical columns safely
# -------------------------------
le = LabelEncoder()
categorical_cols = df.select_dtypes(include=["object"]).columns
for col in categorical_cols:
    df[col] = le.fit_transform(df[col])

# -------------------------------
# Features & Target
# -------------------------------
X = df.drop("premium", axis=1)
y = df["premium"]

# -------------------------------
# Train model
# -------------------------------
model = LinearRegression()
model.fit(X, y)

# -------------------------------
# User Inputs
# -------------------------------
st.subheader("🧾 Enter Customer Details")

age = st.number_input("Age", min_value=18, max_value=100, value=25)
gender = st.selectbox("Gender", ["male", "female"])
marital_status = st.selectbox("Marital Status", ["single", "married"])
annual_income = st.number_input("Annual Income", min_value=10000, value=300000)
dependents = st.number_input("Number of Dependents", min_value=0, value=1)

# Encode inputs manually (safe)
gender_encoded = 1 if gender.lower() == "male" else 0
marital_encoded = 1 if marital_status.lower() == "married" else 0

# Build input DataFrame
input_data = pd.DataFrame([[age, gender_encoded, marital_encoded, annual_income, dependents]],
                          columns=X.columns)

# -------------------------------
# Prediction
# -------------------------------
if st.button("🔮 Predict Premium"):
    try:
        prediction = model.predict(input_data)
        st.success(f"💸 Predicted Insurance Premium: ₹ {prediction[0]:,.2f}")
    except Exception as e:
        st.error(f"❌ Prediction failed: {e}")

# -------------------------------
# Footer
# -------------------------------
st.markdown("---")
st.markdown("📌 *Project: Smart Premium – Predicting Insurance Costs with Machine Learning*")


