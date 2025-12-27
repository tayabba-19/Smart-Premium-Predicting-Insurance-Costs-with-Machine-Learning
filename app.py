import streamlit as st
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import LabelEncoder

st.set_page_config(page_title="Smart Premium Predictor", layout="centered")
st.title("💰 Smart Premium Prediction App")
st.write("Predict insurance premium using Machine Learning")

# -------------------------------
# Upload CSV
# -------------------------------
uploaded_file = st.file_uploader("Upload CSV file", type="csv")
if uploaded_file is not None:
    df = pd.read_csv(uploaded_file)
    st.subheader("📊 Dataset Preview")
    st.dataframe(df.head())

    # Encode categorical columns
    le = LabelEncoder()
    categorical_cols = df.select_dtypes(include=["object"]).columns
    for col in categorical_cols:
        df[col] = le.fit_transform(df[col])

    # Features & Target
    X = df.drop("premium", axis=1)
    y = df["premium"]

    # Train model
    model = LinearRegression()
    model.fit(X, y)

    # -------------------------------
    # User Inputs
    # -------------------------------
    st.subheader("🧾 Enter Customer Details")
    inputs = {}
    for col in X.columns:
        if X[col].dtype == "int64" or X[col].dtype == "float64":
            inputs[col] = st.number_input(f"{col}", value=int(X[col].mean()))
        else:
            options = list(df[col].unique())
            inputs[col] = st.selectbox(f"{col}", options)

    input_df = pd.DataFrame([list(inputs.values())], columns=X.columns)

    # -------------------------------
    # Prediction
    # -------------------------------
    if st.button("🔮 Predict Premium"):
        prediction = model.predict(input_df)
        st.success(f"💸 Predicted Insurance Premium: ₹ {prediction[0]:,.2f}")

# -------------------------------
# Footer
# -------------------------------
st.markdown("---")
st.markdown("📌 *Project: Smart Premium – Predicting Insurance Costs with Machine Learning*")



