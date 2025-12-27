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

    # Clean column names
    df.columns = [col.strip().lower() for col in df.columns]

    st.subheader("📊 Dataset Preview")
    st.dataframe(df.head())

    # -------------------------------
    # Encode categorical columns
    # -------------------------------
    le_dict = {}
    categorical_cols = df.select_dtypes(include=['object']).columns
    for col in categorical_cols:
        le = LabelEncoder()
        df[col] = le.fit_transform(df[col].astype(str))
        le_dict[col] = le  # store encoder for user input

    # -------------------------------
    # Features & Target
    # -------------------------------
    if 'premium' not in df.columns:
        st.error("❌ CSV must contain 'premium' column")
    else:
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
        inputs = {}
        for col in X.columns:
            if X[col].dtype in ['int64', 'float64']:
                inputs[col] = st.number_input(f"{col}", value=int(X[col].mean()))
            else:
                options = df[col].unique()
                choice = st.selectbox(f"{col}", options)
                # encode user input using same LabelEncoder
                inputs[col] = le_dict[col].transform([choice])[0]

        input_df = pd.DataFrame([list(inputs.values())], columns=X.columns)

        # -------------------------------
        # Prediction
        # -------------------------------
        if st.button("🔮 Predict Premium"):
            try:
                prediction = model.predict(input_df)
                st.success(f"💸 Predicted Insurance Premium: ₹ {prediction[0]:,.2f}")
            except Exception as e:
                st.error(f"❌ Prediction failed: {e}")

# -------------------------------
# Footer
# -------------------------------
st.markdown("---")
st.markdown("📌 *Project: Smart Premium – Predicting Insurance Costs with Machine Learning*")





