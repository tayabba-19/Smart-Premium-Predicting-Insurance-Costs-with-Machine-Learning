import streamlit as st
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score

# -------------------------------
# Page Config
# -------------------------------
st.set_page_config(page_title="Smart Premium Predictor", layout="centered")
st.title("💰 Smart Premium Prediction App")
st.write("Predict insurance premium using Machine Learning")

# -------------------------------
# Upload CSV
# -------------------------------
uploaded_file = st.file_uploader("📂 Upload Insurance CSV File", type="csv")

if uploaded_file is not None:
    df = pd.read_csv(uploaded_file)

    # -------------------------------
    # Clean column names
    # -------------------------------
    df.columns = [col.strip().lower() for col in df.columns]

    # Rename premium column if needed
    if 'premium amount' in df.columns:
        df.rename(columns={'premium amount': 'premium'}, inplace=True)

    st.subheader("📊 Dataset Preview")
    st.dataframe(df.head())

    # -------------------------------
    # Check target column
    # -------------------------------
    if 'premium' not in df.columns:
        st.error("❌ CSV must contain 'premium' column")
    else:
        # -------------------------------
        # ONE-HOT ENCODING (IMPORTANT FIX)
        # -------------------------------
        df = pd.get_dummies(df, drop_first=True)

        # -------------------------------
        # Features & Target
        # -------------------------------
        X = df.drop('premium', axis=1)
        y = df['premium']

        # -------------------------------
        # Train-Test Split
        # -------------------------------
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42
        )

        # -------------------------------
        # Train Model
        # -------------------------------
        model = LinearRegression()
        model.fit(X_train, y_train)

        # -------------------------------
        # Model Evaluation
        # -------------------------------
        y_pred = model.predict(X_test)
        score = r2_score(y_test, y_pred)
        st.success(f"✅ Model R² Score: {score:.2f}")

        # -------------------------------
        # User Input (NUMERIC ONLY)
        # -------------------------------
        st.subheader("🧾 Enter Customer Details")

        input_data = {}
        for col in X.columns:
            input_data[col] = st.number_input(
                f"{col}",
                value=float(X[col].mean())
            )

        input_df = pd.DataFrame([input_data])

        # -------------------------------
        # Prediction
        # -------------------------------
        if st.button("🔮 Predict Premium"):
            prediction = model.predict(input_df)
            st.success(
                f"💸 Predicted Insurance Premium: ₹ {prediction[0]:,.2f}"
            )

# -------------------------------
# Footer
# -------------------------------
st.markdown("---")
st.markdown("📌 *Project: Smart Premium – Predicting Insurance Costs Using Machine Learning*")







