import streamlit as st
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score

st.set_page_config(page_title="Smart Premium Predictor", layout="centered")
st.title("💰 Smart Premium Prediction App")

uploaded_file = st.file_uploader("📂 Upload Insurance CSV File", type="csv")

if uploaded_file is not None:
    df = pd.read_csv(uploaded_file)

    # Clean column names
    df.columns = [col.strip().lower() for col in df.columns]

    # Drop ID column (IMPORTANT)
    if 'id' in df.columns:
        df.drop('id', axis=1, inplace=True)

    # Rename target column
    if 'premium amount' in df.columns:
        df.rename(columns={'premium amount': 'premium'}, inplace=True)

    st.subheader("📊 Dataset Preview")
    st.dataframe(df.head())

    if 'premium' not in df.columns:
        st.error("❌ CSV must contain 'premium' column")
    else:
        # Handle missing values
        df = df.fillna(df.mean(numeric_only=True))

        # One-Hot Encoding
        df = pd.get_dummies(df, drop_first=True)

        X = df.drop('premium', axis=1)
        y = df['premium']

        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42
        )

        model = LinearRegression()
        model.fit(X_train, y_train)

        y_pred = model.predict(X_test)
        score = r2_score(y_test, y_pred)
        st.success(f"✅ Model R² Score: {score:.2f}")

        # -------- USER INPUT --------
        st.subheader("🧾 Enter Customer Details")

        input_data = {}
        for col in X.columns:
            input_data[col] = st.number_input(col, value=float(X[col].mean()))

        input_df = pd.DataFrame([input_data])

        # 🔥 ALIGN INPUT WITH TRAINING FEATURES
        input_df = input_df.reindex(columns=X.columns, fill_value=0)

        if st.button("🔮 Predict Premium"):
            prediction = model.predict(input_df)
            st.success(f"💸 Predicted Insurance Premium: ₹ {prediction[0]:,.2f}")







