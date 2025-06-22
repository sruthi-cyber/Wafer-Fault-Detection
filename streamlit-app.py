import streamlit as st
import pandas as pd
import numpy as np
import joblib
from preprocessing import preprocess

st.title("🔬 Wafer Fault Detection")

uploaded_file = st.file_uploader("Upload Wafer CSV File", type=["csv"])

if uploaded_file:
    df = pd.read_csv(uploaded_file)
    st.write("📊 Input Data:", df.head())

    try:
        model = joblib.load("artifacts/model.pkl")
        X, _ = preprocess(df)
        predictions = model.predict(X)
        df['prediction'] = predictions
        st.success("✅ Predictions completed!")
        st.write(df[['prediction']])
    except Exception as e:
        st.error(f"❌ Error: {e}")