import os
import streamlit as st
import joblib
import pandas as pd
from churn_pipeline import ChurnPipeline

# Auto-detect correct path
BASE_DIR = os.path.dirname(os.path.abspath(__file__))  # path of /app/
MODEL_PATH = os.path.join(BASE_DIR, "final_churn_pipeline.pkl")

pipeline = joblib.load(MODEL_PATH)


st.title("📌 Customer Churn Risk Predictor")
st.write("Predict whether a customer is **Low**, **Medium**, or **High** churn risk.")

inputs = {}

st.subheader("Enter Customer Details:")

for feature in pipeline.columns:

    # Categorical input
    if feature in pipeline.cat_features:
        inputs[feature] = st.selectbox(feature, ["Unknown", "Yes", "No", "Male", "Female"])  # UPDATE options

    # Numeric input
    elif feature in pipeline.numeric_features:
        inputs[feature] = st.number_input(feature, value=0.0)

    else:
        inputs[feature] = st.text_input(feature)

# Prediction
if st.button("Predict"):
    prediction, prob = pipeline.predict(inputs)

    st.success(f"Result: **{prediction}**")

    if prob is not None:
        st.write("Prediction Confidence:")
        st.table(pd.DataFrame({"Class": pipeline.model.classes_, "Probability": prob}))
