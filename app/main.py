import streamlit as st
import joblib
import pandas as pd

from churn_pipeline import ChurnPipeline  # <-- IMPORTANT

# pipeline = joblib.load("./final_churn_pipeline.pkl")
pipeline = joblib.load("../notebooks/final_churn_pipeline.pkl")


print("Pipeline loaded successfully!")

# Load the final saved pipeline
# pipeline = joblib.load("final_churn_pipeline.pkl")

st.title("📌 Customer Churn Risk Predictor")
st.write("Predict whether a customer is **Low**, **Medium**, or **High** churn risk.")

# --- FORM UI ---
inputs = {}

st.subheader("Enter Customer Details:")
for feature in pipeline.columns:
    # Categorical Feature Input
    if feature in pipeline.cat_features:
        inputs[feature] = st.text_input(f"{feature} (text)", "")
    
    # Numeric Feature Input
    elif feature in pipeline.numeric_features:
        inputs[feature] = st.number_input(f"{feature} (number)", step=1.0)

    else:
        # General fallback textbox
        inputs[feature] = st.text_input(f"{feature}", "")

# --- Prediction Button ---
if st.button("Predict Churn Risk"):
    prediction, probabilities = pipeline.predict(inputs)

    st.success(f"🧠 Predicted Risk Level: **{prediction}**")

    st.write("📊 Risk Probability Breakdown:")
    labels = ["Low", "Medium", "High"]

    prob_df = pd.DataFrame({
        "Risk Level": labels,
        "Probability": probabilities
    })

    st.table(prob_df)
