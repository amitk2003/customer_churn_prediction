from fastapi import FastAPI
import joblib
import pandas as pd

app = FastAPI(title="Churn Prediction API")

# Load model and metadata
artifact = joblib.load("../model/final_churn_model.pkl")
pipeline = artifact["model"]
threshold = artifact["threshold"]
feature_columns = artifact["feature_columns"]  # MUST be saved earlier

@app.post("/predict_batch")
def predict_batch(customers: list[dict]):

    # Convert input JSON into DataFrame
    df = pd.DataFrame(customers)

    # IMPORTANT: Reindex to match training columns
    df = df.reindex(columns=feature_columns, fill_value=0)

    # Predict churn
    probs = pipeline.predict_proba(df)[:, 1]
    preds = (probs >= threshold).astype(int)

    df["churn_probability"] = probs
    df["prediction"] = preds

    return df.to_dict(orient="records")
