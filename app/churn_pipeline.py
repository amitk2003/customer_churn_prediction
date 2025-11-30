import pandas as pd
import numpy as np

class ChurnPipeline:
    def __init__(self, model, X, cat_features):
        self.model = model
        self.columns = X.columns.tolist()
        self.cat_features = X.columns[cat_features].tolist()
        self.numeric_features = X.select_dtypes(include=["int64", "float64"]).columns.tolist()
        self.medians = X[self.numeric_features].median()

    def preprocess(self, input_dict):
        # Create empty row with correct schema
        df = pd.DataFrame([{col: None for col in self.columns}])

        # Update with user inputs
        for k, v in input_dict.items():
            if k in df.columns:
                df.at[0, k] = v

        # Fill missing values
        df[self.cat_features] = df[self.cat_features].fillna("Unknown")
        df[self.numeric_features] = df[self.numeric_features].fillna(self.medians)

        return df

    def predict(self, data):
        df = self.preprocess(data)
        prediction = self.model.predict(df)[0]
        
        if hasattr(self.model, "predict_proba"):
            probability = self.model.predict_proba(df)[0]
        else:
            probability = None

        return prediction, probability
