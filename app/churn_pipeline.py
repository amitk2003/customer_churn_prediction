import pandas as pd
import numpy as np

class ChurnPipeline:
    def __init__(self, model, X, cat_features):
        self.model = model
        self.columns = X.columns.tolist()
        self.cat_features = X.columns[cat_features].tolist()
        self.numeric_features = X.select_dtypes(include=['int64','float64']).columns.tolist()
        self.medians = X[self.numeric_features].median()

    def preprocess(self, input_dict):
        df = pd.DataFrame({col: pd.Series([], dtype="object") for col in self.columns})
        df.loc[0] = None

        for k, v in input_dict.items():
            if k in df.columns:
                df.at[0, k] = v

        df[self.cat_feature_names] = df[self.cat_feature_names].fillna("Unknown")
        df[self.numeric_features] = df[self.numeric_features].fillna(self.medians)

        return df

    def predict(self, data):
        df = self.preprocess(data)
        prediction = self.model.predict(df)[0]
        probability = self.model.predict_proba(df)[0]
        return prediction, probability
# import os
# print(os.getcwd())
# print(os.listdir("app"))
