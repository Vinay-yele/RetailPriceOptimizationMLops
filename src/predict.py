import pandas as pd
import joblib

model = joblib.load("models/model.pkl")

def predict_sales(features: dict) -> float:
    cols = [
        "unit_price", "freight_price", "lag_price", "comp_1", "ps1", "fp1",
        "comp_2", "ps2", "fp2", "comp_3", "ps3", "fp3"
    ]
    X = pd.DataFrame([features], columns=cols)
    return model.predict(X)[0]
