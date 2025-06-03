import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
import joblib
import mlflow
import os
import numpy as np

df = pd.read_csv("data/retail_price.csv")

features = [
    "unit_price", "freight_price", "lag_price", "comp_1", "ps1", "fp1",
    "comp_2", "ps2", "fp2", "comp_3", "ps3", "fp3"
]
target = "total_price"

X = df[features]
y = df[target]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

model = RandomForestRegressor(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

y_pred = model.predict(X_test)
rmse = np.sqrt(mean_squared_error(y_test, y_pred))

mlflow.start_run()
mlflow.log_param("model", "RandomForest")
mlflow.log_metric("rmse", rmse)
mlflow.sklearn.log_model(model, "model")
mlflow.end_run()

os.makedirs("models", exist_ok=True)
joblib.dump(model, "models/model.pkl")

print(f"✅ Model trained successfully. RMSE: {rmse:.2f}")
