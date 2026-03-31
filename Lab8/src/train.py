import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
import json
import os
import joblib

# MLflow
import mlflow
import mlflow.sklearn

# ----------------------------
# MLflow setup
# ----------------------------
mlflow.set_tracking_uri("http://localhost:5000")
mlflow.set_experiment("lab9-2022bcs0014")

# ----------------------------
# Load data
# ----------------------------
df = pd.read_csv("data/housing.csv")

X = df.drop("MedHouseVal", axis=1)
y = df["MedHouseVal"]

# ----------------------------
# Train-test split
# ----------------------------
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# ----------------------------
# MLflow run
# ----------------------------
with mlflow.start_run():

    # Model
    model = LinearRegression()
    model.fit(X_train, y_train)

    # Predictions
    pred = model.predict(X_test)

    # Metrics
    mse = mean_squared_error(y_test, pred)
    rmse = mse ** 0.5
    r2 = r2_score(y_test, pred)

    # ----------------------------
    # Save outputs (Lab8 requirement)
    # ----------------------------
    os.makedirs("Lab8/outputs", exist_ok=True)

    metrics = {
        "rmse": float(rmse),
        "r2": float(r2),
        "dataset_size": len(df)
    }

    with open("Lab8/outputs/metrics.json", "w") as f:
        json.dump(metrics, f)

    joblib.dump(model, "Lab8/outputs/model.pkl")

    # ----------------------------
    # MLflow logging (Lab9)
    # ----------------------------
    mlflow.log_metric("rmse", rmse)
    mlflow.log_metric("r2", r2)

    mlflow.sklearn.log_model(model, "model")

# Print results
print(metrics)
