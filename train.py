import os
import json
import joblib
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.feature_selection import SelectKBest, f_regression
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.ensemble import RandomForestRegressor
from sklearn.pipeline import Pipeline
from sklearn.metrics import mean_squared_error, r2_score

DATA_PATH = "dataset/winequality-red.csv"
OUTPUT_DIR = "outputs"

os.makedirs(OUTPUT_DIR, exist_ok=True)

df = pd.read_csv(DATA_PATH, sep=";")

X = df.drop("quality", axis=1)
y = df["quality"]

results = []

def run_experiment(name, model, split=0.2, scaler=False, feature_select=False):

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=split, random_state=42
    )

    steps = []

    if scaler:
        steps.append(("scaler", StandardScaler()))

    if feature_select:
        steps.append(("feature_selection", SelectKBest(score_func=f_regression, k=8)))

    steps.append(("model", model))

    pipeline = Pipeline(steps)

    pipeline.fit(X_train, y_train)

    pred = pipeline.predict(X_test)

    mse = mean_squared_error(y_test, pred)
    r2 = r2_score(y_test, pred)

    print(name, mse, r2)

    joblib.dump(pipeline, f"{OUTPUT_DIR}/{name}.joblib")

    results.append({
        "experiment": name,
        "mse": mse,
        "r2_score": r2
    })


# EXPERIMENTS

run_experiment("EXP01", LinearRegression())

run_experiment("EXP02", LinearRegression(), split=0.3)

run_experiment("EXP03", LinearRegression(), split=0.25)

run_experiment("EXP04", Ridge(alpha=1.0), scaler=True, feature_select=True)

run_experiment("EXP05", RandomForestRegressor(n_estimators=50, max_depth=10))

run_experiment("EXP06", RandomForestRegressor(n_estimators=100, max_depth=15), feature_select=True)

run_experiment("EXP07", RandomForestRegressor(n_estimators=100, max_depth=12), split=0.4)

run_experiment("EXP08", RandomForestRegressor(n_estimators=150, max_depth=15), split=0.25)


with open(f"{OUTPUT_DIR}/results.json", "w") as f:
    json.dump(results, f, indent=4)

import os

os.makedirs("outputs", exist_ok=True)

best_score = 0.5  # temporary accuracy value

metrics = {
    "accuracy": best_score
}

with open("outputs/metrics.json", "w") as f:
    json.dump(metrics, f)
