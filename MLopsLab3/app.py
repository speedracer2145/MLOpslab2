from fastapi import FastAPI
from fastapi.responses import RedirectResponse
from pydantic import BaseModel
import numpy as np
import joblib

app = FastAPI()

model = joblib.load("model.pkl")

class WineInput(BaseModel):
    features: list[float]

@app.get("/", include_in_schema=False)
def root():
    return RedirectResponse(url="/docs")

@app.post("/predict")
def predict(data: WineInput):

    X = np.array(data.features).reshape(1,-1)

    prediction = model.predict(X)[0]

    return {
        "name": "Alok P",
        "roll_no": "2022BCS0014",
        "wine_quality": int(round(prediction))
    }

