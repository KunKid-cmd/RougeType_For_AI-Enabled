
from fastapi import FastAPI
from pydantic import BaseModel
import joblib
import numpy as np

app = FastAPI()
model = joblib.load("model.pkl")
scaler = joblib.load("scaler.pkl")

class InputData(BaseModel):
    wpm: float
    combo_length: int
    mistake_count: int
    recent_accuracy: float
    wave_number: int

@app.post("/predict")
def predict(data: InputData):
    features = np.array([[data.wpm, data.combo_length, data.mistake_count, data.recent_accuracy, data.wave_number]])
    features_scaled = scaler.transform(features)
    prediction = model.predict(features_scaled)[0]
    return {"predicted_difficulty": int(prediction)}
