from fastapi import FastAPI
from pydantic import BaseModel
import joblib

app = FastAPI(
    title="MLOps Model API",
    description="Machine Learning model deployment API",
    version="1.0.0"
)

# Load trained model
model = joblib.load("model/model.pkl")


class PredictionRequest(BaseModel):
    sepal_length: float
    sepal_width: float
    petal_length: float
    petal_width: float


@app.get("/")
def home():
    return {"message": "MLOps Model API is running successfully"}


@app.post("/predict")
def predict(data: PredictionRequest):

    features = [[
        data.sepal_length,
        data.sepal_width,
        data.petal_length,
        data.petal_width
    ]]

    prediction = model.predict(features)

    return {
        "prediction": int(prediction[0])
    }