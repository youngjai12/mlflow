import os

from fastapi import FastAPI
import onnxruntime as rt
import sys
from .model.classifier import Classifier
from .model.classifier import PredictRequest

app = FastAPI(
    title="Model-In-Image-Pattern Example"
)

MODEL_PATH = os.environ.get("MODEL_FILEPATH")
PROJ_DIR = os.environ.get("PROJECT_DIR")
classifier = Classifier(f"{PROJ_DIR}/src/model/file/{MODEL_PATH}")


@app.post("/predict")
def predict(request_data: PredictRequest):
    return classifier.predict(request_data.x_data)

