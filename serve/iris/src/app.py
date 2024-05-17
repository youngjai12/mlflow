import os

from fastapi import FastAPI
import onnxruntime as rt
import sys
sys.path.append("/home/talkaa/src")
from model.classifier import SvcClassifier
from model.dto import PredictRequest

app = FastAPI(
    title="Model-In-Image-Pattern Example"
)

MODEL_PATH = os.environ.get("MODEL_FILEPATH")
svc_binary_classifier = SvcClassifier(MODEL_PATH)

@app.post("/predict")
def predict(request_data: PredictRequest):
    labels, probs = svc_binary_classifier.predict(request_data.data)
    return {"labels": labels, "probs": probs}
