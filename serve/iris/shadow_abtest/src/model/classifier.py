from typing import TypedDict
from pydantic import BaseModel
import onnxruntime as rt
import numpy as np

class PredictResponse(TypedDict):
    label: list[int]
    probs: list[float]

class PredictRequest(BaseModel):
    x_data: list[list[float]]

class Classifier:
    def __init__(self, model_path):
        self.classifier=None
        self.model_path = model_path
        self.load_model()

    def load_model(self):
        self.classifier = rt.InferenceSession(self.model_path)
        self.input_name = self.classifier.get_inputs()[0].name
        self.output_name = self.classifier.get_outputs()[0].name

    def predict(self, x_data) -> PredictResponse:
        np_data = np.array(x_data).astype(np.float32)
        prediction = self.classifier.run(None, {self.input_name: np_data})
        return {
            "label": prediction[0].tolist(),
            "probs": prediction[1]
        }

if __name__ == "__main__":
    classifier = Classifier("/Users/brandon/yjgit/mlflow/serve/iris/shadow_abtest/src/model/file/iris_logistic.onnx")
    data = [[5.1, 3.5, 1.4, 0.2], [6.0, 2.2, 4.5, 1.0], [5.1, 3.5, 1.4, 0.2]]
    print(classifier.predict(data))

