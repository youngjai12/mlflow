import mlflow
from mlflow.store.artifact.models_artifact_repo import ModelsArtifactRepository
import onnxruntime as rt
import numpy as np


class ModelClient:
    def __init__(self, mlflow_url="http://127.0.0.1:5000"):
        self.mlflow_host = mlflow_url
        mlflow.set_tracking_uri(self.mlflow_host)
        self.model_dir_template = "models:/{model_name}/{model_version}"


    def download_model_from_mlflow_server(self, model_name, model_version, local_path):
        model_dir = self.model_dir_template.format(model_name=model_name, model_version=model_version)
        print(f"download model path: {model_dir}")
        try:
            return_path = ModelsArtifactRepository(model_dir).download_artifacts(artifact_path="", dst_path=local_path)
            return return_path
        except Exception as e:
            print(f"error occured : {e}")
            return None

    def download_model_from_run_id(self, run_id, model_name, local_path):
        # runs:/{run_id} 까지가 ui 상 aritfacts 탭에 들어간것과 같음.
        # onnx 학습시에 그냥 artifact바로 아래 떨어지도록 했음.
        mlflow.artifacts.download_artifacts(artifact_uri=f"runs:/{run_id}/{model_name}", dst_path=local_path)



class OnnxModelLoader:
    def __init__(self, onnx_model_path):
        self.classifier=None
        self.input_name=None
        self.output_name=None
        self.load_model(onnx_model_path)

    def load_model(self, onnx_model_path):
        self.classifier = rt.InferenceSession(onnx_model_path)
        self.input_name = self.classifier.get_inputs()[0].name
        self.output_name = self.classifier.get_outputs()[0].name

    def predict(self, x_data):
        np_data = np.array(x_data).astype(np.float32)
        prediction = self.classifier.run(None, {self.input_name: np_data})

        return prediction

def download_model(model_client: ModelClient, local_dir):
    model_client.download_model_from_mlflow_server(
        model_name="sklearn-logistic-registered-model",
        model_version="2",
        local_path=f"{local_dir}/2"
    )

def download_model_by_run_id(model_client: ModelClient):
    model_client.download_model_from_run_id(
        run_id="63868c35b27841dd86a6138016bf3ed1",
        model_name="iris_logistic.onnx",
        local_path="./model_dir/iris_logistic"
    )



if __name__ == "__main__":
    model_client = ModelClient()
    # download_model(model_client, "./model_dir/iris_logistic")
    #download_model_by_run_id(model_client)
    onnx_model = OnnxModelLoader("./model_dir/iris_logistic/iris_logistic.onnx")
    x_data = [[5.1, 3.5, 1.4, 0.2], [6.0, 2.2, 4.5, 1.0], [5.1, 3.5, 1.4, 0.2]]
    print(onnx_model.predict(x_data))