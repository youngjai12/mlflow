import mlflow
import numpy as np
from mlflow import MlflowClient
from mlflow.store.artifact.models_artifact_repo import ModelsArtifactRepository
from sklearn.metrics import mean_squared_error
from windpower.data.read_data import WindPowerDataset
from windpower.dl_model import record_mse

mlflow.set_tracking_uri("http://127.0.0.1:5000")

MODEL_DIR_TEMPLATE = "models:/{model_name}/{model_version}"
def load_tf_model(model_name, model_version):
    model_dir = MODEL_DIR_TEMPLATE.format(model_name=model_name, model_version=model_version)
    return mlflow.tensorflow.load_model(model_dir)

def download_model_from_mlflow(model_name, model_version, local_path):
    model_dir = MODEL_DIR_TEMPLATE.format(model_name=model_name, model_version=model_version)
    ModelsArtifactRepository(model_dir).download_artifacts(artifact_path="", dst_path=local_path)

def model_infer(tf_model, x_test, y_test):

    eval_loss, eval_acc = tf_model.evaluate(x_test, y_test)
    preds = tf_model.predict(x_test)
    return preds, eval_loss, eval_acc

def mlflow_load_predict():
    loaded_model = load_tf_model(model_name="mlp_2018-03-01", model_version="4")

    print(loaded_model)
    dataset = WindPowerDataset()
    test_x, test_y = dataset.get_oob_test_set()
    preds = loaded_model.predict(test_x)
    print(preds)




def download_model():
    download_local_path = "./model"
    download_model_from_mlflow(model_name="mlp_2018-03-01", model_version="4", local_path=download_local_path)

if __name__ == "__main__":
    #mlflow_load_predict()
    download_model()
