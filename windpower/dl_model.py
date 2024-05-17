import os.path
import sys
import tensorflow as tf
import mlflow
from tensorflow.keras.layers import Dense
from tensorflow.keras.models import Sequential
from datetime import datetime
from windpower.data.read_data import WindPowerDataset
from sklearn.metrics import mean_squared_error
import pandas as pd
from mlflow.models import infer_signature

DATA_BASE_DIR = "./data"

def get_model_v2(input_shape):
    model = Sequential()
    model.add(Dense(100, input_shape=input_shape, activation="relu", name="hidden_layer1"))
    model.add(Dense(50, input_shape=input_shape, activation="relu", name="hidden_layer2"))
    model.add(Dense(10, input_shape=input_shape, activation="relu", name="hidden_layer3"))
    model.add(Dense(1))
    model.compile(loss="mse", optimizer="adam")

    return model


def get_model(input_shape):
    model = Sequential()
    model.add(Dense(100, input_shape=input_shape, activation="relu", name="hidden_layer"))
    model.add(Dense(1))
    model.compile(loss="mse", optimizer="adam")

    return model

def train_model(model, X_train, y_train, val_x, val_y):
    model.fit(X_train, y_train, epochs=20, batch_size=64, validation_split=.2, validation_data=(val_x, val_y))


def record_mse(phase, y_true, y_pred):
    test_mse = mean_squared_error(y_true, y_pred)
    mlflow.log_metric(f"{phase}_mse", test_mse)

def cur_time_str():
    return datetime.now().strftime('%Y%m%d-%H%M%S')


def write_data_to_local(train_x, train_y, dataset_name):
    path = f"{DATA_BASE_DIR}/{dataset_name}.csv"
    if os.path.exists(path):
        return
    to_write_df = pd.concat([train_x, train_y])
    to_write_df.to_csv(path)
    return path


if __name__ == "__main__":
    #train_end_date = sys.argv[1]
    train_end_date="2018-03-01"
    dataset = WindPowerDataset()
    #input_shape = (10, )

    train_x, train_y, val_x, val_y = dataset.get_train_validation_set(split_date=train_end_date)
    test_x, test_y = dataset.get_oob_test_set()
    dataset.get_oob_test_set()
    written_path = write_data_to_local(train_x, train_y, dataset_name="train")
    written_test_path = write_data_to_local(test_x, test_y, dataset_name="test")

    input_shape = (train_x.shape[-1], )


    experiment_name = "brandon-wind"
    mlflow.set_tracking_uri("http://127.0.0.1:5000")
    #mlflow.set_tracking_uri("http://mlflow-talkaa.dev.onkakao.net")


    exp = mlflow.get_experiment_by_name(experiment_name)
    if exp is not None:
        mlflow.set_experiment(exp.name)
    else:
        mlflow.create_experiment(experiment_name)


    with mlflow.start_run(run_name=f"mlp2_auto_{cur_time_str()}") as run:
        mlflow.tensorflow.autolog()
        mlflow.log_param("train_date", f"2014-01-01 ~ {train_end_date}")
        #mlp = get_model(input_shape)
        mlp = get_model_v2(input_shape)
        train_model(mlp, train_x, train_y, val_x, val_y)

        test_pred_y = mlp.predict(test_x)
        val_pred_y = mlp.predict(val_x)
        train_pred_y = mlp.predict(train_x)

        record_mse("test", test_y, test_pred_y)
        record_mse("train", train_y, train_pred_y)
        record_mse("val", val_y, val_pred_y)

        if written_path is not None:
            mlflow.log_artifact(local_path=written_path)


        artifact_path = "model"
        run_id = run.info.run_id

        #data_signaure = infer_signature(train_x, train_y)

        model_info = mlflow.tensorflow.log_model(
            model=mlp,
            artifact_path="mlp-model",
            registered_model_name="mlp-registed-model1"
        )

        model_name = f"mlp_{train_end_date}"
        model_uri = f"runs:/{run_id}/{artifact_path}".format(run_id=run_id, artifact_path=artifact_path)
        model_details = mlflow.register_model(model_uri=model_uri, name=model_name)





