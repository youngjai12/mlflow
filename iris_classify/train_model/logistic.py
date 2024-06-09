from sklearn.linear_model import LogisticRegression
from sklearn.metrics import precision_score, recall_score, f1_score, accuracy_score
from iris_classify.data.data_collect import IrisDataset
import mlflow
from datetime import datetime
from mlflow.models import infer_signature

def record_metric(phase, true, pred):
    acc = accuracy_score(true, pred)
    prec = precision_score(true, pred, average=None)
    rec = recall_score(true, pred, average=None)
    f1 = f1_score(true, pred, average=None)

    mlflow.log_metric(key=f"{phase}_acc", value=acc)
    for idx, (each_prec, each_rec, each_f1) in enumerate(zip(prec, rec, f1)):
        mlflow.log_metric(key=f"{idx}_{phase}_prec", value=each_prec)
        mlflow.log_metric(key=f"{idx}_{phase}_rec", value=each_rec)
        mlflow.log_metric(key=f"{idx}_{phase}_f1", value=each_f1)
def train(X_train, Y_train, seed=123):
    model = LogisticRegression(random_state=seed).fit(X_train, Y_train)
    pred = model.predict(X_train)
    record_metric("train", Y_train, pred)
    return model


def cur_time_str():
    return datetime.now().strftime('%Y%m%d-%H%M%S')


if __name__ == "__main__":
    iris_dataset = IrisDataset()
    experiment_name = "iris"
    mlflow.set_tracking_uri("http://127.0.0.1:5000")

    exp = mlflow.get_experiment_by_name(experiment_name)
    if exp is not None:
        mlflow.set_experiment(exp.name)
    else:
        mlflow.create_experiment(experiment_name)

    #mlflow.sklearn.autolog()

    x_train, x_test, y_train, y_test = iris_dataset.train_test_split(test_size=0.3)
    signature = infer_signature(x_test, y_test)

    model_name = "sklearn-logistic-registed-model1"
    with mlflow.start_run(run_name=f"logistic_{cur_time_str()}") as run:
        model = train(x_train, y_train)
        y_pred = model.predict(x_test)
        record_metric("test", y_test, y_pred)

        model_info = mlflow.sklearn.log_model(
            sk_model=model,
            artifact_path="sklearn-logistic-k8s_resource",
            signature=signature,
            registered_model_name="sklearn-logistic-registed-model1"
        )

        mlflow.evaluate(
            model=model_info.model_uri,
            data=x_test,
            targets=y_test,
            model_type="classifier"
        )




        artifact_path = "k8s_resource"
        run_id = run.info.run_id

        # model_uri = f"runs:/{run_id}/{artifact_path}".format(run_id=run_id, artifact_path=artifact_path)
        # model_details = mlflow.register_model(model_uri=model_uri, name=model_name)
