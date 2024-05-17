
from iris_classify.data.data_collect import IrisDataset
from sklearn.svm import SVC
from sklearn.metrics import precision_score, recall_score, f1_score, accuracy_score
import mlflow
from datetime import datetime



def train(X_train, Y_train, kernel):

    model = SVC(kernel=kernel)
    model.fit(X_train, Y_train)
    pred = model.predict(X_train)
    mlflow.log_param("kernel", kernel)

    record_metric("train", Y_train, pred)

    return model

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





if __name__ == "__main__":
    iris_dataset = IrisDataset()
    experiment_name = "iris"
    kernel = "poly"
    mlflow.set_tracking_uri("http://127.0.0.1:5000")

    exp = mlflow.get_experiment_by_name(experiment_name)
    if exp is not None:
        mlflow.set_experiment(exp.name)
    else:
        mlflow.create_experiment(experiment_name)

    x_train, x_test, y_train, y_test = iris_dataset.train_test_split(test_size=0.3)

    with mlflow.start_run() as run:
        mlflow.set_tag("mlflow.runName", f"{kernel}_{datetime.now().strftime('%Y%m%d-%H%M%S')}")
        model = train(x_train, y_train, kernel=kernel)
        y_pred = model.predict(x_test)
        record_metric("test", y_test, y_pred)









