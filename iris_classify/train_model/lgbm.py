import mlflow

from iris_classify.data.data_collect import IrisDataset
from lightgbm import LGBMClassifier

MODEL_PARAM = {
    'objective': 'multiclass',
    'num_class': 3,
    'metric': 'multi_logloss',
    'colsample_bytree': 0.8,
    'subsample': 0.8,
    'seed': 42,
}

TRAIN_PARAM = {
    'num_boost_round': 30,
    'verbose_eval': 5,
    'early_stopping_rounds': 5,
}

if __name__ == "__main__":
    iris_dataset = IrisDataset()
    experiment_name = "iris"
    mlflow.set_tracking_uri("http://127.0.0.1:5000")

    exp = mlflow.get_experiment_by_name(experiment_name)
    if exp is not None:
        mlflow.set_experiment(exp.name)
    else:
        mlflow.create_experiment(experiment_name)

    mlflow.lightgbm.autolog()

    with mlflow.start_run():
        model = lgb.train(
            bst_params,
            train_set,
            valid_sets=[train_set, val_set],
            valid_names=['train', 'valid'],
            **train_params,
        )


