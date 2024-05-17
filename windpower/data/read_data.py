import pandas as pd


class WindPowerDataset:
    def __init__(self):
        self.dataset = pd.read_csv("https://github.com/dbczumar/model-registry-demo-notebook/raw/master/dataset/windfarm_data.csv", index_col=0)
        self.test_set = pd.DataFrame(self.dataset["2019-01-01":"2020-01-01"])
        self.train_set = pd.DataFrame(self.dataset["2014-01-01":"2019-01-01"])

    def get_tot_data(self):
        return self.dataset

    def get_train_validation_set(self, split_date="2018-01-01"):
        traindf = pd.DataFrame(self.train_set[:split_date])
        valid_df = pd.DataFrame(self.train_set[split_date:])
        print(traindf.columns)
        print(valid_df.columns)
        return traindf.drop(columns="power"), traindf["power"], valid_df.drop(columns="power"), valid_df["power"]


    def get_oob_test_set(self):
        return self.test_set.drop(columns="power"), self.test_set["power"]


if __name__ == "__main__":
    wind_data = WindPowerDataset()

    train_x, train_y, val_x, val_y = wind_data.get_train_validation_set()
