# External imports

from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import OneHotEncoder
from sklearn.linear_model import LinearRegression
from sklearn.compose import ColumnTransformer
from sklearn.model_selection import train_test_split
import pandas as pd
import mlflow
from mlflow.tracking import MlflowClient
from memoized_property import memoized_property

# Local imports
from TaxiFareModel import data, utils
from TaxiFareModel.encoders import DistanceTransformer, TimeFeaturesEncoder

MLFLOW_URI = "https://mlflow.lewagon.co/"
EXPERIMENT_NAME = "[FR] [lyon] [philippe-lemaire] linreg 2"


class Trainer:
    def __init__(self, X, y):
        """
        X: pandas DataFrame
        y: pandas Series
        """
        self.pipeline = None
        self.X = X
        self.y = y
        self.experiment_name = EXPERIMENT_NAME

    def set_pipeline(self):
        """defines the pipeline as a class attribute"""
        # create distance pipeline
        dist_pipe = Pipeline(
            [("dist_trans", DistanceTransformer()), ("stdscaler", StandardScaler())]
        )
        # create time pipeline
        time_pipe = Pipeline(
            [
                ("time_enc", TimeFeaturesEncoder("pickup_datetime")),
                ("ohe", OneHotEncoder(handle_unknown="ignore")),
            ]
        )
        # create preprocessing pipeline
        preproc_pipe = ColumnTransformer(
            [
                (
                    "distance",
                    dist_pipe,
                    [
                        "pickup_latitude",
                        "pickup_longitude",
                        "dropoff_latitude",
                        "dropoff_longitude",
                    ],
                ),
                ("time", time_pipe, ["pickup_datetime"]),
            ],
            remainder="drop",
        )

        # Add the model of your choice to the pipeline
        self.pipeline = Pipeline(
            [("preproc", preproc_pipe), ("linear_model", LinearRegression())]
        )

    def run(self):
        """set and train the pipeline"""
        self.set_pipeline()
        self.pipeline.fit(self.X, self.y)

    def evaluate(self, X_test, y_test):
        """evaluates the pipeline on df_test and return the RMSE"""
        y_pred = self.pipeline.predict(X_test)
        print(utils.compute_rmse(y_pred, y_test))

    @memoized_property
    def mlflow_client(self):
        mlflow.set_tracking_uri(MLFLOW_URI)
        return MlflowClient()

    @memoized_property
    def mlflow_experiment_id(self):
        try:
            return self.mlflow_client.create_experiment(self.experiment_name)
        except BaseException:
            return self.mlflow_client.get_experiment_by_name(
                self.experiment_name
            ).experiment_id

    @memoized_property
    def mlflow_run(self):
        return self.mlflow_client.create_run(self.mlflow_experiment_id)

    def mlflow_log_param(self, key, value):
        self.mlflow_client.log_param(self.mlflow_run.info.run_id, key, value)

    def mlflow_log_metric(self, key, value):
        self.mlflow_client.log_metric(self.mlflow_run.info.run_id, key, value)


if __name__ == "__main__":

    df = data.get_data()
    df = data.clean_data(df)
    y = df.pop("fare_amount")
    X = df
    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.15)
    trainer = Trainer(X_train, y_train)
    trainer.run()
    # evaluate
    trainer.evaluate(X_val, y_val)
