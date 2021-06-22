# External imports

from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import OneHotEncoder
from sklearn.linear_model import LinearRegression
from sklearn.compose import ColumnTransformer
from sklearn.model_selection import train_test_split
import pandas as pd

# Local imports
from TaxiFareModel import data, utils
from TaxiFareModel.encoders import DistanceTransformer, TimeFeaturesEncoder


class Trainer:
    def __init__(self, X, y):
        """
        X: pandas DataFrame
        y: pandas Series
        """
        self.pipeline = None
        self.X = X
        self.y = y

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
        self.pipe = Pipeline(
            [("preproc", preproc_pipe), ("linear_model", LinearRegression())]
        )

    def run(self):
        """set and train the pipeline"""

        self.pipe.fit(self.X, self.y)

    def evaluate(self, X_test, y_test):
        """evaluates the pipeline on df_test and return the RMSE"""
        y_pred = self.pipe.predict(X_test)
        print(utils.compute_rmse(y_pred, y_test))


if __name__ == "__main__":

    df = data.get_data()
    df = data.clean_data(df)
    y = df.pop("fare_amount")
    X = df
    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.15)
    trainer = Trainer(X_train, y_train)
    trainer.set_pipeline()
    trainer.run()
    # evaluate
    trainer.evaluate(X_val, y_val)
