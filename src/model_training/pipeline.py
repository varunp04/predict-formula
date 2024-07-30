import pandas as pd
from sklearn.model_selection import train_test_split
from utils import save_as_pickle
import numpy as np
from typing import Dict
from data import LoadDataforModelling


import logging

logger = logging.getLogger()
logger.setLevel(logging.INFO)


class RunExperimentOnSagemaker:

    def get_data(self):
        """Get data from Snowflake"""

        try:

            logger.info("Connecting to Snowflake to acquire data.")

            load_data = LoadDataforModelling()

            raw_data = load_data.get_data()

            x = raw_data.drop(columns=["sepal length (cm)"], axis=1)
            y = raw_data["sepal length (cm)"]
            logger.info("Successfully acquired the data.")

            return x, y

        except Exception as e:

            logger.error(f"Error in getting the data: {e}")

    def train_set_split(
        self, x: pd.DataFrame, y: pd.DataFrame, test_size: int
    ) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        """
        split the data and return train and test sets
        Params:
            x: input to the model
            y: desired output
        Returns:
            input and output, split into x_train, y_train
            and x_test, y_test
        """
        return train_test_split(x, y, test_size=test_size, random_state=42)

    def train_the_model(
        self, config: Dict, model, x_train: np.ndarray, y_train: np.ndarray
    ):
        """Train the model"""
        model_path = config.get("MODEL_ARTIFACTS_PATH_PREFIX") + config.get(
            "MODEL_LOCATION"
        )

        trained_model = model.fit(x_train, y_train)
        save_as_pickle(
            path=model_path, artifact_name="model.pkl", artifact=trained_model
        )
