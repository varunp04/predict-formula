import os
import pandas as pd
from typing import Dict
import pickle


class MakeInference:

    def __init__(self, config: Dict) -> None:

        self.config = config
        self.model_artifact_path = config.get(
            "MODEL_ARTIFACTS_PATH_PREFIX"
        ) + config.get("MODEL_LOCATION")

    def model_fn(self):
        """Load the model"""
        with open(self.model_artifact_path + "model.pkl", "rb") as f:
            loaded_model = pickle.load(f)
        return loaded_model

    def perform_inference(self, x_test) -> pd.DataFrame:
        """Perfrom feature engineering and data transformation. get predcitions from the model.
        And lastly, create a dataframe with actuals and predictions for evaluation
        """

        model = self.model_fn()

        predictions = model.predict(x_test)

        return predictions
