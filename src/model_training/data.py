from sklearn.datasets import load_iris
import pandas as pd
import os
from datetime import datetime
import logging
import os
from typing import Dict

logger = logging.getLogger()
logger.setLevel(logging.INFO)


class LoadDataforModelling:

    def get_data(self) -> pd.DataFrame:
        """
        Extract iris data and return a data frame
        Return:
            Data frame with iris dataset
        """
        iris = load_iris()
        iris_df = pd.DataFrame(data=iris.data, columns=iris.feature_names)
        return iris_df


class DataCheckpoint:

    def __init__(self, file_name: str, config: Dict):
        self.file_name = file_name
        self.config = config
        self.output_path = os.path.join(
            config.get("MODEL_ARTIFACTS_PATH_PREFIX"), config.get("OUTPUT_LOCATION")
        )
        self.data_checkpoint_path = self.output_path + self.config.get(
            "DATA_CHECKPOINT_PATH"
        )
        self.file_path = f"{self.data_checkpoint_path}{self.file_name}.csv"

    def create_data_to_save(self, data_df: pd.DataFrame):
        """Add run info such as load_time and username to the dataframe before checkpoint"""
        for column in data_df.columns:
            if " " not in column:
                continue
            column_name = column
            column_name = column_name.replace(" ", "_")

            data_df = data_df.rename(columns={column: column_name})
            logger.info(f"In data, {column} renamed to {column_name}")
        df_data_copy = data_df.copy()
        df_data_copy["LOAD_TIME"] = datetime.today().strftime("%Y-%m-%d %H:%M:%S")

        return df_data_copy

    def checkpoint_at_s3(self, data: pd.DataFrame):
        """checkpoint the data to S3 bucket as a csv file"""

        if not os.path.exists(self.data_checkpoint_path):
            os.mkdir(self.data_checkpoint_path)

        data = self.create_data_to_save(data_df=data)

        data.to_csv(self.file_path, index=False)

        logging.info(
            f"{self.file_name} saved to: {self.data_checkpoint_path} in csv format"
        )
