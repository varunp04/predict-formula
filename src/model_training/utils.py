import os
import pickle
import io
import boto3
from pathlib import Path
import pandas as pd
import yaml
from typing import Dict
import logging
import warnings

warnings.filterwarnings("ignore")
logger = logging.getLogger()


def save_as_pickle(path: str, artifact_name: str, artifact):

    with open(os.path.join(path, artifact_name), "wb") as out:
        pickle.dump(artifact, out)

    logging.info(f"{artifact_name} saved to {path}")


def save_as_yaml(path: str, artifact_name: str, artifact):

    with open(os.path.join(path, artifact_name), "w") as out:
        yaml.dump(artifact, out)

    logging.info(f"{artifact_name} saved to {path}")


def save_trained_model(trained_model, filename: Path) -> None:
    """
    Save the pickle file of the trained model
    Params:
        trained_model: trained_model to save
        filename: Name of the pickled filename
    """
    dir = Path("models")
    if not os.path.exists(dir):
        os.makedirs(dir)
    pickle.dump(trained_model, open(dir / filename, "wb"))


def load_config_file(file_name: str) -> Dict:
    """Load the config in the form of dictionary

    Params:
        file_name: file location
    Returns:
        Return the content of the file as a dictionary
    """
    with open(file_name, "r") as file:
        config_file_content = yaml.safe_load(file)
    return config_file_content


def save_in_s3(df: pd.DataFrame, bucket: str, object_name: str) -> None:
    """Upload a file to S3
    Params:
        df: dataframe to upload to s3 bucket
        bucket: Name of the bucket
        object_name: Name of the object inside the bucket
    """
    client = boto3.client("s3", region_name="us-west-2")

    with io.StringIO() as csv_buffer:
        df.to_csv(csv_buffer, index=False)

        response = client.put_object(
            Bucket=bucket, Key=object_name, Body=csv_buffer.getvalue()
        )

        status = response.get("ResponseMetadata", {}).get("HTTPStatusCode")

    if status == 200:
        logging.info(f"Successful S3 put_object response. Status - {status}")
    else:
        logging.error(f"Unsuccessful S3 put_object response. Status - {status}")
