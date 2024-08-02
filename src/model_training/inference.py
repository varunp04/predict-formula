import pickle
import pandas as pd
from typing import Dict, List, Tuple
from transform import tranformDataInference
import torch


class makeInference:
    def __init__(self, config: Dict) -> None:

        self.config = config
        self.TARGET_COLUMN = self.config.get("TARGET_COLUMN")

    def transfrom_data(self, data: pd.DataFrame) -> Tuple[List, List]:
        """Transform the data for inference"""

        transform_obj = tranformDataInference(
            n_steps_input=self.config.get("N_STEP_INPUT"),
            n_steps_output=self.config.get("N_STEP_OUTPUT"),
            config=self.config,
        )

        scaled_input_df, output_df, lap_df = transform_obj.create_scaled_data_inference(
            data=data
        )

        x_sequential, y_sequential = transform_obj.create_sequence(
            scaled_input_df=scaled_input_df, scaled_output_df=output_df
        )

        _, lap_sequential = transform_obj.create_sequence(
            scaled_input_df=scaled_input_df, scaled_output_df=lap_df
        )

        return x_sequential, y_sequential, lap_sequential

    def model_fn(self):
        """Load model"""
        with open(f"{self.config.get('MODEL_PATH')}model.pkl", "rb") as f:
            loaded_model = pickle.load(f)
        return loaded_model

    def inverse_transform_predictions(self, predictions: torch.Tensor) -> torch.Tensor:
        """Load scaler dict to inverse transform the predictions"""

        with open(f"{self.config.get('MODEL_PATH')}scaler_dict.pkl", "rb") as f:
            scaler_dict = pickle.load(f)

        transformed_predictions = scaler_dict[
            f"{self.TARGET_COLUMN}_scaler"
        ].inverse_transform(predictions)
        return transformed_predictions

    def perform_inference(self, test_data: pd.DataFrame, device: str):
        """Perform necessary transformation and then predict"""

        x_sequential, y_sequential, lap_sequential = self.transfrom_data(data=test_data)

        lstm_model = self.model_fn()

        input_tensor = torch.Tensor(x_sequential)

        input_tensor = torch.reshape(
            input_tensor,
            (
                input_tensor.shape[0],
                2,
                input_tensor.shape[2],
            ),
        )

        input_tensor = input_tensor.to(device)

        with torch.no_grad():
            predictions = lstm_model(input_tensor, device=device)

        print(predictions)

        inverse_transformed_preds = self.inverse_transform_predictions(
            predictions=predictions
        )

        return inverse_transformed_preds, y_sequential, lap_sequential
