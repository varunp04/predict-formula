from typing import Tuple, Dict, List
import torch
from LSTM_model import LSTM
from train import train
import pandas as pd
from feature import splitData
from transform import transformData


class tuneModel:
    def __init__(self, config: Dict, params: Dict) -> None:
        self.config = config
        self.train_obj = train(config=self.config)
        self.params = params
        self.split_obj = splitData()
        self.data_transform_obj = transformData(
            n_steps_input=params["NUMBER_OF_HISTORICAL_LAP"],
            n_steps_output=self.config.get("N_STEP_OUTPUT"),
            config=self.config,
        )

    def engineer_data(self, engineered_data: pd.DataFrame):

        engineered_data = engineered_data.drop(
            columns=self.config.get("DROP_COLUMNS_AVOID_LEAKAGE")
        )
        engineered_lap_times_df = engineered_data.drop(
            columns=self.params["DROP_COLUMNS_LIST"]
        )

        train_data, test_data = self.split_obj.train_test_split(
            data=engineered_lap_times_df,
            race_id_list=self.config.get("RACE_ID_TEST_SET"),
        )

        return train_data, test_data

    def transfrom_data(
        self, train_data: pd.DataFrame
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:

        scaled_input_df_train, scaled_output_df_train, scaler_dict = (
            self.data_transform_obj.create_scaled_input_output_data(
                data=train_data, train=True
            )
        )
        x_sequential_train, y_sequential_train = (
            self.data_transform_obj.create_sequence(
                scaled_input_df=scaled_input_df_train,
                scaled_output_df=scaled_output_df_train,
            )
        )
        x_train, x_val, y_train, y_val = self.split_obj.train_validation_split(
            x_sequential_ls=x_sequential_train, y_sequential_ls=y_sequential_train
        )
        x_train = torch.Tensor(x_train)
        y_train = torch.Tensor(y_train)

        x_train = torch.reshape(
            x_train,
            (
                x_train.shape[0],
                self.params["NUMBER_OF_HISTORICAL_LAP"],
                x_train.shape[2],
            ),
        )

        x_validation = torch.Tensor(x_val)
        y_validation = torch.Tensor(y_val)

        x_validation = torch.reshape(
            x_validation,
            (
                x_validation.shape[0],
                self.params["NUMBER_OF_HISTORICAL_LAP"],
                x_validation.shape[2],
            ),
        )

        return x_train, y_train, x_validation, y_validation, scaler_dict

    def train_model(
        self,
        x_train: torch.Tensor,
        y_train: torch.Tensor,
        x_validation: torch.Tensor,
        y_validation: torch.Tensor,
        device: str,
    ) -> Tuple[float, float, LSTM, List, List]:
        """return validation loss and train loss for the model with the parameters"""

        train_data_loader, validation_data_loader = self.train_obj.create_data_loader(
            batch_size=self.params["BATCH_SIZE"],
            x_train=x_train,
            y_train=y_train,
            x_validation=x_validation,
            y_validation=y_validation,
        )

        lstm_model = LSTM(
            num_classes=self.config.get("N_STEP_OUTPUT"),
            input_size=x_train.shape[2],
            hidden_size_layer_1=self.params["HIDDEN_SIZE_1"],
            hidden_size_layer_2=self.params["HIDDEN_SIZE_2"],
            num_layers=self.config.get("NUM_LAYERS"),
            dense_layer_size=self.config.get("DENSE_LAYER"),
            dropout_rate=0.2,
        )
        lstm_model.to(device)

        loss_fn = torch.nn.MSELoss()

        optimizer = torch.optim.Adam(
            lstm_model.parameters(), lr=self.params["LEARNING_RATE"]
        )

        validation_loss_ls, ls_train_loss, model = self.train_obj.train_validation_loop(
            n_epochs=self.config.get("N_EPOCHS"),
            lstm_model=lstm_model,
            optimizer=optimizer,
            loss_fn=loss_fn,
            train_loader=train_data_loader,
            validation_loader=validation_data_loader,
            device=device,
        )

        return (
            sum(validation_loss_ls) / len(validation_loss_ls),
            sum(ls_train_loss) / len(ls_train_loss),
            model,
            validation_loss_ls,
            ls_train_loss,
        )

    def perform_tuning(self, data: pd.DataFrame, device: str):

        train_data, test_data = self.engineer_data(engineered_data=data)

        x_train, y_train, x_validation, y_validation, scaler_dict = self.transfrom_data(
            train_data=train_data
        )

        (
            avg_val_loss,
            avg_train_loss,
            model,
            validation_loss_ls,
            ls_train_loss,
        ) = self.train_model(
            x_train=x_train,
            y_train=y_train,
            x_validation=x_validation,
            y_validation=y_validation,
            device=device,
        )

        return (
            avg_val_loss,
            avg_train_loss,
            model,
            validation_loss_ls,
            ls_train_loss,
            scaler_dict,
            test_data,
        )
