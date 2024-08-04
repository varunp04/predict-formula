import torch
from torch.utils.data import DataLoader
from data import TensorDataset
from utils import save_as_pickle
from typing import Tuple, Dict, List
from LSTM_model import LSTM


class train:

    def __init__(self, config) -> None:
        self.config = config

    def create_data_loader(
        self,
        batch_size: int,
        x_train: torch.Tensor,
        y_train: torch.Tensor,
        x_validation: torch.Tensor,
        y_validation: torch.Tensor,
    ) -> Tuple[DataLoader, DataLoader]:

        train_dataset = TensorDataset(x_train, y_train)
        validation_dataset = TensorDataset(x_validation, y_validation)

        train_data_loader = DataLoader(
            train_dataset, batch_size=batch_size, shuffle=True
        )
        validation_data_loader = DataLoader(
            validation_dataset, batch_size=batch_size, shuffle=False
        )

        return train_data_loader, validation_data_loader

    def train_one_epoch(
        self,
        lstm: torch.nn.LSTM,
        optimizer: torch.optim.Adam,
        loss_fn: torch.nn.MSELoss,
        train_loader: DataLoader,
        device: str,
    ):
        """Training loop to train and optimize the LSTM model defined"""

        lstm.train(True)
        running_loss = 0.0

        for i, batch in enumerate(train_loader):
            x_batch, y_batch = batch[0].to(device), batch[1].to(device)
            optimizer.zero_grad()

            output = lstm(x_batch, device=device)
            loss = loss_fn(output, y_batch)
            running_loss += loss.item()

            loss.backward()
            optimizer.step()
        avg_train_loss = running_loss / i
        print(f"Average train loss accross batch: {avg_train_loss}")
        return lstm, avg_train_loss

    def test_one_epoch(
        self,
        lstm: torch.nn.LSTM,
        loss_fn: torch.nn.MSELoss,
        validation_loader: DataLoader,
        device: str,
    ):
        """test the trained LSTM model for one epoch"""

        lstm.train(False)
        running_loss = 0.0

        for i, batch in enumerate(validation_loader):
            x_batch, y_batch = batch[0].to(device), batch[1].to(device)

            with torch.no_grad():
                output = lstm(x_batch, device=device)
                loss = loss_fn(output, y_batch)
                running_loss += loss
        avg_loss_accroos_batch_in_test = running_loss / len(validation_loader)

        print(f"Validation loss: {avg_loss_accroos_batch_in_test}")

        return avg_loss_accroos_batch_in_test

    def train_validation_loop(
        self,
        n_epochs: int,
        lstm_model: torch.nn.LSTM,
        optimizer: torch.optim.Adam,
        loss_fn: torch.nn.MSELoss,
        train_loader: DataLoader,
        validation_loader: DataLoader,
        device: str,
    ) -> Tuple[List, List, LSTM]:

        ls_validation_loss = []
        ls_train_loss = []

        for epoch in range(n_epochs):

            print(f"Epoch num: {epoch}")

            lstm, avg_train_loss = self.train_one_epoch(
                lstm=lstm_model,
                optimizer=optimizer,
                loss_fn=loss_fn,
                train_loader=train_loader,
                device=device,
            )
            ls_train_loss.append(avg_train_loss)
            validation_loss = self.test_one_epoch(
                lstm=lstm,
                loss_fn=loss_fn,
                validation_loader=validation_loader,
                device=device,
            )
            ls_validation_loss.append(validation_loss)

        return ls_validation_loss, ls_train_loss, lstm

    def tune_model(
        self,
        params: Dict,
        x_train: torch.Tensor,
        y_train: torch.Tensor,
        x_validation: torch.Tensor,
        y_validation: torch.Tensor,
        device: str,
    ) -> Tuple[float, float, LSTM, List, List]:
        """return validation loss and train loss for the model with the parameters"""

        train_data_loader, validation_data_loader = self.create_data_loader(
            batch_size=params["BATCH_SIZE"],
            x_train=x_train,
            y_train=y_train,
            x_validation=x_validation,
            y_validation=y_validation,
        )

        lstm_model = LSTM(
            num_classes=self.config.get("N_STEP_OUTPUT"),
            input_size=x_train.shape[2],
            hidden_size_layer_1=params["HIDDEN_SIZE_1"],
            hidden_size_layer_2=params["HIDDEN_SIZE_2"],
            num_layers=self.config.get("NUM_LAYERS"),
            dense_layer_size=self.config.get("DENSE_LAYER"),
            dropout_rate=0.2,
        )
        lstm_model.to(device)

        loss_fn = torch.nn.MSELoss()

        optimizer = torch.optim.Adam(
            lstm_model.parameters(), lr=params["LEARNING_RATE"]
        )

        validation_loss_ls, ls_train_loss, model = self.train_validation_loop(
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
