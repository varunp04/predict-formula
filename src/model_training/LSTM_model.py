import torch.nn as nn
from torch.autograd import Variable
import torch
import pandas as pd


class LSTM(nn.Module):
    def __init__(
        self,
        num_classes: int,
        input_size: int,
        hidden_size_layer_1: int,
        hidden_size_layer_2: int,
        num_layers: int,
        dense_layer_size: int,
        dropout_rate: float,
    ):
        super().__init__()
        self.num_classes = num_classes
        self.num_layers = num_layers
        self.hidden_size_1 = hidden_size_layer_1
        self.hidden_size_2 = hidden_size_layer_2
        self.dense_layer_size = dense_layer_size
        self.input_size = input_size

        # LSTM model
        self.lstm = nn.LSTM(
            input_size=self.input_size,
            hidden_size=self.hidden_size_1,
            num_layers=self.num_layers,
            batch_first=True,
            dropout=dropout_rate,
        )

        self.lstm_layer_2 = nn.LSTM(
            input_size=self.hidden_size_1,
            hidden_size=self.hidden_size_2,
            num_layers=self.num_layers,
            batch_first=True,
            dropout=dropout_rate,
        )

        self.batch_norm_1 = nn.BatchNorm1d(self.hidden_size_1)
        self.batch_norm_2 = nn.BatchNorm1d(self.hidden_size_2)

        self.fully_connected_layer_1 = nn.Linear(
            self.hidden_size_2, self.dense_layer_size
        )
        self.fully_connected_layer_2 = nn.Linear(self.dense_layer_size, num_classes)

        self.relu = nn.ReLU()
        self.sigmoid = nn.Sigmoid()
        self.dropout = nn.Dropout(dropout_rate)

    def forward(self, input_features: pd.DataFrame, device: str):

        h0_1 = Variable(
            torch.zeros(self.num_layers, input_features.size(0), self.hidden_size_1).to(
                device
            )
        )
        c0_1 = Variable(
            torch.zeros(self.num_layers, input_features.size(0), self.hidden_size_1).to(
                device
            )
        )

        h0_2 = Variable(
            torch.zeros(self.num_layers, input_features.size(0), self.hidden_size_2).to(
                device
            )
        )
        c0_2 = Variable(
            torch.zeros(self.num_layers, input_features.size(0), self.hidden_size_2).to(
                device
            )
        )

        LSTM_layer_1_output, _ = self.lstm(input_features, (h0_1, c0_1))
        LSTM_layer_1_output = self.batch_norm_1(
            LSTM_layer_1_output.transpose(1, 2)
        ).transpose(1, 2)

        LSTM_layer_2_output, _ = self.lstm_layer_2(LSTM_layer_1_output, (h0_2, c0_2))
        LSTM_layer_2_output = self.batch_norm_2(
            LSTM_layer_2_output.transpose(1, 2)
        ).transpose(1, 2)

        fully_connected_layer_1_output = self.fully_connected_layer_1(
            LSTM_layer_2_output[:, -1, :]
        )
        fully_connected_layer_1_output = self.relu(fully_connected_layer_1_output)
        fully_connected_layer_1_output = self.dropout(fully_connected_layer_1_output)
        fully_connected_layer_2_output = self.fully_connected_layer_2(
            fully_connected_layer_1_output
        )
        fully_connected_layer_2_output = self.relu(fully_connected_layer_2_output)

        return fully_connected_layer_2_output
