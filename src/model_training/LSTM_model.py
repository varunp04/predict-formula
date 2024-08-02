import torch.nn as nn
from torch.autograd import Variable
import torch
import pandas as pd


class LSTM(nn.Module):
    def __init__(
        self,
        num_classes: int,
        input_size: int,
        hidden_size_layer: int,
        num_layers: int,
        dropout_rate: float,
    ):
        super().__init__()
        self.num_classes = num_classes
        self.num_layers = num_layers
        self.hidden_size = hidden_size_layer
        self.input_size = input_size

        
        # LSTM model
        self.lstm = nn.LSTM(
            input_size=self.input_size,
            hidden_size=self.hidden_size,
            num_layers=self.num_layers,
            batch_first=True,
            dropout=dropout_rate,
        )

        self.batch_norm = nn.BatchNorm1d(self.hidden_size)

        self.fully_connected_layer = nn.Linear(
            self.hidden_size, self.num_classes
        )

        self.relu = nn.ReLU()
        self.sigmoid = nn.Sigmoid()
        self.dropout = nn.Dropout(dropout_rate)

    def forward(self, input_features: pd.DataFrame, device: str):

        h0_1 = Variable(
            torch.zeros(self.num_layers, input_features.size(0), self.hidden_size).to(
                device
            )
        )
        c0_1 = Variable(
            torch.zeros(self.num_layers, input_features.size(0), self.hidden_size).to(
                device
            )
        )

        LSTM_layer_output, _ = self.lstm(input_features, (h0_1, c0_1))
        LSTM_layer_output = self.batch_norm(
            LSTM_layer_output.transpose(1, 2)
        ).transpose(1, 2)

        fully_connected_layer_output = self.fully_connected_layer(
            LSTM_layer_output[:, -1, :]
        )
        fully_connected_layer_output = self.relu(fully_connected_layer_output)
        return fully_connected_layer_output
