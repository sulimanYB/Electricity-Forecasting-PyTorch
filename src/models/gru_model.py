import torch
import torch.nn as nn


class GRUModel(nn.Module):
    """
    GRU for one-step electricity consumption forecasting.

    Architecture:
        Input -> 2-layer GRU -> Final time step -> Linear -> Prediction

    Tensor shapes:
        Input:      (batch_size, 96, 1)
        GRU output: (batch_size, 96, 32)
        Hidden h_n: (2, batch_size, 32)
        Final step: (batch_size, 32)
        Prediction: (batch_size, 1)
    """

    def __init__(
        self,
        input_size: int = 1,
        hidden_size: int = 32,
        num_layers: int = 2,
    ):
        super().__init__()

        # GRU processes sequential data using two gates:
        # 1. Reset gate: controls the influence of previous information.
        # 2. Update gate: controls how much previous information is retained.
        #
        # batch_first=True expects:
        # (batch_size, sequence_length, input_size)
        self.gru = nn.GRU(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
        )

        # Convert the final hidden representation into one forecast.
        # (batch_size, 32) -> (batch_size, 1)
        self.fc = nn.Linear(hidden_size, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Perform one-step-ahead electricity consumption forecasting.

        Parameters
        ----------
        x : torch.Tensor
            Input sequences: (batch_size, 96, input_size)

        Returns
        -------
        torch.Tensor
            Predictions: (batch_size, 1)
        """

        # Process all 96 time steps through the two GRU layers.
        #
        # output: (batch_size, 96, hidden_size)
        # h_n:    (num_layers, batch_size, hidden_size)
        output, h_n = self.gru(x)

        # Select the final time-step output from the last GRU layer.
        # (batch_size, 96, 32) -> (batch_size, 32)
        final_output = output[:, -1, :]

        # Convert the final hidden representation into one prediction.
        # (batch_size, 32) -> (batch_size, 1)
        prediction = self.fc(final_output)

        return prediction