import torch
import torch.nn as nn


class LSTMModel(nn.Module):
    """
    LSTM for one-step electricity consumption forecasting.

    Architecture:
        Input -> 2-layer LSTM -> Final time step -> Linear -> Prediction

    Tensor shapes:
        Input:      (batch_size, 96, 1)
        LSTM output:(batch_size, 96, 32)
        Hidden h_n: (2, batch_size, 32)
        Cell c_n:   (2, batch_size, 32)
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

        # LSTM processes sequences while maintaining two states:
        # h_t: hidden state (short-term representation)
        # c_t: cell state (long-term memory)
        #
        # batch_first=True expects input:
        # (batch_size, sequence_length, input_size)
        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
        )

        # Convert the final hidden representation into one prediction.
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

        # Process all 96 time steps through the two LSTM layers.
        #
        # output: (batch_size, 96, hidden_size)
        # h_n:    (num_layers, batch_size, hidden_size)
        # c_n:    (num_layers, batch_size, hidden_size)
        output, (h_n, c_n) = self.lstm(x)

        # Select the last time step from the final LSTM layer.
        # (batch_size, 96, 32) -> (batch_size, 32)
        final_output = output[:, -1, :]

        # Convert the final hidden representation into a prediction.
        # (batch_size, 32) -> (batch_size, 1)
        prediction = self.fc(final_output)

        return prediction