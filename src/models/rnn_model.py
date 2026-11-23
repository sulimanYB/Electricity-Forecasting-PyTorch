import torch
import torch.nn as nn


class RNNModel(nn.Module):
    """
    Vanilla RNN for one-step electricity consumption forecasting.

    Architecture:
        Input -> 2-layer RNN -> Final time step -> Linear -> Prediction

    Tensor shapes:
        Input:      (batch_size, 96, 1)
        RNN output: (batch_size, 96, 32)
        Hidden:     (2, batch_size, 32)
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

        # RNN processes the sequence one time step at a time.
        # batch_first=True expects (batch, sequence_length, features).
        self.rnn = nn.RNN(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
        )

        # Convert the final hidden representation into one prediction.
        self.fc = nn.Linear(hidden_size, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Perform one-step-ahead forecasting.

        Parameters
        ----------
        x : torch.Tensor
            Input sequences: (batch_size, 96, input_size)

        Returns
        -------
        torch.Tensor
            Predictions: (batch_size, 1)
        """

        # Process all time steps through the two RNN layers.
        # output: (batch_size, 96, hidden_size)
        # hidden: (num_layers, batch_size, hidden_size)
        output, hidden = self.rnn(x)

        # Select the output of the final time step (t = 95).
        # (batch_size, 96, hidden_size) -> (batch_size, hidden_size)
        final_output = output[:, -1, :]

        # Map the final hidden representation to one forecast.
        # (batch_size, hidden_size) -> (batch_size, 1)
        prediction = self.fc(final_output)

        return prediction

        