import torch
import torch.nn as nn


class ConformerDecoder(nn.Module):
    """
    Decoder layer contains a single LSTM-layer

    Args:
        input_dims (int): Number of input dimensions
        hidden_dims (int): Number of hidden dimensions
        num_layers (int): Number of LSTM layers
        dropout (float): Dropout rate

    Inputs:
        x (torch.Tensor): Input tensor with shape (batch_size, seq_len, input_dims)

    Returns:
        torch.Tensor: Output tensor with shape (batch_size, seq_len, hidden_dims)
    """
    def __init__(self, input_dims: int, hidden_dims: int, num_layers: int=1, dropout: float=0.1):
        super(ConformerDecoder, self).__init__()
        self.lstm = nn.LSTM(input_dims, hidden_dims, num_layers, batch_first=True, bidirectional=False)
        self.gelu = nn.GELU()
        self.dropout = nn.Dropout(dropout)
        # Apply Kaiming initialization to LSTM weights
        self._initialize_weights()

    def _initialize_weights(self):
        for name, param in self.lstm.named_parameters():
            if 'weight_ih' in name:
                nn.init.kaiming_uniform_(param.data, nonlinearity='relu')
            elif 'weight_hh' in name:
                nn.init.kaiming_uniform_(param.data, nonlinearity='relu')
            elif 'bias' in name:
                param.data.fill_(0)


    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x, _ = self.lstm(x)
        x = self.gelu(x)
        x = self.dropout(x)
        return x
