import torch
import torch.nn as nn

from .activation import Swish

class FeedForwardModule(nn.Module):
    """
    Feed Forward Module - Consists of two linear layers with Swish activation function and dropout

    Args:
        input_dims (int): Number of input channels

    Inputs:
        x (torch.Tensor): Input tensor with shape (batch_size, channels, seq_len)

    Returns:
        torch.Tensor: Output tensor with shape (batch_size, channels, seq_len)
    """
    def __init__(self, input_dims: int, expansion_factor: int, dropout: float)-> None:
        super(FeedForwardModule, self).__init__()
        self.sequential = nn.Sequential(
            nn.LayerNorm(input_dims),
            nn.Linear(input_dims, input_dims*expansion_factor),
            Swish(),
            nn.Dropout(dropout),
            nn.Linear(input_dims*expansion_factor, input_dims),
            nn.Dropout(dropout),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.sequential(x)
