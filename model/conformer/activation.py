import torch
import torch.nn as nn

class Swish(nn.Module):
    """
    Swish activation function - self-gated activation function, 
    which means it has a gating mechanism that allows the activation
    function to learn which part of the input should be passed through
    and which part should be discarded.
    
    Swish(x) = x * sigmoid(x)

    Args:
        x (torch.Tensor): Input tensor

    Returns:
        torch.Tensor: Output tensor
    """
    def __init__(self) -> None:
        super(Swish, self).__init__()
        self.sigmoid = nn.Sigmoid()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x * self.sigmoid(x)
