import torch
import torch.nn as nn


class ConformerDecoder(nn.Module):
    """
    Simple ASR Decoder for use with CTC-based model

    Args:
        input_dims (int): Number of input dimensions
        hidden_dims (int): Number of hidden dimensions

    Inputs:
        enc (torch.Tensor): Encoded tensor with shape (batch_size, seq_len, input_dims)
    
    Returns:
        torch.Tensor: Decoded tensor with shape (batch_size, seq_len, hidden_dims)
    """
    def __init__(self, input_dims: int, hidden_dims: int):
        super(ConformerDecoder, self).__init__()
        self.conv1d = nn.Conv1d(input_dims, hidden_dims, kernel_size=1, bias=True)

    def forward(self, enc: torch.Tensor) -> torch.Tensor:
        x = enc.transpose(1, 2)
        x = self.conv1d(x)
        return x.transpose(1, 2)
