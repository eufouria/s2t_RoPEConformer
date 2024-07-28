import torch
import torch.nn as nn

from .convolution import ConvolutionModule
from .feedforward import FeedForwardModule
from .mhsa import RoPEMultiHeadSelfAttentionModule

class ResConnectionModule(nn.Module):
    """
    Residual connection module: applies step *residual connection to the input tensor

    Args:
        module (nn.Module): Sub neural net module
        step (float): Step value

    Inputs:
        x (torch.Tensor): Input tensor with shape (batch_size, seq_len, hidden_dims)
    """
    def __init__(self, module: nn.Module, step: float = 1.0) -> None:
        super(ResConnectionModule, self).__init__()
        self.module = module
        self.step = step
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.module(x) * self.step

class ConformerEncoder(nn.Module):
    """
    Conformer block contains two Feed Forward modules sandwiching the 
    Multi-Headed Self-Attention module and the Convolution module with half-step residual weights 
    in feed-forward modules and full-step residual weights in attention and convolution modules.

    Args:
        input_dims (int): Dimension of input vector
        ff_expansion_factor (int): Expansion factor of feed forward module
        ff_dropout (float): Dropout rate of feed forward module
        attn_num_heads (int): Number of attention heads
        mhsa_dropout (float): Dropout rate of multi-head self attention module
        kernel_size (int): Kernel size of convolution module
        conv_dropout (float): Dropout rate of convolution module

    Inputs:
        x (torch.Tensor): Input tensor with shape (batch_size, seq_len, input_dims

    Returns:
        torch.Tensor: Output tensor with shape (batch_size, seq_len, input_dims)
    """
    def __init__(self, 
                 input_dims: int = 128, 
                 ff_expansion_factor: int = 4, 
                 ff_dropout: float = 0.1,
                 attn_num_heads: int = 8,
                 mhsa_dropout: float = 0.1,
                 kernel_size: int = 4,
                 conv_dropout: float = 0.1) -> None:
        super(ConformerEncoder, self).__init__()

        self.sequential = nn.Sequential(
            ResConnectionModule(FeedForwardModule(input_dims, ff_expansion_factor, ff_dropout), step=0.5),
            ResConnectionModule(RoPEMultiHeadSelfAttentionModule(input_dims, attn_num_heads, mhsa_dropout), step=1),
            ResConnectionModule(ConvolutionModule(input_dims, kernel_size, conv_dropout), step=1),
            ResConnectionModule(FeedForwardModule(input_dims, ff_expansion_factor, ff_dropout), step=0.5),
            nn.LayerNorm(input_dims)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.sequential(x)
