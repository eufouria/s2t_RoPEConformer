import torch
import torch.nn as nn
from typing import Tuple

from .decoder import ConformerDecoder
from .encoder import ConformerEncoder

class ConvSubSampling(nn.Module):
    """
    Convolutional Subsampling Module

    Args:
        in_channels (int): Number of input channels
        out_channels (int): Number of output channels

    Inputs:
        inputs (torch.Tensor): Input tensor with shape (batch_size, channels, seq_len, input_dim)
        input_lengths (torch.Tensor): Length of input tensor with shape (batch_size)

    Returns:
        outputs (torch.Tensor): Output tensor with shape (batch_size, subsampled_lengths, channels * sumsampled_dim)
        output_lengths (torch.Tensor): Length of output tensor with shape (batch_size)
    """
    def __init__(self, in_channels: int, out_channels: int, input_dims: int = 128) -> None:
        super(ConvSubSampling, self).__init__()
        self.sequential = nn.Sequential(
            nn.LayerNorm(input_dims),
            nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2, ceil_mode=True)
        )
        
        # Initialize the weights
        self._initialize_weights()

    def _initialize_weights(self) -> None:
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

    def forward(self, inputs: torch.Tensor, input_lengths: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        output = self.sequential(inputs)
        batch_size, channels, subsampled_lengths, sumsampled_dim = output.size()
        
        output = output.permute(0, 2, 1, 3)
        output = output.contiguous().view(batch_size, subsampled_lengths, channels * sumsampled_dim)
        output_lengths = input_lengths // 4

        return output, output_lengths

class RoPEConformer(nn.Module):
    """
    Conformer with Rotary Positional Encoding
    
    Args:
        input_dims (int): Dimension of input vector
        hidden_dims (int): Dimension of hidden vector
        dropout (float): Dropout rate
        ff_expansion_factor (int): Expansion factor of feed forward module
        ff_dropout (float): Dropout rate of feed forward module
        attn_num_heads (int): Number of attention heads
        mhsa_dropout (float): Dropout rate of multi-head self attention module
        kernel_size (int): Kernel size of convolution module
        conv_dropout (float): Dropout rate of convolution module
        num_layers (int): Number of conformer blocks
        num_classes (int): Number of classes
    
    Inputs: 
        inputs (torch.Tensor): Input tensor with shape (batch_size, seq_len, input_dims)
        input_lengths (torch.Tensor): Length of input tensor with shape (batch_size)

    Returns:
        torch.Tensor: Log probability of each class with shape (batch_size, num_classes)
        torch.Tensor: The length of input
        """
    def __init__(self, 
                 input_dims: int = 128, 
                 hidden_dims: int = 128, 
                 enc_hidden_dims: int = 128,
                 dropout: float = 0.1,
                 ff_expansion_factor: int = 4, 
                 ff_dropout: float = 0.1,
                 attn_num_heads: int = 8,
                 mhsa_dropout: float = 0.1,
                 kernel_size: int = 3,
                 conv_dropout: float = 0.1,
                 enc_num_layers: int = 8,
                 num_classes: int = 29) -> None:
        super(RoPEConformer, self).__init__()
        self.conv_subsampling = ConvSubSampling(1, hidden_dims, input_dims)
        self.linear = nn.Linear(hidden_dims * input_dims//2, enc_hidden_dims)
        self.dropout = nn.Dropout(dropout)

        self.encoder_layer = nn.ModuleList([
                ConformerEncoder(enc_hidden_dims, 
                               ff_expansion_factor, 
                               ff_dropout, 
                               attn_num_heads, 
                               mhsa_dropout, 
                               kernel_size, 
                               conv_dropout)
                for _ in range(enc_num_layers)
            ])
        self.decoder_layer = ConformerDecoder(enc_hidden_dims, num_classes)
        self.log_softmax = nn.LogSoftmax(dim=-1)
        
    def forward(self, x: torch.Tensor, input_lengths: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        x, output_lengths = self.conv_subsampling(x, input_lengths)
        x = self.linear(x)
        x = self.dropout(x)

        for layer in self.encoder_layer:
            x = layer(x)
        x = self.decoder_layer(x)
        output = self.log_softmax(x)
        return output, output_lengths
