import torch
import torch.nn as nn

from .activation import Swish

class PointWiseConv1d(nn.Module):
    """
    Pointwise Convolution Module - Convolution module with kernel size of 1

    Args:
        in_channels (int): Number of input channels
        out_channels (int): Number of output channels
        stride (int): Stride value
        padding (int): Padding value
        bias (bool): Flag indication whether bias is added or not
    
    Inputs:
        x (torch.Tensor): Input tensor with shape (batch_size, channels, seq_len)

    Returns:
        torch.Tensor: Output tensor with shape (batch_size, out_channels, seq_len)
    """
    def __init__(self, 
                 in_channels: int, 
                 out_channels: int, 
                 stride: int = 1, 
                 padding: int = 0, 
                 bias: bool = False) -> None:
        super(PointWiseConv1d, self).__init__()
        self.conv = nn.Conv1d(in_channels, out_channels, kernel_size=1, stride=stride, padding=padding, bias=bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.conv(x)

class DepthWiseConv1d(nn.Module):
    """
    Depthwise Convolution Module - Convolution module with groups set to in_channels

    Args:
        in_channels (int): Number of input channels
        out_channels (int): Number of output channels
        kernel_size (int): Kernel size of convolution
        stride (int): Stride value
        padding (int): Padding value
        bias (bool): Flag indication whether bias is added or not

    Inputs:
        x (torch.Tensor): Input tensor with shape (batch_size, channels, seq_len)   

    Returns:
        torch.Tensor: Output tensor with shape (batch_size, out_channels, seq_len)
    """
    def __init__(self, 
                 in_channels: int, 
                 out_channels: int, 
                 kernel_size: int,
                 stride: int = 1, 
                 padding: int = 0, 
                 bias: bool = False) -> None:
        super(DepthWiseConv1d, self).__init__()
        self.conv = nn.Conv1d(in_channels,
                              out_channels,
                              kernel_size=kernel_size, 
                              stride=stride,
                              padding=padding,
                              groups=in_channels,
                              bias=bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.conv(x)
    
class ConvolutionModule(nn.Module):
    """
    Convolution Module - Convolution module with GLU activation

    Args:
        input_dims (int): Number of input channels
        kernel_size (int): Kernel size of convolution
        dropout (float): Dropout rate

    Inputs:
        x (torch.Tensor): Input tensor with shape (batch_size, channels, seq_len)

    Returns:
        torch.Tensor: Output tensor with shape (batch_size, channels, seq_len)
    """
    def __init__(self, 
                 input_dims: int, 
                 kernel_size: int, 
                 dropout: float = 0.1) -> None:
        super(ConvolutionModule, self).__init__()
        self.layer_norm = nn.LayerNorm(input_dims)
        self.pointwise_conv1 = PointWiseConv1d(input_dims, input_dims*2, stride=1, padding=0, bias=True)
        self.glu = nn.GLU(dim=1)
        self.depthwise_conv = DepthWiseConv1d(input_dims, input_dims, kernel_size, stride=1, padding=(kernel_size - 1) // 2, bias=True)
        self.batch_norm = nn.BatchNorm1d(input_dims)
        self.swish = Swish()
        self.pointwise_conv2 = PointWiseConv1d(input_dims, input_dims, stride=1, padding=0, bias=True)
        self.layer_norm = nn.LayerNorm(input_dims)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.layer_norm(x).transpose(1, 2)
        x = self.pointwise_conv1(x)
        x = self.glu(x)
        x = self.depthwise_conv(x)
        x = self.batch_norm(x)
        x = self.swish(x)
        x = self.pointwise_conv2(x)
        x = self.dropout(x)
        return x.transpose(1, 2)
