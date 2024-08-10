from model.conformer import ConvSubSampling
import torch.nn as nn
from data_loader import MyDataLoader
from utils import get_n_params
from loguru import logger

class CNNLayerNorm(nn.Module):
    """Layer normalization built for cnns input"""
    def __init__(self, n_feats):
        super(CNNLayerNorm, self).__init__()
        self.layer_norm = nn.LayerNorm(n_feats)

    def forward(self, x):
        # x (batch, channel, feature, time)
        x = self.layer_norm(x)
        return x#.transpose(2, 3).contiguous() # (batch, channel, feature, time)

class CNNBlock(nn.Module):
    def __init__(self, channel_size, kernel_size, stride, dropout, n_feats):
        super(CNNBlock, self).__init__()
        self.cnn1 = nn.Conv2d(channel_size, channel_size, kernel_size, stride, padding=kernel_size//2)
        self.cnn2 = nn.Conv2d(channel_size, channel_size, kernel_size, stride, padding=kernel_size//2)
        self.dropout = nn.Dropout(dropout)
        self.layer_norm = CNNLayerNorm(n_feats)

    def forward(self, x):
        residual = x
        x = self.layer_norm(x)
        x = nn.functional.gelu(x)
        x = self.dropout(x)
        x = self.cnn1(x)
        x = self.layer_norm(x)
        x = nn.functional.gelu(x)
        x = self.dropout(x)
        x = self.cnn2(x)
        x = residual + x
        return x

device = 'cuda:0'
audio_config = {
        "sample_rate": 16000,
        "win_length": 320,
        "hop_length": 160,
        "n_mels": 128
    }
data_loader = MyDataLoader(audio_config)
# Load validation datasets
#==========================================***===========================================
valid_dataset = data_loader.load_librispeech_datasets("data/", ["dev-clean"])

# Get Validation DataLoader
#==========================================***===========================================
valid_loader = data_loader.get_dataloader(valid_dataset, 
                                          batch_size=1, 
                                          shuffle=False, 
                                          num_workers=0)
logger.info(f"Validation DataLoader Size: {len(valid_loader)}")

for spectrograms, labels, input_lengths, label_lengths in valid_loader:
    spectrograms, labels, input_lengths, label_lengths = (
        spectrograms.to(device),
        labels.to(device),
        input_lengths.to(device),
        label_lengths.to(device)
    )
    logger.info(f'{spectrograms.shape[2]}  and {input_lengths}')
    logger.info(f'input shape: {spectrograms.shape}')
    conv = ConvSubSampling(1, 128).to(device)
    output, output_lengths = conv(spectrograms, input_lengths)
    logger.info(f'output shape: {output.shape}')
    logger.info(f'input/output length: {input_lengths} {output_lengths}')

    conv1 = nn.Conv2d(1, 128, kernel_size=3, stride=2, padding=1).to(device)
    output = conv1(spectrograms.transpose(2, 3))
    logger.info(f'output3 shape: {output.shape}')
    
    conv1 = nn.Conv2d(1, 128, kernel_size=3, stride=2, padding=1).to(device)
    output = conv1(spectrograms)
    logger.info(f'output2 shape: {output.shape}')


    block = CNNBlock(128, kernel_size=3, stride=1, dropout=0.1, n_feats=128//2).to(device)
    output = block(output)
    logger.info(f'output2 shape: {output.shape}')
    break

import torch
import torch.nn as nn

class ConvSubsamplingLayer(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride, padding):
        super(ConvSubsamplingLayer, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding)
        self.pool = nn.MaxPool2d(kernel_size=(2, 2), stride=(2, 2))

    def forward(self, x):
        # Apply convolution
        x = self.conv(x)
        # Apply max pooling for subsampling
        x = self.pool(x)
        return x

# Define input shape
input_shape = torch.Size([32, 1, 2941, 128])

# Initialize the layer
conv_subsampling_layer = ConvSubsamplingLayer(
    in_channels=1,       # Since input has 1 channel
    out_channels=16,     # Number of output channels (this can be adjusted)
    kernel_size=(3, 3),  # Kernel size
    stride=(1, 1),       # Stride for the convolution
    padding=(1, 1)       # Padding to maintain the size
)

# Create a random input tensor with the specified shape
input_tensor = torch.randn(input_shape)

# Forward pass through the subsampling layer
output_tensor = conv_subsampling_layer(input_tensor)

# Print output shape
print("Output shape:", output_tensor.shape)