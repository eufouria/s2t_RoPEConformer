from model.conformer import ConvSubSampling
import torch.nn as nn
from data_loader import MyDataLoader
from utils import get_n_params
from loguru import logger
import torch
from model.conformer import RoPEConformer
# Define input shape
input_shape = torch.Size([3, 1, 128, 80])

# Model, Loss, and Optimizer
#==========================================***===========================================
model_params = {
        "input_dims": 80,
        "hidden_dims": 128,
        "enc_hidden_dims": 128,
        "dropout": 0.1,
        "ff_expansion_factor": 4,
        "ff_dropout": 0.2,
        "attn_num_heads": 4,
        "mhsa_dropout": 0.1,
        "kernel_size": 31,
        "conv_dropout": 0.1,
        "enc_num_layers": 16,
        "num_classes": 29
}
logger.info(f"HyperParams: {model_params}")
model = RoPEConformer(**model_params)
logger.info(f"Total params: {get_n_params(model)}")
# Create a random input tensor with the specified shape
inputs = torch.randn(input_shape)
input_lengths = torch.Tensor([12345, 12300, 12000])

# Forward pass through the subsampling layer
# Print output shape
# mask = (input_tensor > 0).unsqueeze(1).repeat(1, input_tensor.size(1), 1).unsqueeze(1)
output, output_lenghts = model(inputs, input_lengths)
print("output shape:", output.shape)
print("output_lenghts:" , output_lenghts)