import torch
import torch.nn as nn


class ConformerDecoder(nn.Module):

    def __init__(self, input_dims: int, hidden_dims: int, num_layers: int=1, dropout: float=0.1):
        super(ConformerDecoder, self).__init__()
        
    def forward