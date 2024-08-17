import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional
import math

from .embedding import RotaryEmbedding

class RoPEMultiHeadSelfAttention(nn.Module):
    """
    RoPE Multi-Head Self Attention

    Args:
        input_dims (int): Number of input dimensions
        num_heads (int): Number of attention heads
        dropout (float, optional): Dropout rate. Defaults to 0.0.
        theta (float, optional): Scaling factor for frequency computation. Defaults to 10000.0.

    Inputs: 
        x (torch.Tensor): Input tensor with shape (batch_size, seq_len, input_dims)
        mask (torch.Tensor, optional): Mask tensor with shape (seq_len, seq_len) or (batch_size, 1, seq_len, seq_len). Defaults to None.

    Returns:
        torch.Tensor: Output tensor with shape (batch_size, seq_len, input_dims)
    """
    def __init__(self, input_dims: int, num_heads: int, dropout: float = 0.0, theta: float = 10000.0) -> None:
        super(RoPEMultiHeadSelfAttention, self).__init__()
        assert input_dims % num_heads == 0, "input_dims must be divisible by num_heads"
        self.input_dims = input_dims
        self.num_heads = num_heads
        self.head_dims = input_dims // num_heads

        self.dropout = dropout

        self.wq = nn.Linear(input_dims, input_dims, bias=False)
        self.wk = nn.Linear(input_dims, input_dims, bias=False)
        self.wv = nn.Linear(input_dims, input_dims, bias=False)
        self.rope = RotaryEmbedding(input_dims // num_heads)
        self.out_proj = nn.Linear(input_dims, input_dims, bias=False)
        self.dropout_layer = nn.Dropout(dropout)

    def scaled_dot_product_attention(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        attn_mask: Optional[torch.Tensor] = None,
        dropout: float = 0.1,
        is_causal: bool = False,
        scale: Optional[float] = None
    ) -> torch.Tensor:
        """
        Scaled Dot-Product Attention calculates the attention weights and outputs
        the weighted sum of the input values using the query, key, and value tensors.

        
        Args:
            query (torch.Tensor): Query tensor of shape (B, H, L, E).
            key (torch.Tensor): Key tensor of shape (B, H, S, E).
            value (torch.Tensor): Value tensor of shape (B, H, S, E).
            attn_mask (torch.Tensor, optional): Attention mask tensor of shape (L, S) or (B, 1, L, S). Defaults to None.
            dropout (float, optional): Dropout rate. Defaults to 0.1.
            is_causal (bool, optional): Whether the attention mask is causal. Defaults to False.
            scale (float, optional): Scale factor. Defaults to None.
            
        Returns:
            torch.Tensor: Output tensor of shape (B, H, L, E).
        """
        L, S = query.size(-2), key.size(-2)
        scale_factor = 1 / math.sqrt(query.size(-1)) if scale is None else scale

        attn_bias = torch.zeros((query.size(0), query.size(1), L, S), dtype=query.dtype, device=query.device)
        
        if is_causal:
            assert attn_mask is None, "attn_mask should be None when is_causal is True"
            temp_mask = torch.ones(L, S, dtype=torch.bool, device=query.device).tril(diagonal=0)
            attn_bias.masked_fill_(~temp_mask, float("-inf"))

        if attn_mask is not None:
            if attn_mask.dtype == torch.bool:
                attn_bias.masked_fill_(~attn_mask, float("-inf"))
            else:
                attn_bias += attn_mask
        
        attn_weights = torch.matmul(query, key.transpose(-2, -1)) * scale_factor
        attn_weights += attn_bias
        attn_weights = F.softmax(attn_weights, dim=-1)
        attn_weights = F.dropout(attn_weights, p=dropout, training=True)
        
        return torch.matmul(attn_weights, value)
    
    def forward(self, x: torch.Tensor, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        bsz, seqlen, _ = x.shape

        # Create padding mask if not provided (1 for non-pad positions, 0 for pad positions)
        if mask is None:
            mask = (x.sum(dim=-1) != 0).unsqueeze(1).unsqueeze(2)  # Shape: (batch_size, 1, 1, seq_len)
            mask = mask.expand(bsz, self.num_heads, seqlen, seqlen)  # Expand mask to match attention shape
            mask = mask.float()

        xq, xk, xv = self.wq(x), self.wk(x), self.wv(x)

        xq = xq.view(bsz, seqlen, self.num_heads, self.head_dims)
        xk = xk.view(bsz, seqlen, self.num_heads, self.head_dims)
        xv = xv.view(bsz, seqlen, self.num_heads, self.head_dims)
        
        # Apply Rotary Embedding
        xq, xk = self.rope(xq), self.rope(xk)
        
        # Transpose to prepare for scaled dot-product attention
        xq = xq.transpose(1, 2)
        xk = xk.transpose(1, 2)
        xv = xv.transpose(1, 2)

        # Apply scaled dot-product attention with mask
        attn_output = self.scaled_dot_product_attention(xq, xk, xv, mask)
        
        # Reshape and project output
        attn_output = attn_output.transpose(1, 2).contiguous().view(bsz, seqlen, self.input_dims)
        
        return self.out_proj(attn_output)


class RoPEMultiHeadSelfAttentionModule(nn.Module):
    def __init__(self, input_dims: int, num_heads: int, dropout: float = 0.1) -> None:
        super(RoPEMultiHeadSelfAttentionModule, self).__init__()
        self.layer_norm = nn.LayerNorm(input_dims)
        self.mhsa = RoPEMultiHeadSelfAttention(input_dims, num_heads)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x) -> torch.Tensor:
        x = self.layer_norm(x)
        x = self.mhsa(x)
        x = self.dropout(x)
        return x
