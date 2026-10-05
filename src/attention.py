import numpy as np
from scipy.special import softmax
import torch
import torch.nn as nn
import math

def scaled_dot_product_attention(Q, K, V):
    K_transposed = K.T ## Transpose matrix K
    S_similarity = Q @ K_transposed
    dk_sqrt = np.sqrt(K.shape[1])
    S_scaled = S_similarity / dk_sqrt
    attention_weights = softmax(S_scaled, axis=1)
    output = attention_weights @ V
    return output, attention_weights

class MultiHeadSelfAttention(nn.Module):

    def __init__(self, d_model, num_heads, max_len):
        super().__init__()
        self.d_model = d_model
        self.num_heads = num_heads
        assert d_model % num_heads == 0, "d_model not divisible by num_heads"
        self.d_k = d_model // num_heads
        self.max_len = max_len
        ## create bias table for relative positioning
        self.relative_bias = nn.Embedding(
            num_embeddings=2 * max_len - 1,
            embedding_dim=num_heads
            )

        self.W_Q = nn.Linear(d_model, d_model, bias=False)
        self.W_K = nn.Linear(d_model, d_model, bias=False)
        self.W_V = nn.Linear(d_model, d_model, bias=False)
        self.W_O = nn.Linear(d_model, d_model, bias=False)

    def forward(self, X):

        ## create Q, K, V from X
        Q = self.W_Q(X)
        K = self.W_K(X)
        V = self.W_V(X)
        B, L, D = X.shape
        assert L <= self.max_len, 'L is larger than relative position matrix'
        H = self.num_heads
        d_k = self.d_k
        Q = Q.reshape(B, L, H, d_k).transpose(1,2)
        K = K.reshape(B, L, H, d_k).transpose(1,2)
        V = V.reshape(B, L, H, d_k).transpose(1,2)
        
        # calculate relative position
        positions = torch.arange(L, device=X.device)
        relative_positions = positions.unsqueeze(0) - positions.unsqueeze(1)
        relative_positions = relative_positions + (self.max_len - 1)
        relative_bias = self.relative_bias(relative_positions)
        ## reshape to (H, L, L)
        relative_bias = relative_bias.permute(2, 0, 1)
        
        ## calculate QK^T
        S_similarity = Q @ K.transpose(-2, -1)

        ## scale by sqrt(d_k)
        dk_sqrt = math.sqrt(d_k)
        S_scaled = S_similarity / dk_sqrt
        ## Add relative positional bias
        S_scaled = S_scaled + relative_bias
        ## row-wise softmax
        attention_weights = torch.softmax(S_scaled, dim = -1)
        ## multiply attention weights by V
        output = attention_weights @ V
        output = output.transpose(1, 2).reshape(B, L, D)
        ## Combine the learned heads
        output = self.W_O(output)
        ## return output and attention weights
        return output, attention_weights