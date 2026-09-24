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
    attention_weights = softmax(S_scaled, axis = 1)
    output = attention_weights @ V
    return(output, attention_weights)

class SelfAttention(nn.Module):

    def __init__(self, d_model):
        super().__init__()

        self.W_Q = nn.Linear(d_model, d_model, bias = False)
        self.W_K = nn.Linear(d_model, d_model, bias = False)
        self.W_V = nn.Linear(d_model, d_model, bias = False)

    def forward(self, X):

        # 1. create Q, K, V from X
        Q = self.W_Q(X)
        K = self.W_K(X)
        V = self.W_V(X)

        # 2. calculate QK^T
        S_similarity = Q @ K.transpose(-2, -1)

        # 3. scale by sqrt(d_k)
        dk_sqrt = math.sqrt(K.shape[1])
        S_scaled = S_similarity / dk_sqrt

        # 4. row-wise softmax
        attention_weights = torch.softmax(S_scaled, dim = -1)
        # 5. multiply attention weights by V
        output = attention_weights @ V
        # 6. return output and attention weights
        return output, attention_weights