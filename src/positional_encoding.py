import torch
import torch.nn as nn


class PositionalEncoding(nn.Module):

    def __init__(self, d_model, max_len):
        super().__init__()
        
        PE = torch.zeros(max_len, d_model)
        position = torch.arange(max_len, dtype=torch.float32).unsqueeze(1)
        even_dims = torch.arange(0, d_model, 2, dtype=torch.float32)
        assert d_model % 2 == 0, 'd_model is not even'
        denominator = 10000 ** (even_dims / d_model)
        angles = position / denominator
        
        PE[:, 0::2] = torch.sin(angles)
        PE[:, 1::2] = torch.cos(angles)

        ## move to buffer
        self.register_buffer("PE", PE)

    def forward(self, X):
        B,L,D = X.shape
        X = X + self.PE[:L]
        return X

