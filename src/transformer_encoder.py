import torch
import torch.nn as nn
from src.attention import MultiHeadSelfAttention

class TransformerEncoderBlock(nn.Module):

    def __init__(self, d_model, num_heads, d_ff):
        super().__init__()

        self.attention = MultiHeadSelfAttention(d_model, num_heads)
        self.attention_norm = nn.LayerNorm(d_model)

        self.ffn = nn.Sequential(
            nn.Linear(d_model, d_ff),
            nn.ReLU(),
            nn.Linear(d_ff, d_model), 
            )
        self.ffn_norm = nn.LayerNorm(d_model)
    
    def forward(self, X):

        attention_output, attention_weights = self.attention(X)

        X_1 = self.attention_norm(X + attention_output)

        F = self.ffn(X_1)

        X_2 = self.ffn_norm(X_1 + F)

        return X_2, attention_weights


class TransformerEncoder(nn.Module):

    def __init__(self, d_model, num_heads, d_ff, num_layers):
        super().__init__()

        self.layers = nn.ModuleList(
            [TransformerEncoderBlock(
                d_model=d_model, num_heads=num_heads, d_ff=d_ff) 
                for _ in range(num_layers)]
        )

    def forward(self, X):
        layer_weights = []
        for layer in self.layers:
            X, attention_weights = layer(X)
            layer_weights.append(attention_weights)
        return X, layer_weights

