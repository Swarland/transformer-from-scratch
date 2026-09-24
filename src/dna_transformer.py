import torch
import torch.nn as nn

from src.positional_encoding import PositionalEncoding
from src.transformer_encoder import TransformerEncoder


class DNATransformer(nn.Module):

    def __init__(self, vocab_size, d_model, num_heads, d_ff, num_layers, max_len):
        super().__init__()

        self.embedding = nn.Embedding(vocab_size, d_model)
        self.positional_encoding = PositionalEncoding(d_model, max_len)
        self.encoder = TransformerEncoder(d_model, num_heads, d_ff, num_layers)
        self.classifier = nn.Linear(d_model, 1)

    def forward(self, X):
        X = self.embedding(X)
        X = self.positional_encoding(X)
        X, layer_weights = self.encoder(X)
        X_pooled = X.mean(dim=1)
        X_classifier = self.classifier(X_pooled)
        return X_classifier, layer_weights