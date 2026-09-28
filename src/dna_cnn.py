import torch
import torch.nn as nn


class DNACNN(nn.Module):

    def __init__(self, vocab_size, embedding_dim, num_filters, kernel_size):
        super().__init__()

        self.embedding = nn.Embedding(vocab_size, embedding_dim)

        self.conv = nn.Conv1d(
            in_channels=embedding_dim, 
            out_channels=num_filters, 
            kernel_size=kernel_size)

        self.classifier = nn.Linear(num_filters,1)

    def forward(self, X):

        X = self.embedding(X)
        X = X.transpose(1,2)

        X = self.conv(X)
        X = torch.relu(X)

        X = torch.max(X, dim=2).values

        logits = self.classifier(X)

        return logits