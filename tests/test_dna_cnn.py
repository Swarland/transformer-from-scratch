import torch

from src.dna_cnn import DNACNN


def test_dna_cnn_output_shape():
    model = DNACNN(
    vocab_size=5,
    embedding_dim=8,
    num_filters=16,
    kernel_size=6)

    X = torch.randint(0, 5, (32, 100))

    logits = model(X)
    assert logits.shape == (32, 1), 'DNACNN output malformed'


def test_dna_cnn_gradient_flow():
    model = DNACNN(
    vocab_size=5,
    embedding_dim=8,
    num_filters=16,
    kernel_size=6)

    X = torch.randint(0, 5, (32, 100))

    logits = model(X)

    loss = logits.sum()
    loss.backward()
    assert model.conv.weight.grad is not None, "gradients did not flow through CNN"