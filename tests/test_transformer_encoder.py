import torch
import torch.nn as nn
from src.attention import MultiHeadSelfAttention
from src.transformer_encoder import TransformerEncoderBlock
from src.transformer_encoder import TransformerEncoder


def test_transformer_encoder_output_shape_preserved():
    X = torch.rand(2, 4, 8)
    encoder= TransformerEncoderBlock(d_model=8, num_heads=2, d_ff=32)
    X_2, attention_weights = encoder(X)
    assert X.shape == X_2.shape, "transformer_encoder output malformed"

def test_attention_weight_shape():
    X = torch.rand(2, 4, 8)
    encoder = TransformerEncoderBlock(d_model=8, num_heads=2, d_ff=32)
    X_2, attention_weights = encoder(X)
    assert attention_weights.shape == (2,2,4,4), 'attention_weights shape malformed'

def test_encoder_transforms_input():
    X = torch.rand(2, 4, 8)
    encoder = TransformerEncoderBlock(d_model=8, num_heads=2, d_ff=32)
    X_2, attention_weights = encoder(X)
    assert not torch.allclose(X_2, X), 'encoder did not transform input'

def test_gradients_flow_through_block():
    X = torch.rand(2, 4, 8, requires_grad=True)
    encoder = TransformerEncoderBlock(d_model=8, num_heads=2, d_ff=32)
    X_2, attention_weights = encoder(X)
    loss = X_2.sum()
    loss.backward()
    assert X.grad is not None, "gradients did not flow through encoder block"

def test_encoder_transformer_layer_shape_preserved():
    X = torch.rand(2, 4, 8)
    encoder = TransformerEncoder(d_model=8, num_heads=2, d_ff=32, num_layers = 3)
    X_1, layer_weights = encoder(X)
    assert X_1.shape == X.shape, 'encoder transformer layer output malformed'

def test_number_returned_attention_layers():
    X = torch.rand(2, 4, 8)
    num_layers = 3
    encoder = TransformerEncoder(d_model=8, num_heads=2, d_ff=32, num_layers = num_layers)
    X_1, layer_weights = encoder(X)
    assert num_layers == len(layer_weights), 'returned different number of layer_weights than specified layers'

def test_shape_layer_attention_weights():
    X = torch.rand(2, 4, 8)
    encoder = TransformerEncoder(d_model=8, num_heads=2, d_ff=32, num_layers = 3)
    X_1, layer_weights = encoder(X)
    assert all(layer.shape == (2,2,4,4) for layer in layer_weights), "attention_weights layer malformed"

def test_layers_have_independent_parameters():
    encoder = TransformerEncoder(d_model=8, num_heads=2, d_ff=32, num_layers = 3)
    assert not torch.allclose(encoder.layers[0].attention.W_Q.weight, encoder.layers[1].attention.W_Q.weight), "layers parameters are not independent"


