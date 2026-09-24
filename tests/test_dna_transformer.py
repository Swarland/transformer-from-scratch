import torch
import torch.nn as nn
from src.dna_transformer import DNATransformer



def test_dna_transformer_output_logits_shape():
    X = torch.randint(0, 5, (3, 10))
    model = DNATransformer(vocab_size=5,d_model=8,num_heads=2,d_ff=32,num_layers=3,max_len=100)
    logits, layer_weights = model(X)
    assert logits.shape == (3,1), 'logits shape malformed'

def test_dna_transformer_number_of_layer_weights():
    X = torch.randint(0, 5, (3, 10))
    model = DNATransformer(vocab_size=5,d_model=8,num_heads=2,d_ff=32,num_layers=3,max_len=100)
    logits, layer_weights = model(X)
    assert len(layer_weights) == 3, 'layer_weights length malformed'

def test_dna_transformer_shape_of_layer_weights():
    X = torch.randint(0, 5, (3, 10))
    model = DNATransformer(vocab_size=5,d_model=8,num_heads=2,d_ff=32,num_layers=3,max_len=100)
    logits, layer_weights = model(X)
    assert all([layer.shape == (3, 2, 10, 10) for layer in layer_weights]), 'layer_weights shape malformed'