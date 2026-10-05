import torch
import torch.nn as nn
from src.attention import MultiHeadSelfAttention
from src.positional_encoding import PositionalEncoding
import pytest

def test_output_shape():
    X = torch.rand(2, 4, 8)
    attention = MultiHeadSelfAttention(d_model=8, num_heads=2, max_len=20)
    output, attention_weights = attention(X)
    assert output.shape == (2,4,8), "output shape is malformed"


def test_attention_shape():
    X = torch.rand(2, 4, 8)
    attention = MultiHeadSelfAttention(d_model=8, num_heads=2, max_len=20)
    output, attention_weights = attention(X)
    assert attention_weights.shape == (2,2,4,4), "attention_weights shape is malformed"

def test_attention_rows_sum_to_one():
    X = torch.rand(2, 4, 8)
    attention = MultiHeadSelfAttention(d_model=8, num_heads=2, max_len=20)
    output, attention_weights = attention(X)
    row_sums = attention_weights.sum(dim=-1)
    assert torch.allclose(row_sums, torch.ones_like(row_sums)), "attention_rows do not sum to one"

def test_invalid_num_heads():
    with pytest.raises(AssertionError):
        ## d_model is not evenly divisible by 3 so rais error
        attention = MultiHeadSelfAttention(d_model=8, num_heads=3, max_len=20)

def test_position_output_shape_preserved():
    X = torch.rand(2, 4, 8)
    output = PositionalEncoding(d_model=8, max_len=10)
    assert output(X).shape == (2,4,8), "position output shape malformed"

def test_different_positions_different_encodings():
    X = torch.zeros(2, 4, 8)
    pos_encoding = PositionalEncoding(d_model=8, max_len=10)
    result = pos_encoding(X)
    assert not torch.allclose(result[0,0,:], result[0,1,:])

def test_same_position_batch_encoding():
    X = torch.zeros(2, 4, 8)
    pos_encoding = PositionalEncoding(d_model=8, max_len=10)
    result = pos_encoding(X)
    assert torch.allclose(result[0,1,:], result[1,1,:])

def test_odd_d_model_raises_error():
    with pytest.raises(AssertionError):
        output = PositionalEncoding(d_model=7, max_len=10)

def test_position_zero_known_encoding():
    X = torch.zeros(2, 4, 8)
    known_encodings = torch.tensor([0,1,0,1,0,1,0,1], dtype=torch.float32)
    pos_encoding = PositionalEncoding(d_model=8, max_len=10)
    result = pos_encoding(X)
    assert torch.allclose(result[0,0,:], known_encodings)

## now we initiiate relative positional encoding 

def test_relative_position_encoding_position_matrix():
    attention = MultiHeadSelfAttention(
        d_model=8,
        num_heads=2,
        max_len=10
    )
    L = 4
    positions = torch.arange(L)

    relative_positions = (
        positions.unsqueeze(0)
        - positions.unsqueeze(1)
    )
    relative_indices = relative_positions + (attention.max_len - 1)
    bias = attention.relative_bias(relative_indices)
    assert torch.equal(bias[0, 1, 0], bias[1, 2, 0]), 'positional distance matrix malformed'
    assert torch.equal(bias[1, 2, 0], bias[2, 3, 0]), 'positional distance matrix malformed'
    assert relative_indices[0, 1] != relative_indices[1, 0], 'positional distance matrix malformed'


    
def test_relative_position_encoding_position_bias():
    attention = MultiHeadSelfAttention(
        d_model=8,
        num_heads=2,
        max_len=10
        )
    with torch.no_grad():
        nn.init.zeros_(attention.W_Q.weight)
        nn.init.zeros_(attention.W_K.weight)
        nn.init.zeros_(attention.relative_bias.weight)

        plus_one_idx = (attention.max_len - 1) + 1
        attention.relative_bias.weight[plus_one_idx, 0] = 10

        X = torch.randn(1, 4, 8)
        output, weights = attention(X)
        ## Head 0 strongly prefers the +1 position
        assert weights[0, 0, 0, 1] > 0.99
        assert weights[0, 0, 1, 2] > 0.99
        assert weights[0, 0, 2, 3] > 0.99
        ## Head 0's final query has no +1 position, so it is uniform
        expected_uniform = torch.full((4,), 0.25)
        assert torch.allclose(
            weights[0, 0, 3],
            expected_uniform,
            atol=1e-5
        )

        ## Head 1 has no positional preference at all
        expected_head1 = torch.full((4, 4), 0.25)
        assert torch.allclose(
            weights[0, 1],
            expected_head1,
            atol=1e-5
        )
    