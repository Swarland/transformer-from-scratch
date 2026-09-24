import torch
from src.attention import MultiHeadSelfAttention
from src.positional_encoding import PositionalEncoding
import pytest

def test_output_shape():
    X = torch.rand(2, 4, 8)
    attention = MultiHeadSelfAttention(d_model=8, num_heads=2)
    output, attention_weights = attention(X)
    assert output.shape == (2,4,8), "output shape is malformed"


def test_attention_shape():
    X = torch.rand(2, 4, 8)
    attention = MultiHeadSelfAttention(d_model=8, num_heads=2)
    output, attention_weights = attention(X)
    assert attention_weights.shape == (2,2,4,4), "attention_weights shape is malformed"

def test_attention_rows_sum_to_one():
    X = torch.rand(2, 4, 8)
    attention = MultiHeadSelfAttention(d_model=8, num_heads=2)
    output, attention_weights = attention(X)
    row_sums = attention_weights.sum(dim=-1)
    assert torch.allclose(row_sums, torch.ones_like(row_sums)), "attention_rows do not sum to one"

def test_invalid_num_heads():
    with pytest.raises(AssertionError):
        ## d_model is not evenly divisible by 3 so rais error
        attention = MultiHeadSelfAttention(d_model=8, num_heads=3)

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

