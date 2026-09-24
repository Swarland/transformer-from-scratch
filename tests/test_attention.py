import torch
from src.attention import MultiHeadSelfAttention
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

