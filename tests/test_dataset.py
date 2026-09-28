import torch
from src.dataset import DNADataset
import pytest


def test_dataset_length():
    sequences = ['ACGT', 'AACC']
    labels = [0,1]
    dataset = DNADataset(sequences, labels)
    assert len(dataset) == 2, 'dataset length malformed'

def test_dataset_tokenization():
    sequences = ['ACGT', 'AACC']
    labels = [0,1]
    dataset = DNADataset(sequences, labels)
    token_dna, label = dataset[0]
    assert torch.equal(token_dna, torch.tensor([5, 0, 1, 2, 3])), 'dataset tokenization off'

def test_dataset_output_shape():
    sequences = ['ACGT', 'AACC']
    labels = [0,1]
    dataset = DNADataset(sequences, labels)
    token_dna, label = dataset[0]
    assert token_dna.shape == (5,), 'dataset output malformed'
    assert label.shape == (1,), 'dataset output malformed'

def test_dataset_output_dtype():
    sequences = ['ACGT', 'AACC']
    labels = [0,1]
    dataset = DNADataset(sequences, labels)
    token_dna, label = dataset[0]
    assert token_dna.dtype == torch.long, 'dataset dtype malformed'
    assert label.dtype == torch.float32, 'dataset dtype malformed'
    
def test_dataset_sequence_label_length_match():
    with pytest.raises(AssertionError):
        sequences = ['ACGT', 'AACC']
        labels = [0,1,2]
        dataset = DNADataset(sequences, labels)


##define tests to work without CLS

def test_dataset_output_shape_without_cls():
    sequences = ['ACGT', 'AACC']
    labels = [0,1]
    dataset = DNADataset(sequences, labels, add_cls=False)
    token_dna, label = dataset[0]
    assert token_dna.shape == (4,), 'dataset output malformed'
    assert label.shape == (1,), 'dataset output malformed'

def test_dataset_tokenization_without_cls():
    sequences = ['ACGT', 'AACC']
    labels = [0,1]
    dataset = DNADataset(sequences, labels, add_cls=False)
    token_dna, label = dataset[0]
    assert torch.equal(token_dna, torch.tensor([0, 1, 2, 3])), 'dataset tokenization off'