from src.dna_tokenizer import tokenize_dna

def test_tokenize_dna_sequence_output():
    sequence = 'ACGTN'
    assert tokenize_dna(sequence) == [0,1,2,3,4], 'tokenizer sequence malformed'

def test_tokenize_dna_lowercase_sequence_output():
    sequence = 'acgtn'
    assert tokenize_dna(sequence) == [0,1,2,3,4], 'tokenizer lowercase test failed'

def test_tokenize_dna_sequence_length_output():
    sequence = 'ACGTN'
    assert len(tokenize_dna(sequence)) == len(sequence), 'tokenizer output length is unexpected'
