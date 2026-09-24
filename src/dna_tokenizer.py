DNA_VOCAB = {'A': 0, 'C': 1, 'G': 2, 'T': 3, 'N': 4}

def tokenize_dna(sequence):
    sequence = sequence.upper()
    sequence_tokens = [DNA_VOCAB[nucleotide] for nucleotide in sequence]
    return sequence_tokens
