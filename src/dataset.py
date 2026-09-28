from torch.utils.data import Dataset
import torch
from src.dna_tokenizer import tokenize_dna, CLS_TOKEN_ID
from torch.utils.data import DataLoader

class DNADataset(Dataset):

    def __init__(self, sequences, labels, add_cls=True):
        self.sequences = sequences
        self.labels = labels
        self.add_cls = add_cls
        assert len(sequences) == len(labels)

    def __len__(self):
        return len(self.sequences)

    def __getitem__(self, idx):

        ## tokenize sequence here
        sequence = self.sequences[idx]

        ## add the CLS token at the beginning of the sequence
        if self.add_cls:
            sequence_tokens = [CLS_TOKEN_ID] + tokenize_dna(sequence)
        else:
            sequence_tokens = tokenize_dna(sequence)


        ## this [] avoids having scaler tensor instead of 1D
        label = torch.tensor([self.labels[idx]], dtype=torch.float32)
        token_dna = torch.tensor(sequence_tokens, dtype=torch.long)

        return token_dna, label



