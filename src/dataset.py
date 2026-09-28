from torch.utils.data import Dataset
import torch
from src.dna_tokenizer import tokenize_dna
from torch.utils.data import DataLoader

class DNADataset(Dataset):

    def __init__(self, sequences, labels):
        self.sequences = sequences
        self.labels = labels
        assert len(sequences) == len(labels)

    def __len__(self):
        return len(self.sequences)

    def __getitem__(self, idx):
        ## tokenize sequence here
        sequence = self.sequences[idx]
        ## this avoids having scaler tensor instead of 1D
        label = torch.tensor([self.labels[idx]], dtype=torch.float32)

        token_dna = torch.tensor(tokenize_dna(sequence), dtype=torch.long)

        return token_dna, label



