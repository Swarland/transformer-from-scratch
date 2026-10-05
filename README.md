# Transformer from Scratch for Biological Sequences
This project implements the core components of a transformer from scratch in PyTorch with the goal of training a transformer block for biological seqeunce classification.

Rather than use nn.Transformer or pretrained models I am building it from first principles. I am using scaled dot-product attention, multi-head attention, positional encoding, Transformer encoder blocks, and the training pipeline directly to develop deeper understanding of transformer architecture and scientific deep learning. 


# Current Roadmap:

- Scaled dot-product attention - Completed
- Single-head self-attention - Completed
- Multi-head attention - Completed
- Positional encoding - Completed
- Transformer encoder block - Completed
- Biological sequence dataset - Completed
- Sequence classification - Completed
- Baseline comparison - Completed
- Model interpretation


Random-position motif classification

                 Validation accuracy
Transformer           ~50%
CNN                    99.5%

CNN Filter 7:
TATAAA → peak at motif: 50/50
AAATAT → peak at motif:  0/50