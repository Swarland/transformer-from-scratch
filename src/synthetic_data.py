import random
def generate_synthetic_dna_no_motif(n_sequences, seq_len, motif):
    sequence_list = []
    bases = ['A', 'C', 'G', 'T']
    for i in range(int(n_sequences/2)):
        while True:
            seq = ''.join(random.choices(bases, k=seq_len))
            if motif not in seq:
                sequence_list.append(seq)
                break
    return sequence_list


def generate_synthetic_dna_with_motif(n_sequences, seq_len, motif):
    sequence_list = []
    bases = ['A', 'C', 'G', 'T']
    for i in range(int(n_sequences/2)):
        remaining_length = seq_len - len(motif)
        
        background = ''.join(random.choices(bases, k=remaining_length))

        insert_pos = random.randint(0, remaining_length)

        motif_dna = background[:insert_pos] + motif + background[insert_pos:]
        sequence_list.append(motif_dna)
    return sequence_list
    
    
def create_combined_dataset(negatives, positives):
    combined = negatives + positives
    labels = ([0] * len(negatives)) + ([1] * len(positives))
    paired = list(zip(combined, labels))
    random.shuffle(paired)
    combined, labels = map(list, zip(*paired))
    return combined, labels