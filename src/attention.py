import numpy as np
from scipy.special import softmax


def scaled_dot_product_attention(Q, K, V):
    K_transposed = K.T ## Transpose matrix K
    S_similarity = Q @ K_transposed
    dk_sqrt = np.sqrt(K.shape[1])
    S_scaled = S_similarity / dk_sqrt
    attention_weights = softmax(S_scaled, axis = 1)
    output = attention_weights @ V
    return(output, attention_weights)