import functools
import operator
import numpy as np
from torch import nn

def get_n_params(model: nn.Module) -> int:
    """
    Count number of parameters in PyTorch model

    Args:
        model (nn.Module): PyTorch model
    
    Returns:
        int: Number of parameters
    """
    return sum((functools.reduce(operator.mul, p.size()) for p in model.parameters()))

def int2text(indices: list) -> str:
    idx_to_char = {1: "a", 2: "b", 3: "c", 4: "d", 5: "e", 6: "f", 7: "g", 8: "h", 9: "i", 10: "j",
                   11: "k", 12: "l", 13: "m", 14: "n", 15: "o", 16: "p", 17: "q", 18: "r", 19: "s",
                   20: "t", 21: "u", 22: "v", 23: "w", 24: "x", 25: "y", 26: "z", 27: "'", 28: " "}
    return ''.join(idx_to_char.get(idx, "") for idx in indices)

def greedy_decode(best_path: list, blank_idx=0) -> list:
    """
    Decode sequence of indices using greedy decoding algorithm, 
    removing consecutive duplicates and blanks characters.

    Args:
        best_path (list): List of indices
        blank_idx (int): Blank index
    
    Returns:
        list: Decoded sequence
    """
    # Remove consecutive duplicates and blanks
    decoded_sequence = []
    previous_char_idx = None

    for char_idx in best_path:
        if char_idx != blank_idx and char_idx != previous_char_idx:
            decoded_sequence.append(char_idx)
        previous_char_idx = char_idx
    return decoded_sequence

def calculate_wer(reference: str, hypothesis: str) -> float:
    """
    Calculate the Word Error Rate (WER) between a reference and a hypothesis.
    
    Args:
        reference (str): Reference text
        hypothesis (str): Hypothesized text

    Returns:
        float: Word Error Rate (WER)
    """
    r = reference.split()
    h = hypothesis.split()
    d = np.zeros((len(r)+1)*(len(h)+1), dtype=np.uint8)
    d = d.reshape((len(r)+1, len(h)+1))
    for i in range(len(r)+1):
        for j in range(len(h)+1):
            if i == 0:
                d[0][j] = j
            elif j == 0:
                d[i][0] = i
            elif r[i-1] == h[j-1]:
                d[i][j] = d[i-1][j-1]
            else:
                substitute = d[i-1][j-1] + 1
                insert = d[i][j-1] + 1
                delete = d[i-1][j] + 1
                d[i][j] = min(substitute, insert, delete)
    return d[len(r)][len(h)] / float(len(r))