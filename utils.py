import jiwer
import functools
import operator
import numpy as np
from torch import nn
from typing import List, Tuple


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
                   20: "t", 21: "u", 22: "v", 23: "w", 24: "x", 25: "y", 26: "z", 27: " "}
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


def cal_wer_detail(
    hypotheses: List[str], references: List[str], use_cer=False
) -> Tuple[float, int, float, float, float]:
    """
    Reference: 
    https://github.com/NVIDIA/NeMo/blob/96b5bc9c5e228364f534ecebc5bfc602934864e1/nemo/collections/asr/metrics/wer.py#L70

    Computes Average Word Error Rate with details (insertion rate, deletion rate, substitution rate)
    between two texts represented as corresponding lists of string.

    Hypotheses and references must have same length.

    Args:
        hypotheses (list): list of hypotheses
        references(list) : list of references
        use_cer (bool): set True to enable cer

    Returns:
        wer (float): average word error rate
        words (int):  Total number of words/charactors of given reference texts
        ins_rate (float): average insertion error rate
        del_rate (float): average deletion error rate
        sub_rate (float): average substitution error rate
    """
    scores = 0
    words = 0
    ops_count = {'substitutions': 0, 'insertions': 0, 'deletions': 0}

    if len(hypotheses) != len(references):
        raise ValueError(
            "In word error rate calculation, hypotheses and reference"
            " lists must have the same number of elements. But I got:"
            "{0} and {1} correspondingly".format(len(hypotheses), len(references))
        )

    for h, r in zip(hypotheses, references):
        if use_cer:
            h_list = list(h)
            r_list = list(r)
        else:
            h_list = h.split()
            r_list = r.split()

        # To get rid of the issue that jiwer does not allow empty string
        if len(r_list) == 0:
            if len(h_list) != 0:
                errors = len(h_list)
                ops_count['insertions'] += errors
            else:
                errors = 0
        else:
            if use_cer:
                measures = jiwer.cer(r, h, return_dict=True)
            else:
                measures = jiwer.compute_measures(r, h)

            errors = measures['insertions'] + measures['deletions'] + measures['substitutions']
            ops_count['insertions'] += measures['insertions']
            ops_count['deletions'] += measures['deletions']
            ops_count['substitutions'] += measures['substitutions']

        scores += errors
        words += len(r_list)

    if words != 0:
        wer = 1.0 * scores / words
        ins_rate = 1.0 * ops_count['insertions'] / words
        del_rate = 1.0 * ops_count['deletions'] / words
        sub_rate = 1.0 * ops_count['substitutions'] / words
    else:
        wer, ins_rate, del_rate, sub_rate = float('inf'), float('inf'), float('inf'), float('inf')

    return wer, words, ins_rate, del_rate, sub_rate