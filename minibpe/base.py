"""
Contains the base Tokenizer class and a few common helper functions. 
"""

import unicodedata 


def get_stats(ids, counts=None):
    """Given a list of integers, return a dictionary of counts of consecutive pairs
    Example: [1, 2, 3, 1, 2] -> {(1, 2): 2, (2, 3): 1, (3, 1): 1}
    Optionally allows to update an existing dictionary of counts
    """
    counts = {} if counts is None else counts 
    for pair in zip(ids, ids[1:]):
        counts[pair] = counts.get(pair, 0) + 1
    return counts


def merge(ids, pair, idx):
    """
    In the list of integers (ids), replace all consecutive occurrences
    of pair with the new integer token idx
    Example: ids=[1, 2, 3, 1, 2], pair=(1, 2), idx=4 -> [4, 3, 4]
    """
    newids = []
    i = 0
    while i < len(ids):
         # if not at the very last position AND the pair matches, replace it
         if ids[i] == pair[0] and i < len(ids) - 1 and ids[i + 1] == pair[1]:
             newids.append(idx)
             i += 2
         else:
             newids.append(ids[i])
             i += 1
    return newids


# the base Tokenizer class 

class Tokenizer:
    """Base class for tokenizers""" 
    
    def __init__(self):
         # default: vocab size of 256 (all bytes), no merges, no patterns
        self.merges = {}  # (int, int) -> int 
        self.pattern = "" # str 