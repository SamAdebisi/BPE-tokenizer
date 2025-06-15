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
    for pair in zip(ids[:-1], ids[1:]):
        counts[pair] = counts.get(pair, 0) + 1
    return counts
