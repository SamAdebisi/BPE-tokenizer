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
        self.special_tokens = {} # str -> int, e.g. {'<|endoftext|>': 100257} 
        self.vocab = self._build_vocab() # int -> bytes 
        
    def train(self, text, vocab_size, verbose=False):
        # Tokenizer can train a vocabulary of size vocab_size from text 
        raise NotImplementedError 
    
    def encode(self, text):
        # Tokenizer can encode a string into a list of integers 
        raise NotImplementedError 
    
    def decode(self, ids):
        # Tokenizer can decode a list of integers into a string
        raise NotImplementedError 
    
    def _build_vocab(self):
        # vocab is simply and deterministically derived from merges 
        vocab = {idx: bytes([idx]) for idx in range(256)}
        for (p0, p1), idx in self.merges.items():
            vocab[idx] = vocab[p0] + vocab[p1]
        for special, idx in self.special_tokens.items():
            vocab[idx] = special.encode("utf-8")
        return vocab 
    
    def save(self, file_prefix): 
        """
       Saves two files: file_prefix.vocab and file_prefix.model 
       This is inspired (but not equivalent to!) sentencepiece's model saving:  
        """
        pass
    
    def load(self, file_prefix):
        """
        
        """
        pass 
    

# Two helper functions 

def replace_control_characters(s: str) -> str:
    pass 

def render_token(t: bytes) -> str:
    pass
