"""
    Minimal (byte-level) Byte Pair Encoding tokenizer.
    
    Algorithmically follows along the GPT tokenizer:
https://github.com/openai/gpt-2/blob/master/src/encoder.py

But:
- Does not handle the regular expression splitting pattern.
- Does not handle any special tokens.
"""

from .base import get_stats, merge, Tokenizer 


class BasicsicTokenizer(Tokenizer):
    """
    A basic Byte Pair Encoding tokenizer.
    """
    
    def __init__(self):
        super().__init__() 
        
    def train(self, text, vocab_size, verbose=False):
        assert vocab_size >= 256 
        num_merges = vocab_size - 256 
        
        # input text preprocessing 
        text_bytes = text.encode("utf-8") # raw bytes 
        ids = list(text_bytes) # list of integers in range 0...255 
        
        # iteratively merge the most common pairs to create new tokens 
        merges = {} # (int, int) -> int 
        vocab = {idx: bytes([idx]) for idx in range(256)} # int -> bytes 
        
        for i in range(num_merges):
            # count up the number of times every consecutive pair appears 
            stats = get_stats(ids) 
            # find the pair with the highest count 
            pair = max(stats, key=stats.get)
            # mint a new token: assign it the next available id 
            idx = 256 + i 
            # replace all occurrences of pair in ids with idx 
            ids = merge(ids, pair, idx)
            # save the merge 
            merges[pair] = idx 
            vocab[idx] = vocab[pair[0]] + vocab[pair[1]]
            # print 
            if verbose:
                print(f"merge {i+1}/{num_merges}: {pair} -> {idx} ({vocab[idx]}) had {stats[pair]} occurrences")
                
            