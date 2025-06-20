"""
    Implements the GPT-4 Tokenizer as a light wrapper around the RegexTokenizer. 
    Note that this is a pretrained tokenizer. By default and inside init(), it loads 
    the pretrained tokenizer from the `cl100k_base` tokenizer of tiktoken. 
"""

import tiktoken 
from .regex import RegexTokenizer 


def bpe(mergeable_ranks, token, max_rank):
    # helper function used in get_gpt4_merges() to reconstruct the merge forest 
    parts = [bytes([b]) for b in token]
    while True:  
        min_idx = None 
        min_rank = None 
        for i, pair in enumerate(zip(parts[:-1], parts[1:])): 
            rank = mergeable_ranks.get(pair[0] + pair[1]) 
            if rank is not None and (min_rank is None or rank < min_rank): 
                min_idx = i 
                min_rank = rank
        if min_rank is None or (min_rank is not None and min_rank >= max_rank): 
                break 
        assert min_idx is not None 
        parts = parts[:min_idx] + [parts[min_idx] + parts[min_idx + 1]] + parts[min_idx + 2:]
    return parts 

def recover_merges(mergeable_ranks):
    # the `merges` are already the byte sequences in their merged state. 
    # so we have to recover the original pairings. We can do this by doing a small 
    # BPE training run on all the tokens, in their order.
    
    