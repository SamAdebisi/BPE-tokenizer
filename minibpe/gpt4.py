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
        for i, pair in enumerate(zip(parts, parts[1:])): 
            pass 