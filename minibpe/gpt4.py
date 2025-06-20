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
    # also see https://github.com/openai/tiktoken/issues/60
    # also see https://github.com/karpathy/minbpe/issues/11#issuecomment-1950805306 
    
    merges = {} 
    for token, rank in mergeable_ranks.items():
        if len(token) == 1:
            continue # skip raw bytes 
        pair = tuple(bpe(mergeable_ranks, token, max_rank=rank))
        assert len(pair) == 2 
        # recover the integer ranks of the pair 
        ix0 = mergeable_ranks[pair[0]]
        ix1 = mergeable_ranks[pair[1]]
        pair = (ix0, ix1)
        merges[pair] = rank 
    return merges

GPT4_SPLIT_PATTERN = r"""'(?i:[sdmt]|ll|ve|re)|[^\r\n\p{L}\p{N}]?+\p{L}+|\p{N}{1,3}| ?[^\s\p{L}\p{N}]++[\r\n]*|\s*[\r\n]|\s+(?!\S)|\s+"""
GPT4_SPECIAL_TOKENS = {
    '<|endoftext|>': 100257,
    '<|fim_prefix|>': 100258,
    '<|fim_middle|>': 100259,
    '<|fim_suffix|>': 100260,
    '<|endofprompt|>': 100276
}

class GPT4Tokenizer(RegexTokenizer):
    """Lightweight wrapper on RegexTokenizer that matches GPT-4's tokenizer."""
    
    def __init__(self):
        super().__init__(pattern=GPT4_SPLIT_PATTERN)
        # get the official tokenizer and its merges 
        enc = tiktoken.get_encoding("cl100k_base")
        mergeable_ranks = enc.mergeable_ranks 
        # the merges are those of gpt4, but we have to recover them 
        self.merges = recover_merges(mergeable_ranks)
        # reconstruct the vocab from the mergers 
        vocab = {idx: bytes([idx]) for idx in range(256)}
        for (p0, p1), idx in self.merges.items():
            vocab[idx] = vocab[p0] + vocab[p1]
        self.vocab = vocab 
        # now here is another tricky part. 
        # for some reason, the tokens corresponding to individual bytes 
        # are permuted in a different order. This is completely non-sensical 
        # and probably historical, but therefore we have to deal with it here. 
        self.byte_shuffle = {i: mergeable_ranks[bytes([i])] for i in range(256)}
        self.inverse_special_tokens = {v: k for k, v in self.byte_shuffle.items()}
        # finally register the special tokens 