"""
Minimal (byte-level) Byte Pair Encoding tokenizer.

Algorithmically follows along the GPT tokenizer:
https://github.com/openai/gpt-2/blob/master/src/encoder.py

Unlike BasicTokenizer:
- RegexTokenizer handles an optional regex splitting pattern.
- RegexTokenizer handles optional special tokens.
"""

import regex as re 
from .base import Tokenizer, get_stats, merge 

# the main GPT text split patterns, see
# https://github.com/openai/tiktoken/blob/main/tiktoken_ext/openai_public.py 

GPT2_SPLIT_PATTERN = r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"""
GPT4_SPLIT_PATTERN = r"""'(?i:[sdmt]|ll|ve|re)|[^\r\n\p{L}\p{N}]?+\p{L}+|\p{N}{1,3}| ?[^\s\p{L}\p{N}]++[\r\n]*|\s*[\r\n]|\s+(?!\S)|\s+"""

class RegexTokenizer(Tokenizer):
    
    def __init__(self, pattern=None):
        """
        - pattern: optional string to override the default (GPT-4 split pattern)
        - special_tokens: str -> int dictionary of special tokens
          example: {'<|endoftext|>': 100257}
        """
        super().__init__()
        self.pattern = GPT4_SPLIT_PATTERN if pattern is None else pattern 
        self.compiled_pattern = re.compile(self.pattern) 
        self.special_tokens = {} # str -> int 
        self.inverse_special_tokens = {} # int -> str 
        
    def train(self, text, vocab_size, verbose=False):
        assert vocab_size >= 256
        num_merges = vocab_size - 256 
        
        # split the text up into text chunks 
        text_chunks = re.findall(self.compiled_pattern, text) 
        
        # input text preprocessing 
        ids = [list(ch.encode("utf-8")) for ch in text_chunks] 
        
        # iteratively merge the most common pairs to create new tokens 
        merges = {} # (int, int) -> int 
        vocab = {idx: bytes([idx]) for idx in range(256)} # int -> bytes 
        for i in range(num_merges): 
            # count up the number of times every consecutive pair appears 
            stats = {} 
            for chunk_ids in ids:
                # passing in stats will update it in place, adding up counts 
                get_stats(chunk_ids, stats) 
            # find the pair with the highest count 
            pair = max(stats, key=stats.get)
            # mint a new token: assign it the next available id  
            idx = 256 + i 
            # replace all occurrences of pair in ids with idx 
            ids = [merge(chunk_ids, pair, idx) for chunk_ids in ids] 
            # save the merge 
            merges[pair] = idx
            vocab[idx] = vocab[pair[0]] + vocab[pair[1]] 
            # prints 
            if verbose:
                print(f"merge {i+1}/{num_merges}: {pair} -> {idx}) ({vocab[idx]}) had {stats[pair]} occurrences") 
                
        # save class variables 
        self.merges = merges # used in encode()
        self.vocab = vocab   # used in decode() 
        
    def register_special_tokens(self, special_tokens):
        # special_tokens is a dictionary of str -> int 
        # example: {"<|endoftext|>": 100257}
        self.special_tokens = special_tokens 
        self.inverse_special_tokens = {v: k for k, v in special_tokens.items()} 
        
    def decode(self, ids):
        # given ids (list of integers), return Python string 
        part_bytes = [] 
        for idx in ids:
            if idx in self.vocab:
                part_bytes.append(self.vocab[idx]) 
            elif idx in self.inverse_special_tokens:
                part_bytes.append(self.inverse_special_tokens[idx].encode("utf-8")) 
            else:
                raise ValueError(f"Invalid token id: {idx}") 
        text_bytes = b"".join(part_bytes)
        text = text_bytes.decode("utf-8", errors="replace")
        return text 
    
    def _encode_chunk(self, text_bytes):
        # return the token ids 
        # let's begin... first, convert all bytes to integers in range 0...255 
        ids = list(text_bytes)
        while len(ids) >= 2:
            # find the pair with the lowest merge index 
            stats = get_stats(ids)
            pair = min(stats, key=lambda p: self.merges.get(p, float('inf')))