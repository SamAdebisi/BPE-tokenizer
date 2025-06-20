"""
    Implements the GPT-4 Tokenizer as a light wrapper around the RegexTokenizer. 
    Note that this is a pretrained tokenizer. By default and inside init(), it loads 
    the pretrained tokenizer from the `cl100k_base` tokenizer of tiktoken. 
"""

import tiktoken´
from .regex import RegexTokenizer 

