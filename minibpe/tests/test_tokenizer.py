import pytest 
import tiktoken 
import os 

from minibpe import BasicsicTokenizer, GPT4Tokenizer, RegexTokenizer 

# -------------------------------------------------------------------------------
# common test data 

# a few strings to test the tokenizers on 