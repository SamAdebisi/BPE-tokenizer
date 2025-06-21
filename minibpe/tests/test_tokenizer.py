import pytest 
import tiktoken 
import os 

from minibpe import BasicsicTokenizer, GPT4Tokenizer, RegexTokenizer 

# -------------------------------------------------------------------------------
# common test data 

# a few strings to test the tokenizers on 
test_strings = [
    "", # empty string 
    "?", # single character 
    "hello world!!!? (안녕하세요!) lol123 😉", # fun small string 
    "FILE:taylorswift.txt", # FILE: is handled as a special string in unpack()
]
def unpack(text):
    pass 