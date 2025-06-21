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
    # we do this because `pytest -v.` prints the arguments to console, and we don't 
    # want to print the entire contents of the file, it creates a mess. So here we go. 
    if text.startswith("FILE:"):
        dirname = os.path.dirname(os.path.abspath(__file__))
        taylorswift_file = os.path.join(dirname, text[5:])
        contents = open(taylorswift_file, "r", encoding="utf-8").read()
        return contents  
    else:
        return text 

special_string = """
<|endoftext|>
<|endoftext|>
<|endoftext|>
<|endoftext|>
"""