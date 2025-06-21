"""Train our Tokenizer on some data, just to see them in action.
The whole thing runs in ~?? seconds on my laptop.
"""

import os 
import time 
from minibpe import BasicsicTokenizer, GPT4Tokenizer, RegexTokenizer 

# open some text and train a vocab of 512 tokens 
text = open("tests/taylorswift.txt", "r", encoding="utf-8").read()
