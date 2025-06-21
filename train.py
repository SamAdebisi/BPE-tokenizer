"""Train our Tokenizer on some data, just to see them in action.
The whole thing runs in ~?? seconds on my laptop.
"""

import os 
import time 
from minibpe import BasicsicTokenizer, GPT4Tokenizer, RegexTokenizer 

# open some text and train a vocab of 512 tokens 
text = open("tests/taylorswift.txt", "r", encoding="utf-8").read()

# create a directory for models, so we don't pollute the current directory 
os.makedirs("models", exist_ok=True)

t0 = time.time()
for TokenizerClass, name in zip([BasicsicTokenizer, GPT4Tokenizer, RegexTokenizer], ["basic", "gpt4", "regex"]):
    
    # construct the Tokenizer object and kick off verbose training 
    tokenizer = TokenizerClass()
    tokenizer.train(text, vocab_size=512, verbose=True)