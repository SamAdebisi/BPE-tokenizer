"""Train our Tokenizer on some data, just to see them in action.
The whole thing runs in ~15 seconds on my laptop.
"""

import os 
import time 

from minibpe import BasicTokenizer, GPT4Tokenizer, RegexTokenizer 

# open some text and train a vocab of 512 tokens 
text = open("minibpe/tests/taylorswift.txt", "r", encoding="utf-8").read()

# create a directory for models, so we don't pollute the current directory 
os.makedirs("models", exist_ok=True)

t0 = time.time()
for TokenizerClass, name in zip([BasicTokenizer, RegexTokenizer], ["basic", "regex"]):
    
    # construct the Tokenizer object and kick off verbose training 
    tokenizer = TokenizerClass()
    tokenizer.train(text, vocab_size=512, verbose=True)
    # writes two files in the models directory: name.model, and name.vocab 
    prefix = os.path.join("models", name)
    tokenizer.save(prefix)
t1 = time.time()

print(f"Training took {t1 - t0:.2f} seconds.")
