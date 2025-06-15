"""
    Minimal (byte-level) Byte Pair Encoding tokenizer.
    
    Algorithmically follows along the GPT tokenizer:
https://github.com/openai/gpt-2/blob/master/src/encoder.py

But:
- Does not handle the regular expression splitting pattern.
- Does not handle any special tokens.
    """
    