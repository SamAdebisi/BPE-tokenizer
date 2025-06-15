"""
    Minimal (byte-level) Byte Pair Encoding tokenizer.
    
    Algorithmically follows along the GPT tokenizer:
https://github.com/openai/gpt-2/blob/master/src/encoder.py

But:
- Does not handle the regular expression splitting pattern.
- Does not handle any special tokens.
"""

from .base import get_stats, merge, Tokenizer 


class BasicsicTokenizer:
    """
    A basic Byte Pair Encoding tokenizer.
    """

    def __init__(self):
        # the main attributes of the tokenizer
        self.merges = {}  # (int, int) -> int
        self.vocab = {}  # int -> bytes

        # attributes for decoding
        self.inverse_merges = {}  # int -> (int, int)

    def train(self, text, vocab_size, verbose=False):
        """
        Train the tokenizer on the given text.
        """
        # convert text to bytes
        text_bytes = text.encode("utf-8")
        ids = list(text_bytes)

        # populate the initial vocab
        self.vocab = {idx: bytes([idx]) for idx in range(256)}

        # perform merges
        for i in range(vocab_size - 256):
            stats = get_stats(ids)
            if not stats:
                break  # no more pairs to merge

            # find the most frequent pair
            pair = max(stats, key=lambda p: stats[p])
            idx = 256 + i

            if verbose:
                print(f"merging {pair} into {idx}")

            ids = merge(ids, pair, idx)

            # update vocab and merges
            self.merges[pair] = idx
            self.inverse_merges[idx] = pair
            self.vocab[idx] = self.vocab[pair[0]] + self.vocab[pair[1]]

    def encode(self, text):
        """
        Encode the given text into a list of integers.
        """
        text_bytes = text.encode("utf-8")
        ids = list(text_bytes)

        while True:
            stats = get_stats(ids)
            if not stats:
                break

            pair = min(stats, key=lambda p: self.merges.get(p, float('inf')))
            if pair not in self.merges:
                break  # no more pairs to merge that are in our vocabulary

            idx = self.merges[pair]
            ids = merge(ids, pair, idx)

        return ids

