# 🧠 MiniBPE Tokenizer

**MiniBPE** is a clean and minimal byte-level Byte Pair Encoding (BPE) tokenizer implementation, built with clarity and modularity in mind. It supports training from raw text, encoding/decoding operations, and compatibility with GPT-4-style tokenization logic.

---

## 🔍 About

Byte Pair Encoding is a subword tokenization algorithm foundational to many large language models like GPT, LLaMA, and Mistral. This tokenizer operates at the UTF-8 byte level, which ensures language-agnostic consistency and allows it to handle arbitrary text including emojis and Unicode.

Inspired by the practices in [GPT-2](https://github.com/openai/gpt-2) and later [tiktoken](https://github.com/openai/tiktoken), `minibpe` offers both a **pure BPE tokenizer** and an **enhanced regex-based variant** similar to what’s used in GPT-4.

---

```
minibpe/
│
├── minibpe/
│   ├── __init__.py       # Main importable interface for all tokenizers
│   ├── base.py           # Shared base class: encode/decode I/O + save/load utils
│   ├── basic.py          # Simple BPE tokenizer (trainable)
│   ├── gpt4.py           # GPT-4-compatible tokenizer (rank-based)
│   ├── regex.py          # Regex-based tokenizer with pre-tokenization splitting
│
├── tests/
│   ├── test_tokenizer.py      # Tests for the functionalities 
│
├── train.py              # Optional script for training and saving tokenizers
├── requirements.txt
└── README.md
```

---

## ✨ Features

- 🔠 Byte-level BPE encoding
- 🧩 Regex-based token splitting (GPT-style)
- 🔄 Trainable on any corpus
- 🧠 GPT-4-style rank-based encoding
- 💾 Save/load model vocab and merges
- 🧪 Fully tested with `pytest`

---

## ⚡ Quick Start

### Basic Example

```python
from minibpe import BasicTokenizer

text = "aaabdaaabac"
tokenizer = BasicTokenizer()
tokenizer.train(text, vocab_size=259)  # 256 bytes + 3 merges
tokens = tokenizer.encode(text)
print(tokens)
# → [258, 100, 258, 97, 99]
print(tokenizer.decode(tokens))
# → "aaabdaaabac"
```

---

## 🧠 Tokenizer Modules

### `base.py`

Defines `BaseTokenizer`:
- Abstract interface for training, encoding, decoding
- Includes I/O functions like `save()` and `load()`
- Provides token-id handling and safety checks

---

### `basic.py`

Implements `BasicTokenizer`:
- Minimal BPE training on raw byte-level text
- Merges most frequent pairs iteratively
- Good for research, small models, and experimentation

---

### `regex.py`

Implements `RegexTokenizer`:
- Splits text into pre-tokens via Unicode-aware regex
- Prevents merges across boundaries (e.g., letters/numbers/emojis)
- Supports adding special tokens like `<|endoftext|>`

```python
from minibpe import RegexTokenizer

tokenizer = RegexTokenizer()
tokenizer.train("some large dataset...", vocab_size=32768)
tokenizer.register_special_tokens({"<|endoftext|>": 32768})
tokens = tokenizer.encode("<|endoftext|>hello", allowed_special="all")
```

---

### `gpt4.py`

Implements `GPT4Tokenizer`:
- Mimics GPT-4's behavior using external `.tiktoken` rank files
- Wraps around `RegexTokenizer` and enforces exact token splits
- Compatible with OpenAI’s `cl100k_base` format

```python
from minibpe import GPT4Tokenizer

tokenizer = GPT4Tokenizer(rank_file="cl100k_base.tiktoken")
print(tokenizer.encode("hello123!!!? (안녕하세요!) 😉"))
```

---

## 🧪 Testing

Tests are written with `pytest`. To run them:

```bash
pip install pytest
pytest -v
```

Each module has corresponding unit tests:
- Base functionality
- BPE merge logic
- Regex boundaries
- Special token behavior
- GPT-4 compatibility (if rank file is provided)

---

## 🏗️ Training a Custom Tokenizer

```python
from minibpe import RegexTokenizer

tokenizer = RegexTokenizer()
tokenizer.train(open("my_corpus.txt").read(), vocab_size=8192)
tokenizer.save("my_tokenizer.model")
```

You can later load and use the trained model:

```python
tokenizer.load("my_tokenizer.model")
tokens = tokenizer.encode("this is a test")
text = tokenizer.decode(tokens)
```

---

## ⚙️ Special Tokens

Special tokens must be registered explicitly and must follow your vocabulary range:

```python
tokenizer.register_special_tokens({
    "<|endoftext|>": 8192,
    "<|pad|>": 8193,
})
```

They’re ignored unless explicitly allowed during encoding:

```python
tokenizer.encode("<|pad|>hello", allowed_special=["<|pad|>"])
```

---

## 📚 References

- [GPT-2: Language Models Are Unsupervised Multitask Learners](https://d4mucfpksywv.cloudfront.net/better-language-models/language_models_are_unsupervised_multitask_learners.pdf)
- [Tiktoken: OpenAI's Tokenizer Library](https://github.com/openai/tiktoken)
- [Sennrich et al. (2015) - BPE for NLP](https://arxiv.org/abs/1508.07909)

---

## 📄 License

MIT License

---

## 🙏 Acknowledgments

- Inspired by the tokenizer design of OpenAI's GPT family.
- Regex rules adapted from the tiktoken tokenizer.
- Developed for educational clarity, tokenizer research, and LLM prototyping.