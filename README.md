# Transformer from Scratch with TorchText and HuggingFace Tokenizers

This repository contains an implementation of a **Transformer model** from scratch using **PyTorch**, **TorchText**, and **HuggingFace’s CharBPETokenizer**. The notebook demonstrates preprocessing, tokenization, positional encoding, multi-head attention, encoder-decoder layers, and a working Transformer forward pass on a small dataset.

---

## Table of Contents
- [Overview](#overview)
- [Installation](#installation)
- [Dataset](#dataset)
- [Tokenization](#tokenization)
- [Model Architecture](#model-architecture)
- [Usage](#usage)
- [References](#references)

---

## Overview
This project demonstrates how to implement a Transformer model for sequence-to-sequence tasks, such as machine translation. The notebook covers:

- Loading the **Multi30k dataset** for German-to-English translation.
- Tokenizing data using **HuggingFace CharBPETokenizer**.
- Defining **positional encoding**, **multi-head attention**, and **feed-forward layers**.
- Building **encoder** and **decoder layers**.
- Creating a full **Transformer model**.
- Sample forward pass through the Transformer.

---

## Installation
Install the required Python packages using `pip`:

```bash
pip install torch==2.3.0
pip install torchtext
pip install torchdata
pip install portalocker
pip install einops
pip install tokenizers
````

> **Note:** Ensure your Python version is compatible (tested with Python 3.9). GPU is optional but recommended for faster training.

---

## Dataset

We use the **Multi30k dataset** for German-to-English translation:

```python
from torchtext.datasets import Multi30k

SRC_LANGUAGE = 'de'
TGT_LANGUAGE = 'en'

train_iter, valid_iter, test_iter = Multi30k(language_pair=(SRC_LANGUAGE, TGT_LANGUAGE))
```

The dataset contains German sentences paired with English translations.

---

## Tokenization

Tokenization is done using HuggingFace’s **CharBPETokenizer**:

* Special symbols: `<unk>`, `<pad>`, `<bos>`, `<eos>`
* Trained separately on English and German sentences.

```python
from tokenizers import CharBPETokenizer

en_tokenizer = CharBPETokenizer()
de_tokenizer = CharBPETokenizer()
```

---

## Model Architecture

### Positional Encoding

Adds positional information to the embeddings for sequence modeling.

### Multi-Head Attention

Supports two implementations:

1. Matrix multiplication with `einops`.
2. Linear projections with reshaping for attention heads.

### Feed-Forward Network

Position-wise feed-forward layers with ReLU activation.

### Encoder and Decoder Layers

* **EncoderLayer**: Multi-head self-attention + feed-forward + residual + layer norm.
* **DecoderLayer**: Masked self-attention + encoder-decoder attention + feed-forward + residual + layer norm.

### Transformer

Full Transformer combining `n` encoder layers and `n` decoder layers with embedding and output projection.

---

## Usage

```python
from transformer import Transformer
import torch

# Example forward pass
t = Transformer(d_model=8, d_hidden=8, num_heads=4, d_ff=10, src_vocab_size=3, tgt_vocab_size=7)
src, tgt = torch.arange(0, 3).tile(2, 1), torch.arange(0, 3).tile(2, 1)
output = t(src, tgt)
print(output.shape)
```

---

## References

1. [Attention Is All You Need - Vaswani et al., 2017](https://arxiv.org/abs/1706.03762)
2. [TorchText Documentation](https://pytorch.org/text/stable/index.html)
3. [HuggingFace Tokenizers](https://huggingface.co/docs/tokenizers/index)

---

## License

This project is licensed under the MIT License.
