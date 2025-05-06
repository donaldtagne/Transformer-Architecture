# Transformer Architecture Explained

Transformers are at the core of modern AI models like GPT, BERT, T5, and LLaMA. Since the groundbreaking paper "Attention is All You Need" (Vaswani et al., 2017), they have revolutionized the field of neural networks.


## What is a Transformer?

A transformer is a neural network architecture specifically designed to process sequences (e.g., language). Unlike RNNs or LSTMs, transformers handle all input positions simultaneously (in parallel).


## Core Idea: Self-Attention

The heart of a transformer is the self-attention mechanism, which allows the model to determine which other positions in the sequence are important.

### Example:
"The cat sat on the mat because she was tired."

The model learns that "she" refers to "cat," even though the words are far apart.


## How Does Self-Attention Work?

Each input token is transformed into three vectors:

- Query (Q): What am I looking for?
- Key (K): What do I represent?
- Value (V): What information do I provide?

The transformer computes attention scores for each token:

1. Compare Query and Key (dot product)
2. Apply softmax to generate weights
3. Compute a weighted sum of all Values

This produces a new, context-aware representation for each word.


## Components of a Transformer

A typical encoder or decoder block includes:

- Multi-Head Self-Attention: multiple parallel attention heads
- Feedforward Networks: non-linear processing
- Add & Norm: residual connections and layer normalization
- Positional Encodings: add sequence order information


## Comparison to RNNs

| Aspect             | RNN                            | Transformer                         |
|--------------------|----------------------------------|-------------------------------------|
| Processing         | Sequential                     | Parallel                             |
| Context Range      | Limited (loses long-term context) | Global via self-attention           |
| Training           | Slow, hard to scale            | Fast, GPU-optimized                  |


## Applications

Text generation (GPT, T5)  
Machine translation (e.g., MarianMT)  
Question answering (BERT, RoBERTa)  
Code generation (Codex, StarCoder)  
Vision Transformer (ViT) for image tasks


## Further Resources

Paper: Attention Is All You Need: https://arxiv.org/abs/1706.03762  
Illustrated Guide: The Illustrated Transformer (Jay Alammar): https://jalammar.github.io/illustrated-transformer/  
Open source models on Hugging Face: https://huggingface.co/models


Created for developers looking to build a solid understanding of the Transformer architecture.
