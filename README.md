PyTorch Transformer — English to Spanish Translation

A Transformer-based English-to-Spanish translation system built from scratch using PyTorch.
The project implements the core Encoder-Decoder Transformer architecture, providing a practical understanding of how modern sequence-to-sequence models work.

  Features
1. Custom Encoder-Decoder Transformer
2. Multi-Head Self-Attention and Cross-Attention
3. Sinusoidal Positional Encoding
4. Token Embeddings and Feed-Forward Networks
5. Residual Connections and Feature Normalization
6. Padding and Causal Attention Masks
7. Custom Word-Level Tokenizers
8. GPU, Apple MPS, and CPU support
9. Model Checkpointing and Resume Training
10. TensorBoard Training Monitoring
11. BLEU, WER, and CER Evaluation
12. Autoregressive Translation

Project Structure

dataset.py     → Dataset preparation and attention masks

arch.py     → Transformer architecture

prep.py       → Training, validation, metrics, and checkpoints

Conversion.py   → Inference and text generation

Configuration.py  → Project configuration, paths, model and training settings

README.md      → Project documentation

The dataset pipeline tokenizes English and Spanish translation pairs, prepares encoder and decoder inputs, applies padding, and generates appropriate attention masks.

The Transformer implements attention mechanisms, encoder and decoder stacks, positional encoding, and vocabulary projection. 
The training pipeline uses Adam optimization and Cross-Entropy loss with label smoothing, while TensorBoard tracks training and validation metrics.

During inference, the trained model generates Spanish translations token-by-token autoregressively, stopping when the end-of-sequence token is produced.

This project is designed for learning and experimentation with Transformers, NLP, Neural Machine Translation, and PyTorch.
