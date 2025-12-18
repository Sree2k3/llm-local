# LLM-LOCAL PROJECT REFERENCE

Goal:
Build a GPT-style language model locally from scratch (tokenizer → data → model → training → generation).

Key components:
1. SentencePiece tokenizer (BPE)
2. GPTLite decoder-only Transformer
3. Fixed-length causal language modeling
4. PyTorch training with AMP
5. Autoregressive text generation

Important lessons:
- Tokenizer vocab_size MUST match model vocab_size
- seq_len must match data and positional embeddings
- Low loss (~0.003) can indicate overfitting/collapse
- Small models repeat tokens if undertrained or overfit
- Temperature/top-k only help if model is healthy
- Windows requires Python scripts, not bash heredocs

Future improvements:
- Longer training (50k+ steps)
- LR warmup
- Slightly larger model
- Instruction-style data
