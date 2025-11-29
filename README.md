<h1 align="center">🚀 MiniGPT — Train Your Own GPT Model from Scratch</h1>

<p align="center">
  <img src="https://img.shields.io/badge/Python-3.10-blue?logo=python" />
  <img src="https://img.shields.io/badge/PyTorch-2.5.1-red?logo=pytorch" />
  <img src="https://img.shields.io/badge/CUDA-12.9-green?logo=nvidia" />
  <img src="https://img.shields.io/badge/VS_Code-Project-blue?logo=visualstudiocode" />
  <img src="https://img.shields.io/badge/Model-GPT%20from%20Scratch-orange?logo=openai" />
  <img src="https://img.shields.io/badge/Device-GTX%201650-lightgrey?logo=nvidia" />
  <img src="https://img.shields.io/badge/Training-AMP%20Enabled-yellow?logo=lightning" />
  <img src="https://img.shields.io/badge/Status-Active-success?logo=github" />
</p>

---

## 📌 Project Overview

MiniGPT is a **fully-custom, from-scratch GPT-style language model** implemented using **PyTorch**, trained locally on an **NVIDIA GTX 1650** GPU.  
This project demonstrates a deep understanding of:

- Transformer architectures  
- Tokenization (SentencePiece)  
- Multi-head self-attention  
- Positional embeddings  
- Training pipelines  
- Mixed-precision (AMP)  
- Checkpointing  
- Inference serving  

Everything is coded manually—**no HuggingFace model classes**.

---

## 📂 Repository Structure
llm-local/
│
├── configs/
│ └── small.json # Model config
│
├── data/
│ ├── raw/ # Raw .txt files for tokenizer + training
│ ├── processed/ # Preprocessed binary/token files
│ └── tokenizer/ # SentencePiece tokenizer outputs
│
├── scripts/
│ ├── preprocess.bat # Tokenizer training script
│ └── run_train.bat # Training runner
│
├── src/
│ ├── model/
│ │ └── transformer.py # Full GPT model from scratch
│ ├── server/
│ │ └── server.py # FastAPI inference server
│ ├── data_pipeline.py # Dataset + dataloader
│ ├── tokenizer.py # SentencePiece tokenizer builder
│ └── train.py # Training loop + AMP + checkpoints
│
└── requirements.txt


---

## 🧠 Model Architecture

- **Embedding Layer**  
- **Positional Encoding**  
- **N Transformer Blocks**
  - Multi-head attention  
  - Feed-forward MLP  
  - LayerNorm  
  - Residual connections  
- **Language Modeling Head (LM Head)**  

Configurable via `configs/small.json`:

```json
{
    "vocab_size": 20000,
    "seq_len": 128,
    "n_layer": 6,
    "n_head": 6,
    "d_model": 384,
    "ff_dim": 1536
}
