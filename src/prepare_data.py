# src/prepare_data.py
import sentencepiece as spm
import argparse
import os
import torch
from tqdm import tqdm


def encode_file(sp_model, input_path, seq_len):
    sp = spm.SentencePieceProcessor()
    sp.Load(sp_model)
    ids = []
    with open(input_path, "r", encoding="utf-8", errors="ignore") as f:
        for line in tqdm(f, desc="Reading lines"):
            line = line.strip()
            if not line:
                continue
            toks = sp.EncodeAsIds(line)
            if len(toks) == 0:
                continue
            ids.extend(toks + [sp.eos_id()])
    # chunk into sequences (non-overlapping)
    seqs = []
    for i in range(0, len(ids) - seq_len + 1, seq_len):
        seqs.append(ids[i : i + seq_len])
    return seqs


def save_split(seqs, out_prefix, val_frac=0.02):
    n = len(seqs)
    n_val = max(1, int(n * val_frac))
    n_train = n - n_val
    train = torch.tensor(seqs[:n_train], dtype=torch.long)
    val = torch.tensor(seqs[n_train:], dtype=torch.long)
    os.makedirs(os.path.dirname(out_prefix), exist_ok=True)
    torch.save(train, out_prefix + "_train.pt")
    torch.save(val, out_prefix + "_val.pt")
    print(
        f"Saved {train.shape[0]} train sequences and {val.shape[0]} val sequences to {out_prefix}_*.pt"
    )


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--sp_model", type=str, default="data/tokenizer/spm.model")
    p.add_argument("--input", type=str, default="data/raw/corpus.txt")
    p.add_argument("--seq_len", type=int, default=128)
    p.add_argument("--out_prefix", type=str, default="data/processed/seq")
    p.add_argument("--val_frac", type=float, default=0.02)
    args = p.parse_args()

    if not os.path.exists(args.sp_model):
        raise SystemExit(f"SentencePiece model not found: {args.sp_model}")
    if not os.path.exists(args.input):
        raise SystemExit(f"Input file not found: {args.input}")

    seqs = encode_file(args.sp_model, args.input, args.seq_len)
    if len(seqs) == 0:
        raise SystemExit("No sequences produced — check corpus length or seq_len.")
    save_split(seqs, args.out_prefix, args.val_frac)
