# src/generate_local.py  (improved)
import argparse
import os
import math
import torch
import torch.nn.functional as F

# defensive import (works whether running as module or script)
try:
    # when running as package: python -m src.generate_local
    from .tokenizer import load_tokenizer
    from .model.transformer import GPTLite
except Exception:
    # when running as script: python src/generate_local.py
    from tokenizer import load_tokenizer
    from model.transformer import GPTLite


# ---------- sampling helpers ----------
def top_k_filter(logits, k):
    """Keep top-k logits (set others to -inf). Works with shape (..., V)."""
    if k <= 0:
        return logits
    # find the threshold
    topk_vals, _ = torch.topk(logits, k)
    min_topk = topk_vals[..., -1].unsqueeze(-1)
    # use masked_fill for dtype/device compatibility:
    out = logits.clone()
    mask = logits < min_topk
    out = out.masked_fill(mask, float("-1e9"))
    return out


def sample_next(logits, temperature=1.0, top_k=0):
    if temperature != 1.0 and temperature > 0:
        logits = logits / float(temperature)
    if top_k > 0:
        logits = top_k_filter(logits, top_k)
    probs = F.softmax(logits, dim=-1)
    next_id = torch.multinomial(probs, num_samples=1)
    return next_id  # shape (batch, 1)


# ---------- generation ----------
def generate(
    model,
    tokenizer,
    prompt,
    max_new_tokens=40,
    device="cpu",
    temperature=1.0,
    top_k=0,
    deterministic=False,
):
    model.eval()

    # determine model sequence length
    max_len = None
    if hasattr(model, "seq_len") and model.seq_len:
        max_len = int(model.seq_len)
    else:
        # fallback: try to infer from positional embeddings
        if hasattr(model, "pos_emb"):
            try:
                max_len = model.pos_emb.size(1)
            except Exception:
                max_len = None

    if max_len is None:
        raise RuntimeError(
            "Cannot determine model seq_len. Ensure model.seq_len or pos_emb exists."
        )

    # encode prompt
    ids = tokenizer.encode(prompt)
    if isinstance(ids, torch.Tensor):
        ids = ids.tolist()
    if len(ids) == 0:
        # prefer BOS if available
        if hasattr(tokenizer, "bos_id"):
            ids = [tokenizer.bos_id]
        else:
            ids = [0]

    # trim to last max_len tokens
    if len(ids) > max_len:
        ids = ids[-max_len:]

    x = torch.tensor([ids], dtype=torch.long, device=device)  # (1, L)

    with torch.no_grad():
        for step in range(max_new_tokens):
            x_input = x[:, -max_len:]  # (1, <=max_len)
            logits = model(x_input)  # expected (B, T, V)
            last = logits[:, -1, :]  # (B, V)

            if deterministic:
                next_id = torch.argmax(last, dim=-1, keepdim=True)
            else:
                next_id = sample_next(last, temperature=temperature, top_k=top_k)

            x = torch.cat([x, next_id], dim=1)

    out_ids = x[0].tolist()

    # decode only the *generated* part (not including prompt)
    gen_ids = out_ids[len(ids) :] if len(out_ids) > len(ids) else []
    # best-effort: drop special tokens if tokenizer provides them
    try:
        # prefer tokenizer.decode for full list; if tokenizer has decode_ids or similar use that
        generated_text = tokenizer.decode(gen_ids)
    except Exception:
        # fallback: decode full and trim by naive method
        generated_text = tokenizer.decode(out_ids)

    return generated_text


# ---------- robust checkpoint loader ----------
def build_model_from_checkpoint(checkpoint, local_vocab_size, device):
    cfg = checkpoint.get("config", checkpoint.get("cfg", None))
    if cfg is None:
        raise RuntimeError(
            "No config found in checkpoint. Keys: " + ", ".join(list(checkpoint.keys()))
        )
    required = ["seq_len", "n_layer", "n_head", "d_model", "ff_dim"]
    for k in required:
        if k not in cfg:
            raise RuntimeError(f"Missing '{k}' in checkpoint config")
    model = GPTLite(
        local_vocab_size,
        cfg["seq_len"],
        cfg["n_layer"],
        cfg["n_head"],
        cfg["d_model"],
        cfg["ff_dim"],
    ).to(device)
    try:
        model.seq_len = int(cfg["seq_len"])
    except Exception:
        model.seq_len = None

    # choose best state key
    for candidate in ["model", "model_state", "model_state_dict", "state_dict"]:
        if candidate in checkpoint:
            return model, candidate
    possible = [k for k in checkpoint.keys() if isinstance(checkpoint[k], dict)]
    if len(possible) > 0:
        return model, possible[0]
    raise RuntimeError("No model state key found in checkpoint")


def load_weights_tolerant(model, state_dict):
    """
    Load only matching keys into model. Return list of skipped keys with shapes.
    """
    ms = model.state_dict()
    to_load = {}
    skipped = []
    for k, v in state_dict.items():
        if k in ms and hasattr(v, "size") and tuple(v.size()) == tuple(ms[k].size()):
            to_load[k] = v
        else:
            # shapes differ or key missing
            ck_shape = tuple(v.size()) if hasattr(v, "size") else None
            model_shape = tuple(ms[k].size()) if k in ms else None
            skipped.append((k, ck_shape, model_shape))
    # load partially
    model.load_state_dict(to_load, strict=False)
    return skipped


# ---------- CLI ----------
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=str, default="checkpoints/ckpt_last.pth")
    parser.add_argument("--tokenizer", type=str, default="data/tokenizer/spm.model")
    parser.add_argument("--prompt", type=str, default="Hello")
    parser.add_argument("--max_new_tokens", type=int, default=40)
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--top_k", type=int, default=0)
    parser.add_argument("--deterministic", action="store_true")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Device:", device)

    if not os.path.exists(args.tokenizer):
        raise FileNotFoundError(f"Tokenizer model not found: {args.tokenizer}")
    tokenizer = load_tokenizer(args.tokenizer)

    # support various tokenizer attribute names
    vocab_size = (
        getattr(tokenizer, "vocab_size", None)
        or getattr(tokenizer, "get_piece_size", lambda: None)()
    )
    print("Tokenizer vocab size:", vocab_size)

    print("Loading checkpoint:", args.checkpoint)
    ck = torch.load(args.checkpoint, map_location=device)

    model, state_key = build_model_from_checkpoint(ck, vocab_size, device)
    print("Using state key:", state_key)

    state_dict = ck[state_key]
    skipped = load_weights_tolerant(model, state_dict)
    if skipped:
        print(
            "Skipped loading these keys due to shape mismatch (key, ck_shape, model_shape):"
        )
        for s in skipped:
            print(" ", s)

    out = generate(
        model,
        tokenizer,
        args.prompt,
        max_new_tokens=args.max_new_tokens,
        device=device,
        temperature=args.temperature,
        top_k=args.top_k,
        deterministic=args.deterministic,
    )

    print("\n--- Generated text ---\n")
    print(out)
    print("\n----------------------\n")


if __name__ == "__main__":
    main()
