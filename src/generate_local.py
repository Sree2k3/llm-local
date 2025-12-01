# src/generate_local.py
import argparse
import torch
import math
from tokenizer import load_tokenizer
from model.transformer import GPTLite
import os
import torch.nn.functional as F


# ---------- sampling helpers ----------
def top_k_filter(logits, k):
    """Keep top-k logits (set others to -inf)"""
    if k <= 0:
        return logits
    topk_vals, topk_idx = torch.topk(logits, k)
    min_topk = topk_vals[..., -1].unsqueeze(-1)
    filtered = torch.where(
        logits < min_topk, torch.tensor(-1e9, device=logits.device), logits
    )
    return filtered


def sample_next(logits, temperature=1.0, top_k=0):
    logits = logits / (temperature if temperature > 0 else 1.0)
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
    """
    Robust generation using a sliding window so model always receives inputs
    with length <= model.seq_len (positional embeddings length).
    """
    model.eval()

    # get seq_len from model (set earlier when building model)
    max_len = None
    if hasattr(model, "seq_len") and model.seq_len:
        max_len = int(model.seq_len)
    else:
        # try to infer from positional embeddings if available
        if hasattr(model, "pos_emb"):
            try:
                max_len = model.pos_emb.size(1)
            except Exception:
                max_len = None

    if max_len is None:
        raise RuntimeError(
            "Cannot determine model seq_len. Ensure model.seq_len or model.pos_emb exists."
        )

    # Encode prompt and trim to fit max_len
    ids = tokenizer.encode(prompt)
    if len(ids) == 0:
        ids = [tokenizer.bos_id] if hasattr(tokenizer, "bos_id") else [tokenizer.unk_id]

    # Trim prompt to at most max_len tokens (keep most recent tokens)
    if len(ids) > max_len:
        ids = ids[-max_len:]

    # x holds the full generated history (we will call model with sliding window)
    x = torch.tensor([ids], dtype=torch.long, device=device)  # shape (1, L)

    with torch.no_grad():
        for step in range(max_new_tokens):
            # Use only the last `max_len` tokens as model input
            x_input = x[:, -max_len:]  # (1, <=max_len)
            logits = model(x_input)  # model expects (B, T, V)
            last = logits[:, -1, :]  # (B, V)

            if deterministic:
                next_id = torch.argmax(last, dim=-1, keepdim=True)
            else:
                next_id = sample_next(
                    last, temperature=temperature, top_k=top_k
                )  # (B,1)

            x = torch.cat([x, next_id], dim=1)

    out_ids = x[0].tolist()
    return tokenizer.decode(out_ids)


# ---------- robust checkpoint loader ----------
def build_model_from_checkpoint(checkpoint, local_vocab_size, device):
    """
    Uses checkpoint['config'] to instantiate GPTLite, but overrides vocab_size
    with local_vocab_size. Returns model and the checkpoint state key used.
    Also attaches model.seq_len for downstream trimming.
    """
    cfg = checkpoint.get("config", checkpoint.get("cfg", None))
    if cfg is None:
        raise RuntimeError(
            "No config found in checkpoint. Keys: " + ", ".join(list(checkpoint.keys()))
        )
    # ensure required keys are present
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

    # attach seq_len to model so caller can access it safely
    try:
        model.seq_len = int(cfg["seq_len"])
    except Exception:
        # fallback: try to infer from position embeddings if present later
        model.seq_len = None

    # find best state key
    for candidate in ["model", "model_state", "model_state_dict", "state_dict"]:
        if candidate in checkpoint:
            return model, candidate

    # fallback: maybe entire checkpoint is state-dict
    possible = [k for k in checkpoint.keys() if isinstance(checkpoint[k], dict)]
    if len(possible) > 0:
        return model, possible[0]
    raise RuntimeError("No model state key found in checkpoint")


def load_weights_tolerant(model, state_dict):
    """
    Copy over only weights that match in shape. Skip incompatible keys (usually embeddings/head).
    """
    model_state = model.state_dict()
    matched = {}
    skipped = []
    for k, v in state_dict.items():
        if k in model_state and v.size() == model_state[k].size():
            matched[k] = v
        else:
            skipped.append(
                (
                    k,
                    tuple(v.size()) if hasattr(v, "size") else None,
                    tuple(model_state[k].size()) if k in model_state else None,
                )
            )
    model_state.update(matched)
    model.load_state_dict(model_state)
    return skipped


# ---------- CLI ----------
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=str, default="checkpoints/ckpt_last.pth")
    parser.add_argument(
        "--tokenizer",
        type=str,
        default="data/tokenizer/spm.model",
        help="path to sentencepiece .model file (not directory)",
    )
    parser.add_argument("--prompt", type=str, default="Hello")
    parser.add_argument("--max_new_tokens", type=int, default=40)
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--top_k", type=int, default=0)
    parser.add_argument(
        "--deterministic", action="store_true", help="use argmax instead of sampling"
    )
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Device:", device)

    # load tokenizer
    if not os.path.exists(args.tokenizer):
        raise FileNotFoundError(f"Tokenizer model not found: {args.tokenizer}")
    tokenizer = load_tokenizer(args.tokenizer)
    print("Tokenizer vocab size:", tokenizer.vocab_size)

    # load checkpoint
    print("Loading checkpoint:", args.checkpoint)
    ck = torch.load(args.checkpoint, map_location=device)

    # build model (using local tokenizer vocab size)
    model, state_key = build_model_from_checkpoint(ck, tokenizer.vocab_size, device)
    print("Using state key:", state_key)

    # attempt to load state, tolerantly
    state_dict = ck[state_key]
    skipped = load_weights_tolerant(model, state_dict)
    if skipped:
        print(
            "Skipped loading these keys due to shape mismatch (key, ck_shape, model_shape):"
        )
        for s in skipped:
            print("  ", s)

    # generate
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
