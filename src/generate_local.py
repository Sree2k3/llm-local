import torch
import argparse
from tokenizer import load_tokenizer, decode_tokens
from model.transformer import GPTLite


def generate_text(model, tokenizer, prompt, max_new_tokens, device):
    model.eval()
    tokens = tokenizer.encode(prompt)
    tokens = torch.tensor(tokens, dtype=torch.long).unsqueeze(0).to(device)

    for _ in range(max_new_tokens):
        with torch.no_grad():
            logits = model(tokens)
            next_token = torch.argmax(logits[:, -1, :], dim=-1)
            tokens = torch.cat([tokens, next_token.unsqueeze(0)], dim=1)

    return decode_tokens(tokenizer, tokens[0].tolist())


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--prompt", type=str, default="Hello")
    parser.add_argument("--max_new_tokens", type=int, default=50)
    parser.add_argument("--checkpoint", type=str, default="checkpoints/ckpt_last.pth")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    tokenizer = load_tokenizer("data/tokenizer/spm.model")
    vocab_size = tokenizer.get_piece_size()

    # load checkpoint


checkpoint = torch.load(args.checkpoint, map_location=device)
cfg = checkpoint.get("config", checkpoint.get("cfg", None))
if cfg is None:
    print("Checkpoint keys:", list(checkpoint.keys()))
    raise SystemExit("Could not find config inside checkpoint.")

# build model with local vocab size (already done earlier)
model = GPTLite(
    vocab_size,
    cfg["seq_len"],
    cfg["n_layer"],
    cfg["n_head"],
    cfg["d_model"],
    cfg["ff_dim"],
).to(device)

# pick state dict key
if "model_state" in checkpoint:
    state_key = "model_state"
elif "model" in checkpoint:
    state_key = "model"
elif "state_dict" in checkpoint:
    state_key = "state_dict"
else:
    print("Checkpoint keys:", list(checkpoint.keys()))
    raise SystemExit("No recognized model state key in checkpoint.")

ck_state = checkpoint[state_key]
model_state = model.state_dict()

# filter and copy only matching shapes
matched = {}
for k, v in ck_state.items():
    if k in model_state and v.size() == model_state[k].size():
        matched[k] = v
    else:
        print(
            f"Skipping {k}: checkpoint {tuple(v.size())} != model {tuple(model_state[k].size())}"
            if k in model_state
            else f"Key {k} not in model"
        )

# update model_state and load
model_state.update(matched)
model.load_state_dict(model_state)
model.eval()


if __name__ == "__main__":
    main()
