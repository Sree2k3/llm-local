# src/train.py
import torch
import json
import argparse
import os
from src.model.transformer import GPTLite
from src.data_pipeline import SimpleDataset


def load_config(path):
    return json.load(open(path))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="configs/small.json")
    parser.add_argument("--batch", type=int, default=1)
    parser.add_argument("--seq_len", type=int, default=None)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--use_amp", action="store_true")
    parser.add_argument("--steps", type=int, default=20)
    parser.add_argument(
        "--data_files",
        type=str,
        default=None,
        help="comma separated paths to .txt files",
    )
    args = parser.parse_args()

    cfg = load_config(args.config)
    vocab_size = cfg["vocab_size"]
    seq_len = args.seq_len or cfg["seq_len"]
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")

    model = GPTLite(
        vocab_size,
        seq_len,
        cfg["n_layer"],
        cfg["n_head"],
        cfg["d_model"],
        cfg["ff_dim"],
        cfg.get("dropout", 0.1),
    ).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4)

    scaler = torch.cuda.amp.GradScaler(enabled=args.use_amp)
    model.train()

    if args.data_files:
        files = args.data_files.split(",")
        dataset = SimpleDataset(
            files[0], files, seq_len
        )  # files[0] arg unused in class; keeping signature
        it = dataset.iterate()
    else:
        it = None

    for step in range(args.steps):
        if it is None:
            x = torch.randint(
                0, vocab_size, (args.batch, seq_len), dtype=torch.long, device=device
            )
            y = x.clone()
        else:
            try:
                x = next(it).unsqueeze(0).to(device)
                y = x.clone()
            except StopIteration:
                print("Dataset exhausted.")
                break

        optimizer.zero_grad()
        with torch.cuda.amp.autocast(enabled=args.use_amp):
            logits = model(x)
            loss = torch.nn.functional.cross_entropy(
                logits.view(-1, logits.size(-1)), y.view(-1)
            )
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()

        if step % 1 == 0:
            print(f"step {step} loss {loss.item():.4f}")

    os.makedirs("checkpoints", exist_ok=True)
    ckpt = {"model_state": model.state_dict(), "config": cfg}
    torch.save(ckpt, "checkpoints/ckpt_last.pth")
    print("Saved checkpoint to checkpoints/ckpt_last.pth")


if __name__ == "__main__":
    main()
