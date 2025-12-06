import torch

ck = "checkpoints/ckpt_last.pth"
print("Loading checkpoint:", ck)

d = torch.load(ck, map_location="cpu")
sd = d.get("model_state", d)

for k, v in sd.items():
    if hasattr(v, "shape"):
        print(f"{k}: {tuple(v.shape)}")
