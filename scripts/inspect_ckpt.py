import torch

ck = "checkpoints/ckpt_last.pth"
d = torch.load(ck, map_location="cpu")
# try a few keys that often store state_dict
if "model_state" in d:
    sd = d["model_state"]
else:
    sd = d
for k, v in sd.items():
    try:
        print(k, getattr(v, "shape", None))
    except Exception:
        pass
