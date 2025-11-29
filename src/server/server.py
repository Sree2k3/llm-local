# src/server/server.py
from fastapi import FastAPI
import uvicorn
import torch
from pydantic import BaseModel
from src.model.transformer import GPTLite

app = FastAPI()
MODEL = None
CFG = None
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class GenRequest(BaseModel):
    prompt_ids: list
    max_new_tokens: int = 32


@app.on_event("startup")
def load_model():
    global MODEL, CFG
    try:
        ckpt = torch.load("checkpoints/ckpt_last.pth", map_location=DEVICE)
        CFG = ckpt.get("config", None)
        if CFG is None:
            raise RuntimeError("No config in checkpoint.")
        MODEL = GPTLite(
            CFG["vocab_size"],
            CFG["seq_len"],
            CFG["n_layer"],
            CFG["n_head"],
            CFG["d_model"],
            CFG["ff_dim"],
        ).to(DEVICE)
        MODEL.load_state_dict(ckpt["model_state"])
        MODEL.eval()
        print("Model loaded.")
    except Exception as e:
        print("No model loaded:", e)


@app.post("/generate")
async def generate(req: GenRequest):
    global MODEL, CFG
    if MODEL is None:
        return {"error": "Model not loaded."}
    input_ids = torch.tensor(req.prompt_ids, dtype=torch.long, device=DEVICE).unsqueeze(
        0
    )
    out_ids = input_ids.tolist()[0]
    with torch.no_grad():
        for _ in range(req.max_new_tokens):
            logits = MODEL(input_ids)
            next_token = logits[:, -1, :].argmax(-1, keepdim=True)
            out_ids.append(next_token.item())
            input_ids = torch.cat([input_ids, next_token], dim=1)
    return {"ids": out_ids}


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
