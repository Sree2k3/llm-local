import torch

train = torch.load("data/processed/seq_train.pt")
val = torch.load("data/processed/seq_val.pt")

print("train shape:", train.shape)
print("val shape:", val.shape)
