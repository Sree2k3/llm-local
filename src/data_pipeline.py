# src/data_pipeline.py
import sentencepiece as spm
import os
import torch


class SimpleDataset:
    def __init__(self, sp_model_path, file_list, seq_len=128):
        self.sp = spm.SentencePieceProcessor()
        self.sp.Load(sp_model_path)
        self.file_list = file_list
        self.seq_len = seq_len

    def iterate(self):
        for f in self.file_list:
            with open(f, "r", encoding="utf-8", errors="ignore") as fr:
                for line in fr:
                    line = line.strip()
                    if not line:
                        continue
                    ids = self.sp.EncodeAsIds(line)
                    if len(ids) < 2:
                        continue
                    for i in range(0, len(ids), self.seq_len):
                        chunk = ids[i : i + self.seq_len]
                        if len(chunk) < 2:
                            continue
                        if len(chunk) < self.seq_len:
                            chunk = chunk + [0] * (self.seq_len - len(chunk))
                        yield torch.tensor(chunk, dtype=torch.long)
