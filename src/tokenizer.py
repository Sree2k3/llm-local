# src/tokenizer.py
import sentencepiece as spm
import os


def train_sentencepiece(
    input_files, model_prefix="data/tokenizer/spm", vocab_size=20000, model_type="bpe"
):
    os.makedirs(os.path.dirname(model_prefix), exist_ok=True)
    concat_path = os.path.join(os.path.dirname(model_prefix), "input_for_spm.txt")
    with open(concat_path, "w", encoding="utf-8") as fw:
        for p in input_files:
            with open(p, "r", encoding="utf-8", errors="ignore") as fr:
                for line in fr:
                    line = line.strip()
                    if line:
                        fw.write(line + "\n")
    spm.SentencePieceTrainer.Train(
        f"--input={concat_path} --model_prefix={model_prefix} --vocab_size={vocab_size} --model_type={model_type} --character_coverage=1.0"
    )
    print(f"Trained sentencepiece model: {model_prefix}.model / .vocab")


if __name__ == "__main__":
    raw_dir = "data/raw"
    files = [
        os.path.join(raw_dir, f) for f in os.listdir(raw_dir) if f.endswith(".txt")
    ]
    if not files:
        print("No input .txt files in data/raw. Add some and rerun.")
    else:
        train_sentencepiece(files, model_prefix="data/tokenizer/spm", vocab_size=20000)
