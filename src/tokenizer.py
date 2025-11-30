# src/tokenizer.py
import sentencepiece as spm
import argparse
import os


def train_sentencepiece(
    input_path, model_prefix="data/tokenizer/spm", vocab_size=5000, model_type="bpe"
):
    os.makedirs(os.path.dirname(model_prefix), exist_ok=True)
    spm.SentencePieceTrainer.Train(
        f"--input={input_path} "
        f"--model_prefix={model_prefix} "
        f"--vocab_size={vocab_size} "
        f"--model_type={model_type} "
        f"--character_coverage=1.0 "
        f"--pad_id=0 --unk_id=1 --bos_id=2 --eos_id=3"
    )
    print(
        f"Tokenizer trained successfully → {model_prefix}.model and {model_prefix}.vocab"
    )


def load_tokenizer(model_path):
    sp = spm.SentencePieceProcessor()
    sp.load(model_path)
    return sp


def decode_tokens(sp, ids):
    return sp.decode_ids(ids)


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--input", type=str, default="data/raw/corpus.txt")
    p.add_argument("--vocab_size", type=int, default=5000)
    p.add_argument("--model_prefix", type=str, default="data/tokenizer/spm")
    p.add_argument("--model_type", type=str, default="bpe")
    args = p.parse_args()

    if not os.path.exists(args.input):
        raise SystemExit(f"❌ Input file not found: {args.input}")

    train_sentencepiece(
        args.input,
        model_prefix=args.model_prefix,
        vocab_size=args.vocab_size,
        model_type=args.model_type,
    )
