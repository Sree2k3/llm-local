# src/tokenizer.py
import sentencepiece as spm
import os


def train_sentencepiece(
    input_path: str,
    model_prefix: str = "data/tokenizer/spm",
    vocab_size: int = 1500,
    model_type: str = "bpe",
    character_coverage: float = 1.0,
    pad_id: int = 0,
    unk_id: int = 1,
    bos_id: int = 2,
    eos_id: int = 3,
):
    """
    Train SentencePiece model and save to model_prefix.model / .vocab
    """
    os.makedirs(os.path.dirname(model_prefix), exist_ok=True)
    args = (
        f"--input={input_path} --model_prefix={model_prefix} --vocab_size={vocab_size} "
        f"--model_type={model_type} --character_coverage={character_coverage} "
        f"--pad_id={pad_id} --unk_id={unk_id} --bos_id={bos_id} --eos_id={eos_id}"
    )
    spm.SentencePieceTrainer.Train(args)


class SpmTokenizer:
    """
    Thin wrapper around SentencePieceProcessor with helpful methods.
    """

    def __init__(self, model_file: str):
        if not os.path.exists(model_file):
            raise FileNotFoundError(f"SentencePiece model not found: {model_file}")
        self.sp = spm.SentencePieceProcessor()
        self.sp.Load(model_file)
        # expose a stable attribute
        self.vocab_size = self.sp.get_piece_size()
        # helpful ids
        try:
            self.pad_id = self.sp.piece_to_id("<pad>")
            self.unk_id = self.sp.piece_to_id("<unk>")
            self.bos_id = self.sp.piece_to_id("<s>")
            self.eos_id = self.sp.piece_to_id("</s>")
        except Exception:
            # if tokens don't exist, fallback to defaults (may be 0..3)
            self.pad_id = 0
            self.unk_id = 1
            self.bos_id = 2
            self.eos_id = 3

    # encode a text into list of ids
    def encode(self, text: str):
        return self.sp.EncodeAsIds(text)

    # decode id list to text
    def decode(self, ids):
        return self.sp.DecodeIds(ids)

    # convenience: encode many lines to a flat id list with eos between lines
    def encode_lines(self, lines, append_eos=True):
        out = []
        for ln in lines:
            ids = self.encode(ln)
            if len(ids) == 0:
                continue
            out.extend(ids)
            if append_eos:
                out.append(self.eos_id)
        return out


def load_tokenizer(model_path: str = "data/tokenizer/spm.model"):
    """
    Returns an SpmTokenizer instance.
    Use explicit model file path (not directory).
    """
    return SpmTokenizer(model_path)
