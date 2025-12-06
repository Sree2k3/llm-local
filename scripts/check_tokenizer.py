from sentencepiece import SentencePieceProcessor

sp = SentencePieceProcessor()
sp.Load("data/tokenizer/spm.model")
print("Tokenizer vocab size =", sp.GetPieceSize())
