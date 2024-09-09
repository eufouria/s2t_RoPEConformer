from datasets import load_dataset
from tokenizers import Tokenizer, models, pre_tokenizers, decoders, trainers

dataset = load_dataset("librispeech_asr", "clean")

# Extract the text labels
texts = [example["text"] for example in dataset["train"]]
tokenizer = Tokenizer(models.WordPiece(unk_token="[UNK]"))

tokenizer.pre_tokenizer = pre_tokenizers.Sequence([
    pre_tokenizers.Whitespace(),  # Split on whitespace
    pre_tokenizers.Punctuation(),  
])

tokenizer.decoder = decoders.WordPiece()
trainer = trainers.WordPieceTrainer(
    vocab_size=30000,  
    special_tokens=["[UNK]", "[CLS]", "[SEP]", "[PAD]", "[MASK]"],
)

tokenizer.train_from_iterator(texts, trainer=trainer)
tokenizer.save("tokenizer.json")
