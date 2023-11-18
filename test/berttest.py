from transformers import AutoTokenizer

# Load the Mongolian BERT tokenizer
tokenizer = AutoTokenizer.from_pretrained('tugstugi/bert-base-mongolian-cased')

# Get the size of the tokenizer's vocabulary
vocab_size = tokenizer.vocab_size

print(f"The vocabulary size of the Mongolian BERT tokenizer is: {vocab_size}")

