import os
import torch
from torch.utils.data import Dataset
from datasets import load_dataset
from tokenizers import ByteLevelBPETokenizer
from tqdm import tqdm

# paths
TOKENIZER_PATH = "../tokenize"
LOCAL_DIR = "../data/local_texts"

class TextDataset(Dataset):
    def __init__(self, tokenizer_path=TOKENIZER_PATH, seq_len=512, num_docs=20000):
        print("🔤 loading tokenizer...")
        self.tokenizer = ByteLevelBPETokenizer(
            os.path.join(tokenizer_path, "vocab.json"),
            os.path.join(tokenizer_path, "merges.txt")
        )

        print("📚 loading local + wikitext2 data...")
        self.texts = self._load_texts(num_docs)
        print(f"loaded {len(self.texts)} documents")

        print("🧩 tokenizing and chunking...")
        self.tokens = self._tokenize_and_chunk(seq_len)
        print(f"created {len(self.tokens)} training chunks (sequences of length {seq_len})")


    def _load_texts(self, num_docs):
        texts = []
        
        # local .txt files
        for root, _, files in os.walk(LOCAL_DIR):
            for f in files:
                if f.endswith(".txt"):
                    with open(os.path.join(root, f), "r", encoding="utf-8") as src:
                        texts.append(src.read())

        # wikitext-2 dataset (used for evaluation/generalization)
        wiki = load_dataset("wikitext", "wikitext-2-raw-v1", split="train")
        wiki_train = [x["text"] for x in wiki if len(x["text"]) > 0]
        texts.extend(wiki_train[:num_docs])

        return texts

    def _tokenize_and_chunk(self, seq_len):
        """Encodes all texts and splits the resulting stream into fixed-length chunks."""
        all_tokens = []
        for text in tqdm(self.texts, desc="tokenizing"):
            # Encode text to list of token IDs
            ids = self.tokenizer.encode(text).ids
            
            # Slide a window of size seq_len across the token IDs
            for i in range(0, len(ids) - seq_len, seq_len):
                chunk = ids[i : i + seq_len]
                all_tokens.append(torch.tensor(chunk, dtype=torch.long))
        return all_tokens

    def __len__(self):
        return len(self.tokens)

    def __getitem__(self, idx):
        # The key for next token prediction:
        # Input (x) is tokens [0] to [T-1]
        # Target (y) is tokens [1] to [T] (shifted by one)
        data = self.tokens[idx]
        return data[:-1], data[1:] # input, target