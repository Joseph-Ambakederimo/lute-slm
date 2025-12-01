import os
from datasets import load_dataset
from tokenizers import ByteLevelBPETokenizer

# Define paths relative to the project root
LOCAL_DIR = "../data/local_texts"
SAVE_PATH = "../tokenize" 
os.makedirs(SAVE_PATH, exist_ok=True)

def get_local_files(local_dir):
    """Recursively finds all .txt files in the local data directory."""
    txt_files = []
    for root, _, files in os.walk(local_dir):
        for f in files:
            if f.endswith(".txt"):
                txt_files.append(os.path.join(root, f))
    return txt_files

def prepare_corpus():
    """Combines local and HuggingFace data into a single corpus file for tokenizer training."""
    # 1. Collect all local .txt files
    local_files = get_local_files(LOCAL_DIR)
    print(f"found {len(local_files)} local text files")

    # 2. Load wikitext-2 from Hugging Face
    wiki = load_dataset("wikitext", "wikitext-2-raw-v1", split="train")
    wiki_train = [x["text"] for x in wiki if len(x["text"]) > 0]

    # 3. Write combined corpus
    combined_path = "./corpus.txt"
    with open(combined_path, "w", encoding="utf-8") as f:
        for file_path in local_files:
            with open(file_path, "r", encoding="utf-8") as src:
                f.write(src.read() + "\n")
        
        # Add WikiText lines
        for line in wiki_train:
            f.write(line + "\n")

    print(f"combined corpus saved to {combined_path}")
    return combined_path

def train_tokenizer(corpus_path):
    """Trains a ByteLevelBPE Tokenizer."""
    tokenizer = ByteLevelBPETokenizer()
    tokenizer.train(
        files=[corpus_path],
        vocab_size=50000, # Should match ModelConfig.vocab_size
        min_frequency=2,
        special_tokens=["<s>", "<pad>", "</s>", "<unk>", "<mask>"]
    )
    # The tokenizer saves vocab.json and merges.txt inside the SAVE_PATH
    tokenizer.save_model(SAVE_PATH)
    print(f"✅ tokenizer saved to {SAVE_PATH}")

if __name__ == "__main__":
    # Ensure the data directory exists for the combined corpus file
    os.makedirs("./data", exist_ok=True)
    # You must manually ensure ../data/local_texts contains some .txt files or create it.
    corpus = prepare_corpus()
    train_tokenizer(corpus)