import torch
import torch.nn.functional as F
from model.model import LuteModel
from model.config import ModelConfig
from tokenizers import ByteLevelBPETokenizer

# -----------------------
# CONFIG
# -----------------------
# Ensure this matches the ModelConfig used for training
config = ModelConfig(vocab_size=50000) 
device = "cuda" if torch.cuda.is_available() else "cpu"

# -----------------------
# LOAD MODEL & CHECKPOINT
# -----------------------
model = LuteModel(config).to(device)
model.eval()

# Load checkpoint (assumes checkpoint_latest.pt exists after training)
ckpt_path = "checkpoints/checkpoint_latest.pt"
try:
    if ckpt_path:
        ckpt = torch.load(ckpt_path, map_location=device)
        model.load_state_dict(ckpt["model_state_dict"])
        print(f"♻️ loaded checkpoint from step {ckpt['step']}")
except FileNotFoundError:
    print(f"⚠️ Checkpoint not found at {ckpt_path}. Starting with untraining weights.")
    
# -----------------------
# LOAD TOKENIZER
# -----------------------
tokenizer = ByteLevelBPETokenizer(
    "tokenize/vocab.json",
    "tokenize/merges.txt"
)

# -----------------------
# GENERATION FUNCTION (with Top-K and Top-P)
# -----------------------
@torch.no_grad()
def generate(model, tokenizer, prompt, max_len=100, temperature=0.8, top_k=0, top_p=0.9, device=device):
    """Generates text from a prompt using sampling techniques."""
    model.eval()
    
    # 1. Prepare input tokens
    tokens = tokenizer.encode(prompt).ids
    input_ids = torch.tensor(tokens, dtype=torch.long, device=device).unsqueeze(0)
    
    # Define EOS token ID
    EOS_TOKEN = tokenizer.token_to_id("</s>")
    
    for _ in range(max_len):
        # 2. Forward pass to get logits for the *last* token
        # NOTE: For efficiency, in a production setting, you would only pass the last token
        # and use KV-caching. Here, we pass the full sequence as a simple implementation.
        logits = model(input_ids)
        next_token_logits = logits[0, -1, :] / temperature

        # 3. Apply Top-K filtering
        if top_k > 0:
            # Keep only top_k logits, set others to -infinity
            indices_to_remove = next_token_logits < torch.topk(next_token_logits, top_k)[0][..., -1, None]
            next_token_logits[indices_to_remove] = float('-inf')

        # 4. Apply Top-P (Nucleus) filtering
        if top_p < 1.0:
            sorted_logits, sorted_indices = torch.sort(next_token_logits, descending=True)
            cumulative_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)
            
            # Remove tokens where cumulative probability exceeds top_p
            sorted_indices_to_remove = cumulative_probs > top_p
            # Shift the indices to the right to keep the first token above the threshold
            sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
            sorted_indices_to_remove[..., 0] = 0

            indices_to_remove = sorted_indices[sorted_indices_to_remove]
            next_token_logits[indices_to_remove] = float('-inf')

        # 5. Sample the next token
        probs = F.softmax(next_token_logits, dim=-1)
        next_token = torch.multinomial(probs, num_samples=1)
        
        # Check for EOS
        if next_token.item() == EOS_TOKEN:
            break

        # 6. Append to input
        input_ids = torch.cat([input_ids, next_token.unsqueeze(0)], dim=1)

    # 7. Decode and return
    output = tokenizer.decode(input_ids[0].tolist(), skip_special_tokens=True)
    return output

# -----------------------
# SAMPLE TEXT
# -----------------------
prompt = "The most challenging part of training a small language model is"
generated_text = generate(
    model, 
    tokenizer, 
    prompt, 
    max_len=150, 
    temperature=0.7, 
    top_k=50, 
    top_p=0.9,
)
print("📝 GENERATED TEXT:\n")
print(generated_text)