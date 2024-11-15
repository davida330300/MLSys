from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline
import torch
import gc
import numpy as np

# Clear Cache
torch.cuda.synchronize()
gc.collect()
torch.cuda.empty_cache()

# Load model and tokenizer
model = AutoModelForCausalLM.from_pretrained("bigcode/octocoder", torch_dtype=torch.bfloat16, device_map="cuda:0")
tokenizer = AutoTokenizer.from_pretrained("bigcode/octocoder")

# Load first token
next_token_id = torch.load("next_token_id.pt")

# Load KVCache
past_key_values = torch.load("past_key_values.pt")

# Run Phase 2 (Generate text iteratively)
generated_tokens = [next_token_id.item()]
for i in range(9):
  next_logits, past_key_values = model(next_token_id, past_key_values=past_key_values, use_cache=True).to_tuple()
  next_logits = next_logits[:, -1:]
  next_token_id = torch.argmax(next_logits, dim=-1)

  generated_tokens.append(next_token_id.item())

# Decode and print result
generated_text = tokenizer.batch_decode(generated_tokens)
print(generated_text)