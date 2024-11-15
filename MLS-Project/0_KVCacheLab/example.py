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

# Processing prompt
prompt = "Question: Please write a function in Python that transforms bytes to Giga bytes.\n\nAnswer:"
next_token_id = tokenizer(prompt, return_tensors="pt")["input_ids"].to("cuda:0")

# Run Phase1 and Phase2
generated_tokens = []
past_key_values = None
for i in range(10):
  next_logits, past_key_values = model(next_token_id, past_key_values=past_key_values, use_cache=True).to_tuple()
  next_logits = next_logits[:, -1:]
  next_token_id = torch.argmax(next_logits, dim=-1)
  generated_tokens.append(next_token_id.item())

# Decode and print result
generated_text = tokenizer.batch_decode(generated_tokens)
print(generated_text)