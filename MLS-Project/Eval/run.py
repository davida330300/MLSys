from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline
import torch
import gc
import os
import sys

torch.cuda.synchronize()
gc.collect()
torch.cuda.empty_cache()

# model = AutoModelForCausalLM.from_pretrained("bigcode/octocoder", torch_dtype=torch.bfloat16, device_map="auto")
# tokenizer = AutoTokenizer.from_pretrained("bigcode/octocoder")

# model = AutoModelForCausalLM.from_pretrained("bigscience/bloom-1b1", torch_dtype=torch.bfloat16, device_map="auto")
# tokenizer = AutoTokenizer.from_pretrained("bigscience/bloom-1b1")

model = AutoModelForCausalLM.from_pretrained("bigscience/bloom-3b", torch_dtype=torch.bfloat16, device_map="auto")
tokenizer = AutoTokenizer.from_pretrained("bigscience/bloom-3b")

# model = AutoModelForCausalLM.from_pretrained("bigscience/bloom-7b1", torch_dtype=torch.bfloat16, device_map="auto")
# tokenizer = AutoTokenizer.from_pretrained("bigscience/bloom-7b1")

if not os.path.exists('output'): 
    os.mkdir('output')

for i in range (128, 2048+1, 128):
    print(i)
    prompt = " hello" * i

    past_key_values = None # past_key_values is the key-value cache
    next_token_id = tokenizer(prompt, return_tensors="pt")["input_ids"].to("cuda")
    next_logits, past_key_values = model(next_token_id, past_key_values=past_key_values, use_cache=True).to_tuple()
    torch.save(past_key_values, f"output/kvcache_Llama-2-7b_{str(len(next_token_id[0])).zfill(4)}.pt")
