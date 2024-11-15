import torch
import argparse
from comm import TorchDistComm
from transformers import AutoModelForCausalLM, AutoTokenizer

parser = argparse.ArgumentParser(description="Run Distribute")
parser.add_argument("--rank", type=int, required=True, help="Rank ID")
parser.add_argument("--world-size", type=int, required=True)
parser.add_argument("--full-model-name", type=str, required=True)
args = parser.parse_args()

# Load model and tokenizer
model = AutoModelForCausalLM.from_pretrained(args.full_model_name, torch_dtype=torch.bfloat16, device_map=f"cuda:{args.rank}")
tokenizer = AutoTokenizer.from_pretrained(args.full_model_name)

# Distrubte Init
comm = TorchDistComm(args.rank, args.world_size)

# Processing prompt
prompt = "Question: Please write a function in Python that transforms bytes to Giga bytes.\n\nAnswer:"
next_token_id = tokenizer(prompt, return_tensors="pt")["input_ids"].to(comm.device)

# Run Phase 1 (Generate KVCache and first token)
past_key_values = None
next_logits, past_key_values = model(next_token_id, past_key_values=past_key_values, use_cache=True).to_tuple()
next_logits = next_logits[:, -1:]
next_token_id = torch.argmax(next_logits, dim=-1)

for i in range(1, comm.world_size):
    comm.naive_send(next_token_id, i)
    comm.naive_send(torch.stack(past_key_values), i)
