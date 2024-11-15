import torch
import argparse
from comm import TorchDistComm
import torch.distributed as dist

from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline

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

# Recv first token and KVCache
master_id = 0
next_token_id = comm.naive_recv(torch.int64, master_id)
past_key_values = comm.naive_recv(torch.bfloat16, master_id)

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