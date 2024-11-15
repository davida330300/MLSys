import torch
import torch.distributed as dist
import argparse
import time
import random

parser = argparse.ArgumentParser(description="Run Distribute")
parser.add_argument("--rank", type=str, required=True, help="Rank ID")
parser.add_argument("--world-size", type=int, required=True)
args = parser.parse_args()

rank = int(args.rank)
world_size = int(args.world_size)
dist.init_process_group(backend='nccl', init_method='tcp://localhost:12345', rank=rank, world_size=world_size)
gloo_group = dist.new_group(backend='gloo', ranks=[i for i in range(world_size)])

print(f"Worker {rank} is waiting")

dist.barrier()

print(f"Worker {rank} found everyone is online")

if rank == 0:
    # The sender process sends a tensor to the receiver process
    for i in range(10):
        r1 = random.randint(10, 100)
        r2 = random.randint(10, 100)
        tensor_size = torch.tensor([r1, r2]).to('cpu')
        tensor_to_send = torch.randn(r1, r2).to(f'cuda:{rank}')
        
        print(f" ======= Round {i} ======= ")
        for recv_id in range(1, world_size):
            print(f"Rank 0 sent tensor size {tensor_size} to Rank {recv_id}")
            dist.send(tensor_size, dst=recv_id, group=gloo_group)
            
            print(f"Rank 0 sent tensor to Rank {recv_id}")
            dist.send(tensor_to_send, dst=recv_id)
        time.sleep(i)

else:
    # The receiver process receives the tensor sent by process 0
    for _ in range(10):
        tensor_size = torch.zeros(2, dtype=torch.long).to('cpu')
        dist.recv(tensor_size, src=0, group=gloo_group)
        print(f"Rank {rank} received tensor size:", tensor_size)
        
        received_tensor = torch.zeros(tensor_size[0].item(), tensor_size[1].item()).to(f'cuda:{rank}')
        dist.recv(received_tensor, src=0)
        print(f"Rank {rank} received tensor")
