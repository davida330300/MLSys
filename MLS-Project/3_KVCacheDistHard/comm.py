
import torch.distributed as dist
import torch
from collections.abc import Iterable

class TorchDistComm:
    def __init__(self, rank, world_size):
        self.rank = rank
        self.world_size = world_size
        self.device = f"cuda:{self.rank}"
        
        dist.init_process_group(backend='nccl', init_method='tcp://localhost:12345', rank=rank, world_size=world_size)
        self.cpu_group = dist.new_group(backend='gloo', ranks=[i for i in range(world_size)])
        
    # Should hava better ways to reduce communication messages
    # This is just a easiest way to receive tensors with arbitrary shape
    def naive_recv(self, data_type, id):
        # Recv Shape Length
        tensor_shape_len = torch.zeros(1, dtype=torch.int64)
        dist.recv(tensor_shape_len, src=id, group=self.cpu_group)
        print(tensor_shape_len)
        
        # Recv Shape
        tensor_shape = torch.zeros(tensor_shape_len.item(), dtype=torch.int64)
        dist.recv(tensor_shape, src=id, group=self.cpu_group)
        print(tensor_shape)
        
        # Recv Tensor
        tensor = torch.zeros(tensor_shape.tolist(), dtype=data_type).to(self.device)
        dist.recv(tensor, src=id)
        
        print("complete recv")
        return tensor
    
    # Should hava better ways to reduce communication messages
    # This is just a easiest way to send tensors with arbitrary shape
    def naive_send(self, data, id):
        # Send Shape Length
        dist.send(torch.tensor(len(data.shape)), dst=id, group=self.cpu_group)
        print(torch.tensor(len(data.shape)).dtype)
        
        # Send Shape
        dist.send(torch.tensor(data.shape), dst=id, group=self.cpu_group)
        print(torch.tensor(data.shape).dtype)
        
        # Send Tensor
        dist.send(data.to(self.device), dst=id)
        print(data.dtype)