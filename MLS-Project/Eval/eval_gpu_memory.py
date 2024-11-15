import matplotlib.pyplot as plt
import torch
import gc
import time
from pynvml import nvmlInit, nvmlDeviceGetHandleByIndex, nvmlDeviceGetMemoryInfo
from statistics import mean

models = ['octocoder', 'bloom-1b1', 'bloom-3b', 'bloom-7b1']
# models = ['octocoder']
token_size = [str(i) for i in range(128, 2048+1, 128)]
mem_usage = {}

# GPU Init
nvmlInit()
gpu_handle = nvmlDeviceGetHandleByIndex(0)

markers = ['o', 's', '^', 'd', 'x', '+']
plt.figure(figsize=(10, 6))

for i in range(len(models)):
    model = models[i]
    mem_usage[model] = []
    for size in token_size:
        # Clean up cache
        gc.collect()
        torch.cuda.empty_cache()
        torch.cuda.synchronize()
        
        # Load Tensor
        path = f'output/{model}/kvcache_{model}_{size.zfill(4)}.pt'
        print(f'loading tensor {path}')
        past_key_values = torch.load(path, map_location=torch.device('cuda'))
        torch.cuda.synchronize()
        
        # Measure GPU memory usage after torch load KVCache
        mem_info = nvmlDeviceGetMemoryInfo(gpu_handle)
        mem_usage[model].append(mem_info.used / (1024 ** 2))
    
    # Add mem_usage to plot
    plt.plot(token_size, mem_usage[model], label=model, marker=markers[i])

# Adding a legend
plt.legend()

# Adding a title and axis labels
plt.title('GPU memory usage after loading KV-cache')
plt.xlabel('Token Size')
plt.ylabel('GPU Memory Usage (MB)')

# Displaying the plot
plt.savefig('gpu_memory_usage2.png')
print(mem_usage)