import matplotlib.pyplot as plt
import torch
import gc
import time
import numpy
from statistics import mean 

def n_load_avg(n, path):
    records = []
    for i in range(n):
        torch.cuda.synchronize()
        gc.collect()
        torch.cuda.empty_cache()
        
        load_start_time = time.perf_counter()
        past_key_values = torch.load(path, map_location=torch.device('cuda'))
        load_end_time = time.perf_counter()
        records.append((load_end_time - load_start_time) * 1000)
    
    return mean(records)

models = ['bloom-1b1', 'bloom-3b', 'bloom-7b1', 'octocoder']
token_size = [str(i) for i in range(128, 2048+1, 128)]
timing = {}

markers = ['o', 's', '^', 'd', 'x', '+']
plt.figure(figsize=(10, 6))

for i in range(len(models)):
    model = models[i]
    timing[model] = []
    for size in token_size:
        path = f'output/{model}/kvcache_{model}_{size.zfill(4)}.pt'
        
        print(f'loading tensor {path}')
        timing[model].append(n_load_avg(10, path))
        
    plt.plot(token_size, timing[model], label=model, marker=markers[i])


# Adding a legend
plt.legend()

# Adding a title and axis labels
plt.title('Overhead of the loading KV-cache from disk to GPU on L40 machine')
plt.xlabel('Token Size')
plt.ylabel('Loading Time (ms)')

# Displaying the plot
plt.savefig('load_time2.png')