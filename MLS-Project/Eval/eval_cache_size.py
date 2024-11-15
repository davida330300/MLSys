import matplotlib.pyplot as plt
import os

models = ['bloom-1b1', 'bloom-3b', 'bloom-7b1', 'octocoder']
token_size = [str(i) for i in range(128, 2048+1, 128)]
file_size = {}

markers = ['o', 's', '^', 'd', 'x', '+']
plt.figure(figsize=(10, 6))

for i in range(len(models)):
    model = models[i]
    file_size[model] = []
    for size in token_size:
        path = f'output/{model}/kvcache_{model}_{size.zfill(4)}.pt'
        file_stats = os.stat(path)
        file_size[model].append(file_stats.st_size / (1024 * 1024))
        
    plt.plot(token_size, file_size[model], label=model, marker=markers[i])

# Adding a legend
plt.legend()

# Adding a title and axis labels
plt.title('KVCache Size for different models with different batch sizes')
plt.xlabel('Token Size')
plt.ylabel('KVCache Size (MB)')

# Displaying the plot
plt.savefig('kvcache_size.png')