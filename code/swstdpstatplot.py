import time
import numpy as np
import matplotlib.pyplot as plt
import slowwavemodel
import figplot

t1 = time.time()

result_stdp_pre = np.load('data/stdpstat.npy')
trial_number = np.shape(result_stdp_pre)[0]
result_stdp_pre_mean = np.mean(result_stdp_pre, axis=0)
result_stdp_pre_se = np.std(result_stdp_pre, axis=0) / np.sqrt(trial_number-1)

def plot_weight_se(x, ylist, selist, fig, plot_x, plot_y, plot_area):
    ax = fig.add_subplot(plot_x, plot_y, plot_area)
    x_min, x_max, y_min, y_max = 0.8, 2.2, -3.2, 12.2
    aspect = 2.0
    ax.hlines([0], x_min - 0.05, x_max + 0.05, 'black', alpha=1.0, zorder=-1, linewidth=1.0, linestyle='dashed')
    ax.set_aspect(aspect * (x_max - x_min) / (y_max - y_min))
    ax.set_xlim(x_min - 0.02, x_max + 0.02)
    ax.set_ylim(y_min, y_max)
    ax.set_xticks([1, 2])
    ax.errorbar(x, ylist[0], selist[0], capsize = 8, fmt='o', markersize=3, color='green', zorder=1)
    ax.plot(x, ylist[0], color='green', linestyle='dashed', zorder=1)
    ax.errorbar(x, ylist[1], selist[1], capsize = 8, fmt='o', markersize=3, color='saddlebrown', zorder=1)
    ax.plot(x, ylist[1], color='saddlebrown', linestyle='dashed', zorder=1)
    ax.errorbar(x, ylist[2], selist[2], capsize = 8, fmt='o', markersize=3, color='red', zorder=1)
    ax.plot(x, ylist[2], color='red', linestyle='dashed', zorder=1)
    ax.errorbar(x, ylist[3], selist[3], capsize = 8, fmt='o', markersize=3, color='purple', zorder=1)
    ax.plot(x, ylist[3], color='purple', linestyle='dashed', zorder=1)
    
fig = plt.figure(figsize=(10, 10))
plot_weight_se(np.array([1, 2]) , result_stdp_pre_mean, result_stdp_pre_se, fig, 2, 1, 1)
plt.savefig('figure/slowwave/stdpstat.pdf')

t2 = time.time()
elapsed_time = t2-t1
print(f"Total time：{elapsed_time}")