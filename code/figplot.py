import numpy as np
import matplotlib.pyplot as plt
plt.rcParams['pdf.fonttype'] = 42
plt.rcParams['ps.fonttype'] = 42
plt.rcParams['font.size'] = 12
plt.rcParams['font.family'] = 'sans-serif'
plt.rcParams['font.sans-serif'] = ['Arial']

# All functions are used for plotting figures for the paper. 

def plot_timeseries(g, g_min, g_max, t_min, t_max, aspect, fig, plot_x, plot_y, plot_area, plot_color='tab:blue'):
    ax = fig.add_subplot(plot_x, plot_y, plot_area)
    ax.plot(g, color=plot_color)
    ax.set_aspect(aspect * (t_max-t_min) / (g_max - g_min))
    ax.set_xlim(t_min , t_max)
    ax.set_ylim(g_min , g_max)

def plot_stagetimeseries(g, g_min, g_max, t_min, t_max, aspect, fig, plot_x, plot_y, plot_area, stages):
    ax = fig.add_subplot(plot_x, plot_y, plot_area)
    ax.plot(g, color='tab:blue')
    ax.set_aspect(aspect * (t_max - t_min) / (g_max - g_min))
    ax.set_xlim(t_min, t_max)
    ax.set_ylim(g_min, g_max)
    if stages == 'updown':
        ax.set_yticks([0, 8])
        ax.set_yticklabels(['Down', 'Up'])
    if stages == 'gsoldw':
        ax.set_yticks([0, 2, 6, 8])
        ax.set_yticklabels(['Global down', 'Local down', 'Global up', 'Local up'])

def plot_timeseries_sd(g, g_sd, g_min, g_max, t_min, t_max, aspect, fig, plot_x, plot_y, plot_area): 
    time_trace = np.arange(t_min, t_max)
    ax = fig.add_subplot(plot_x, plot_y, plot_area)
    ax.plot(time_trace, g, marker='o', markersize=3, color='tab:blue')
    ax.set_aspect(aspect * (t_max - t_min) / (g_max - g_min))
    ax.set_xlim(t_min, t_max)
    ax.set_ylim(g_min, g_max)
    ax.fill_between(time_trace, g[t_min:t_max] + g_sd[t_min:t_max], g[t_min:t_max] - g_sd[t_min:t_max], alpha=0.15)

def plot_timeseries_sd_multi(g_1, g_2, g_3, g_sd_1, g_sd_2, g_sd_3, g_min, g_max, t_min, t_max, aspect, fig, plot_x, plot_y, plot_area): 
    time_trace=np.arange(t_min, t_max)
    ax = fig.add_subplot(plot_x, plot_y, plot_area)
    ax.plot(time_trace, g_1, color='darkgreen', zorder=-2)
    ax.plot(time_trace, g_2, color='red', zorder=-3)
    ax.plot(time_trace, g_3, color='purple', zorder=-4)
    ax.set_aspect(aspect * (t_max - t_min) / (g_max - g_min))
    ax.set_xlim(t_min, t_max)
    ax.set_ylim(g_min, g_max)
    ax.fill_between(time_trace, g_1[t_min:t_max] + g_sd_1[t_min:t_max], g_1[t_min:t_max] - g_sd_1[t_min:t_max], alpha=0.15, color='darkgreen', zorder=-8)
    ax.fill_between(time_trace, g_2[t_min:t_max] + g_sd_2[t_min:t_max], g_2[t_min:t_max] - g_sd_2[t_min:t_max], alpha=0.15, color='red', zorder=-9)
    ax.fill_between(time_trace, g_3[t_min:t_max] + g_sd_3[t_min:t_max], g_3[t_min:t_max] - g_sd_3[t_min:t_max], alpha=0.15, color='purple', zorder=-10)
    ax.set_rasterization_zorder(-5)

def plot_timeseries_points(g, g_min, g_max, t_min, t_max, aspect, fig, plot_x, plot_y, plot_area): 
    time_trace=np.arange(t_min, t_max)
    ax = fig.add_subplot(plot_x, plot_y, plot_area)
    ax.plot(time_trace, g, marker='o', markersize=3, color='tab:blue')
    ax.set_aspect(aspect * (t_max - t_min) / (g_max - g_min))
    ax.set_xlim(t_min, t_max)
    ax.set_ylim(g_min, g_max)

def plot_function(x, y_1, y_2, aspect, fig):
    ax = fig.add_subplot(1, 1, 1)
    ax.plot(x, y_1, color = 'darkorange')
    ax.plot(x, y_2, color = 'navy')
    x_min = np.min(x) - 1.0
    x_max = np.max(x) + 1.0
    y_min = -5.0
    y_max = np.max(y_2) + 5.0
    ax.set_xlim(x_min, x_max)
    ax.set_ylim(y_min, y_max)
    ax.set_aspect(aspect * (x_max - x_min) / (y_max - y_min))

def plot_stdp_curve(delta_t, g, preonly, g_min, g_max, t_min, t_max, aspect, fig, plot_x, plot_y, plot_area):
    ax = fig.add_subplot(plot_x, plot_y, plot_area)
    ax.hlines([0], t_min - 2, t_max + 2, 'black', alpha=1.0, zorder=-1, linewidth=1.0)
    ax.hlines([preonly], t_min - 2, t_max + 2, 'orange', zorder=1, linestyle='dashed')
    ax.vlines([0], g_min, g_max, 'black', alpha=1.0, zorder=-2, linewidth=1.0)
    ax.tick_params(bottom=True, left=True, right=False, top=False)
    ax.set_xticks([-80, -40, 0, 40, 80])
    ax.scatter(delta_t, g, s=5, color='tab:blue', zorder=1)
    ax.set_aspect(aspect * (t_max - t_min) / (g_max - g_min))
    ax.set_xlim(t_min - 2, t_max + 2)
    ax.set_ylim(g_min, g_max)
    ax.grid(which='major', axis='x', color='gray', alpha=0.5,
        linestyle='--', linewidth=0.5, zorder=-3)
    ax.grid(which='major', axis='y', color='gray', alpha=0.5,
        linestyle='--', linewidth=0.5, zorder=-4)
    
def plot_scatter(x, y, x_min, x_max, y_min, y_max, aspect, fig, plot_x, plot_y, plot_area):
    ax = fig.add_subplot(plot_x, plot_y, plot_area)
    ax.hlines([0], x_min - 2, x_max + 2, 'black', alpha=1.0, zorder=-1, linewidth=1.0)
    ax.scatter(x, y, s=5, color='tab:blue', zorder=1)
    ax.set_aspect(aspect * (x_max - x_min) / (y_max - y_min))
    ax.set_xlim(x_min - 2, x_max + 2)
    ax.set_ylim(y_min, y_max)

def plot_heatmap_fr(g, g_min, g_max, t_min, t_max, aspect, fig, plot_x, plot_y, plot_area, cmap):
    ax = fig.add_subplot(plot_x, plot_y, plot_area)
    plt.imshow(g, cmap=cmap, vmin=g_min, vmax=g_max)
    plt.colorbar()
    ax.set_aspect(aspect * (t_max - t_min) / np.shape(g)[0])
    ax.set_xlim(t_min, t_max)

def plot_bar(g, g_min, g_max, aspect, labels, fig, plot_x, plot_y, plot_area):
    ax = fig.add_subplot(plot_x, plot_y, plot_area)
    x_position = np.array([i + 1 for i in range(np.size(g))])
    ax.bar(x_position, g, tick_label=labels)
    ax.set_aspect(np.size(g) * aspect / (g_max - g_min))
    ax.set_ylim(g_min, g_max)

def plot_bar_se(g, g_se, g_min, g_max, aspect, labels, fig, plot_x, plot_y, plot_area):
    ax = fig.add_subplot(plot_x, plot_y, plot_area)
    x_position = np.array([i + 1 for i in range(np.size(g))])
    ax.bar(x_position, g, yerr=g_se, tick_label=labels, capsize=10)
    ax.set_aspect(np.size(g) * aspect / (g_max - g_min))
    ax.set_ylim(g_min, g_max)

def plot_histogram(g, threshold_to_down, threshold_to_up, fig, plot_x, plot_y, plot_area):
    ax = fig.add_subplot(plot_x, plot_y, plot_area)
    ax.hist(g, bins=160, density=True)
    ax.set_ylim(0.0, 0.6)
    ax.vlines(threshold_to_down, ymin=0.0, ymax=0.6)
    ax.vlines(threshold_to_up, ymin=0.0, ymax=0.6)

def plot_scatter_weight(x, y, x_min, x_max, y_min, y_max, aspect, fig, plot_x, plot_y, plot_area):
    ax = fig.add_subplot(plot_x, plot_y, plot_area)
    ax.hlines([0], x_min - 0.05, x_max + 0.05, 'black', alpha=1.0, zorder=-1, linewidth=1.0)
    ax.scatter(x, y, s=5, color='tab:blue', zorder=1)
    ax.set_aspect(aspect * (x_max - x_min) / (y_max - y_min))
    ax.set_xlim(x_min - 0.05, x_max + 0.05)
    ax.set_ylim(y_min, y_max)

   

