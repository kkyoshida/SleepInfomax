import time
import numpy as np
import matplotlib.pyplot as plt
import figplot

t1 = time.time() 

conv_time = 200000

def plot_timeseries_sd_two(g_1, g_2, g_sd_1, g_sd_2, g_min, g_max, t_min, t_max, aspect, fig, plot_x, plot_y, plot_area): 
    time_trace = np.arange(t_min - conv_time + 1, t_max - conv_time + 1)
    ax = fig.add_subplot(plot_x, plot_y, plot_area)
    ax.plot(time_trace, g_1, color='red', zorder=-2)
    ax.plot(time_trace, g_2, color='purple', zorder=-3)
    ax.set_aspect(aspect * (t_max - t_min) / (g_max - g_min))
    ax.set_xlim(t_min - conv_time + 1, t_max - conv_time + 1)
    ax.set_ylim(g_min, g_max)
    ax.fill_between(time_trace, g_1[t_min:t_max] + g_sd_1[t_min:t_max], g_1[t_min:t_max] - g_sd_1[t_min:t_max], alpha=0.15, color='red', zorder=-8)
    ax.fill_between(time_trace, g_2[t_min:t_max] + g_sd_2[t_min:t_max], g_2[t_min:t_max] - g_sd_2[t_min:t_max], alpha=0.15, color='purple', zorder=-9)
    ax.set_rasterization_zorder(-5)

def plot_timeseries_sd_two_one(g_1, g_2, g_sd_1, g_sd_2, g_min, g_max, t_min, t_max, aspect, fig, plot_x, plot_y, plot_area, g_3): 
    time_trace = np.arange(t_min, t_max)
    ax = fig.add_subplot(plot_x, plot_y, plot_area)
    ax.plot(time_trace, g_1, color='red', zorder=-2)
    ax.plot(time_trace, g_2, color='purple', zorder=-3)
    ax.set_aspect(aspect * (t_max - t_min) / (g_max - g_min))
    ax.set_xlim(t_min, t_max)
    ax.set_ylim(g_min, g_max)
    ax.fill_between(time_trace, g_1[t_min:t_max] + g_sd_1[t_min:t_max], g_1[t_min:t_max] - g_sd_1[t_min:t_max], alpha=0.15, color='red', zorder=-8)
    ax.fill_between(time_trace, g_2[t_min:t_max] + g_sd_2[t_min:t_max], g_2[t_min:t_max] - g_sd_2[t_min:t_max], alpha=0.15, color='purple', zorder=-9)
    ax.plot(time_trace, g_3, color='tab:blue', linestyle='dashed', zorder=-10)
    ax.set_rasterization_zorder(-5)

result_weight_globalpre = np.load('data/result_weight_globalpre_down.npy')
result_weight_localpre = np.load('data/result_weight_localpre_down.npy')
#result_reactivation_global = np.load('data/result_reactivation_global.npy')
#result_reactivation_local = np.load('data/result_reactivation_local.npy')
result_perf = np.load('data/result_perf_down.npy')
result_perf_beforesleep = np.load('data/result_perf_beforesleep_down.npy')
base_performance = np.load('data/base_performance_down.npy')

epolen = 50000  # time length per one epoch
eponum = 18  # the number of total epoch 
trial_number = 100
#v = np.ones(conv_time) / conv_time    

#basefr_global = 5.93
#basefr_local = 7.97
#result_reactivation_global = (result_reactivation_global - basefr_global) / (np.mean(result_reactivation_global[:, :, 0]) - basefr_global) * 100
#result_reactivation_local = (result_reactivation_local - basefr_local) / (np.mean(result_reactivation_local[:, :, 0]) - basefr_local) * 100

#reactivation_pre = np.convolve(np.array([7.5 - 2.5/900000 * i for i in range(900000)]), v, mode='valid')
#reactivation_pre = reactivation_pre / reactivation_pre[0] * 100

result_weight_globalpre_mean = np.mean(result_weight_globalpre[:,:,:], axis=0)  # mean synaptic weights
result_weight_globalpre_se = np.std(result_weight_globalpre[:,:,:], axis=0) / np.sqrt(trial_number-1)  # standard error of synaptic weights
result_weight_localpre_mean = np.mean(result_weight_localpre[:,:,:], axis=0)  # mean synaptic weights
result_weight_localpre_se = np.std(result_weight_localpre[:,:,:], axis=0) / np.sqrt(trial_number-1)  # standard error of synaptic weights

#result_reactivation_global_mean = np.mean(result_reactivation_global[:,:,:], axis=0)  
#result_reactivation_global_se = np.std(result_reactivation_global[:,:,:], axis=0) / np.sqrt(trial_number-1)  
#result_reactivation_local_mean = np.mean(result_reactivation_local[:,:,:], axis=0)  
#result_reactivation_local_se = np.std(result_reactivation_local[:,:,:], axis=0) / np.sqrt(trial_number-1)  

result_perf_mean = np.mean(result_perf, axis=0)  # mean performance
result_perf_se = np.std(result_perf, axis=0) / np.sqrt(trial_number-1)  # standard error of performance
result_perf_beforesleep_mean = np.mean(result_perf_beforesleep)
result_perf_beforesleep_se = np.std(result_perf_beforesleep) / np.sqrt(trial_number-1)

fig = plt.figure(figsize=(25, 25))

perf_increase = np.array([result_perf_beforesleep_mean - base_performance, result_perf_mean[0] - base_performance, result_perf_mean[1] - base_performance, result_perf_mean[2] - base_performance])
perf_increase_se = np.array([result_perf_beforesleep_se, result_perf_se[0], result_perf_se[1], result_perf_se[2]])

figplot.plot_bar_se(perf_increase, perf_increase_se, g_min=0, g_max=4.0, aspect=1, labels=['Before sleep', 'Up', 'Global up', 'Local up'], fig=fig, plot_x=7, plot_y=1, plot_area=7)

for i in range(3):
    plot_timeseries_sd_two(g_1=result_weight_globalpre_mean[i], g_2=result_weight_localpre_mean[i], g_sd_1=result_weight_globalpre_se[i], g_sd_2=result_weight_localpre_se[i], g_min=0.07, g_max=0.37, t_min=0, t_max=epolen*eponum, aspect=0.8, fig=fig, plot_x=7, plot_y=1, plot_area=2*i+1)
    #plot_timeseries_sd_two_one(g_1=result_reactivation_global_mean[i], g_2=result_reactivation_local_mean[i], g_sd_1=result_reactivation_global_se[i], g_sd_2=result_reactivation_local_se[i], g_min=45, g_max=110, t_min=0, t_max=epolen*eponum-conv_time+1, aspect=0.8, fig=fig, plot_x=7, plot_y=1, plot_area=2*i+2, g_3=reactivation_pre)

plt.savefig('figure/slowwave/task_weight_nested_down.pdf', dpi=300)

t2 = time.time()
elapsed_time = t2-t1
print(f"Total time：{elapsed_time}")
