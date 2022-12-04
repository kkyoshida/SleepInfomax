import time
import numpy as np
import matplotlib.pyplot as plt
import feedforwardmodel
import figplot

t1 = time.time() 

np.random.seed(5)
g_average_list=np.load('data/glist.npy')

def calculate_single_trace(g_index, w_initial, pre_neuron, mode):
    """
    Calculating the time series of synaptic weights and postsynaptic spikes 
    in two modes: post-pre stimulations or pre-only stimulations
    """
    if mode == 'postpre':
        cal_stdp = feedforwardmodel.FeedforwardSTDP(*feedforwardmodel.stdp_variables(g_index, g_average_list, w_initial, pre_neuron), delta_t=-10)
    elif mode == 'preonly':
        cal_stdp = feedforwardmodel.FeedforwardSTDPPreonly(*feedforwardmodel.stdp_variables(g_index, g_average_list, w_initial, pre_neuron))
    cal_stdp.change(plot_option=True)
    return np.mean(cal_stdp.w_plot[10050:10200,:], axis=1), cal_stdp.y_plot[10050:10200]

def calculate_average_trace(state, w_initial, pre_neuron, mode):
    """
    Calculating the average time series of synaptic weights and postsynaptic spikes 
    in two modes: post-pre stimulations or pre-only stimulations
    """
    g_index = feedforwardmodel.up_down_parameters(state=state)
    trial_number = 20000
    result_w = np.zeros((150, trial_number))
    result_y = np.zeros((150, trial_number))
    for trial in range(trial_number):
        result_w[:,trial], result_y[:,trial] = calculate_single_trace(g_index=g_index, w_initial=w_initial, pre_neuron=pre_neuron, mode=mode)
    return result_w, result_y

fig = plt.figure(figsize=(16,10))
for i in range(2):
    for j in range(2):
        w_trace, y_trace = calculate_average_trace(state=['up', 'down'][i], w_initial=0.5, pre_neuron=20, mode=['postpre', 'preonly'][j])
        w_average = np.mean(w_trace, axis=1)
        w_sd = np.std(w_trace, axis=1)
        y_average = np.mean(y_trace, axis=1)
        figplot.plot_timeseries_sd(g=w_average, g_sd=w_sd, g_min=0.995, g_max=1.004, t_min=0, t_max=150, aspect=0.5, fig=fig, plot_x=2, plot_y=4, plot_area=1+4*i+2*j)
        figplot.plot_timeseries_points(g=y_average, g_min = -0.0025, g_max = 0.05, t_min=0, t_max=150, aspect=0.5, fig=fig, plot_x=2, plot_y=4, plot_area=2+4*i+2*j)
        
plt.savefig('figure/feedforward/averagetrace.pdf')


t2 = time.time()
elapsed_time = t2-t1
print(f"Total time：{elapsed_time}")

