import time
import numpy as np
import matplotlib.pyplot as plt
import feedforwardmodel
import figplot

t1 = time.time() 

g_average_list=np.load('data/glist.npy')

np.random.seed(3)

def calculate_simple_case(state, w_initial, delta_t, pre_neuron):
    """
    Calculating the simple case in which there are no spontaneous spikes. 
    """
    variables = feedforwardmodel.stdp_variables(feedforwardmodel.up_down_parameters(state), g_average_list, w_initial, pre_neuron)[0:6]
    simple_feedforward = feedforwardmodel.FeedforwardSTDP(*variables, delta_t=delta_t) 
    simple_feedforward.change(plot_option=True)
    t_start = 10050
    t_end = 10200
    return simple_feedforward.x_plot[t_start:t_end], simple_feedforward.y_plot[t_start:t_end], simple_feedforward.g_plot[t_start:t_end]/simple_feedforward.g_average_plot[t_start:t_end], simple_feedforward.large_c_plot[t_start:t_end], simple_feedforward.bpost_plot[t_start:t_end], simple_feedforward.w_plot[t_start:t_end], simple_feedforward.didt_plot[t_start:t_end], simple_feedforward.dphidt_plot[t_start:t_end]

# plot time series
fig = plt.figure(figsize=(8,10))
state_list = ['down' , 'up']
for i in range(2):
    state=state_list[i]
    x, y, g_ratio, large_c, bpost, w, didt, dphidt = calculate_simple_case(state=state, w_initial=0.5, delta_t=10, pre_neuron=20)
    #Plot figures
    figplot.plot_timeseries(g=x[:,0], g_min=-0.1, g_max=1.1, t_min=0, t_max=150, aspect=0.2, fig=fig, plot_x=8, plot_y=2, plot_area=i+1)
    figplot.plot_timeseries(g=y, g_min=-0.1, g_max=1.1, t_min=0, t_max=150, aspect=0.2, fig=fig, plot_x=8, plot_y=2, plot_area=i+3)
    figplot.plot_timeseries(g=g_ratio, g_min=-5.0, g_max=65.0, t_min=0, t_max=150, aspect=0.2, fig=fig, plot_x=8, plot_y=2, plot_area=i+5)
    figplot.plot_timeseries(g=large_c[:,0], g_min=-0.04, g_max=0.12, t_min=0, t_max=150, aspect=0.2, fig=fig, plot_x=8, plot_y=2, plot_area=i+7)
    figplot.plot_timeseries(g=bpost, g_min=-0.5, g_max=4.5, t_min=0, t_max=150, aspect=0.2, fig=fig, plot_x=8, plot_y=2, plot_area=i+9)
    figplot.plot_timeseries(g=w[:,0], g_min=0.996, g_max=1.008, t_min=0, t_max=150, aspect=0.2, fig=fig, plot_x=8, plot_y=2, plot_area=i+11)
    figplot.plot_timeseries(g=didt[:,0], g_min=0.996, g_max=1.008, t_min=0, t_max=150, aspect=0.2, fig=fig, plot_x=8, plot_y=2, plot_area=i+13)
    figplot.plot_timeseries(g=dphidt[:,0], g_min=0.996, g_max=1.008, t_min=0, t_max=150, aspect=0.2, fig=fig, plot_x=8, plot_y=2, plot_area=i+15)
plt.savefig('figure/feedforward/representative.pdf')


t2 = time.time()
elapsed_time = t2-t1
print(f"Total time：{elapsed_time}")




