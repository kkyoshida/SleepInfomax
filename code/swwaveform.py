import time
import numpy as np
import matplotlib.pyplot as plt
import slowwavemodel
import figplot

t1 = time.time() 

np.random.seed(3)

def calculate_fr_heatmap(y):
    """
    Plotting the time series of firing rates calculated by time series of postsynaptic spikes y
    """
    v = np.ones(1000)
    fr = np.zeros(y.T.shape)
    for i in range(fr.shape[0]):
        fr[i, :] = np.convolve(y[:, i], v, mode='same') #calculating the number of spikes in 1000 msec
    return fr
    

def plot_slowwave(file_name, connect_prob_ee, connect_prob_ie):
    """
    Plotting the simulation result of the slow wave model for given connecting probability 
    of the E to E (connect_prob_ee) and E to I (connect_prob_ie) in different populations
    """
    
    sw = slowwavemodel.Slowwave(epoch_length=30000, epoch_number=1, pre_w_initial=0.0, plasticity_option=False, plot_option=True, connect_prob_ee=connect_prob_ee, connect_prob_ie=connect_prob_ie) 
    sw.simulate()
    fig = plt.figure(figsize=(20, 30))
    t_start = 10000
    t_end = 30000
    
    figplot.plot_stagetimeseries(sw.state_plot, -0.4, 8.4, t_start, t_end, 0.2, fig, 12, 1, 1, stages='gsoldw')
    figplot.plot_stagetimeseries(sw.up_or_down_plot, -0.4, 8.4, t_start, t_end, 0.2, fig, 12, 1, 2, stages='updown')
    figplot.plot_timeseries(np.mean(sw.u_e_plot[:, 0:sw.exc_popu], axis=1), -73, -60, t_start, t_end, 0.2, fig, 12, 1, 3, plot_color='darkorange')
    figplot.plot_timeseries(np.mean(sw.u_e_plot[:, sw.exc_popu:2*sw.exc_popu], axis=1), -73, -60, t_start, t_end, 0.2, fig, 12, 1, 4, plot_color='darkorange')
    figplot.plot_timeseries(np.mean(sw.u_e_plot[:, 2*sw.exc_popu:3*sw.exc_popu], axis=1), -73, -60, t_start, t_end, 0.2, fig, 12, 1, 5, plot_color='darkorange')
    figplot.plot_timeseries(np.mean(sw.u_e_plot[:, 3*sw.exc_popu:4*sw.exc_popu], axis=1), -73, -60, t_start, t_end, 0.2, fig, 12, 1, 6, plot_color='darkorange')
    figplot.plot_timeseries(np.mean(sw.adapt_plot[:, 0:sw.exc_popu], axis=1), -0.01, 0.1, t_start, t_end, 0.2, fig, 12, 1, 7)
    figplot.plot_timeseries(np.mean(sw.u_i_plot[:, 0:sw.inh_popu], axis=1), -73, -52, t_start, t_end, 0.2, fig, 12, 1, 8, plot_color='navy')

    fr_e = calculate_fr_heatmap(sw.y_e_plot)
    figplot.plot_heatmap_fr(fr_e, 0.0, np.max(fr_e), t_start, t_end, 0.2, fig, 12, 1, 9, 'Oranges')
    
    fr_i = calculate_fr_heatmap(sw.y_i_plot)
    figplot.plot_heatmap_fr(fr_i, 0.0, np.max(fr_i), t_start, t_end, 0.2, fig, 12, 1, 10, 'Blues')
    
    fr_e = calculate_fr_heatmap(sw.y_e_plot[:, 0:sw.exc_popu])
    figplot.plot_heatmap_fr(fr_e, 0.0, np.max(fr_e), t_start, t_end, 0.2, fig, 12, 1, 11, 'Oranges')
    
    fr_i = calculate_fr_heatmap(sw.y_i_plot[:, 0:sw.inh_popu])
    figplot.plot_heatmap_fr(fr_i, 0.0, np.max(fr_i), t_start, t_end, 0.2, fig, 12, 1, 12, 'Blues')
    
    plt.savefig(file_name)
    
plot_slowwave('figure/slowwave/slowwave_globallocal.pdf', connect_prob_ee=0.05, connect_prob_ie=0.3)
np.random.seed(5)
plot_slowwave('figure/slowwave/slowwave_independent.pdf', connect_prob_ee=0.0, connect_prob_ie=0.0)

t2 = time.time()
elapsed_time = t2-t1
print(f"Total time：{elapsed_time}")
