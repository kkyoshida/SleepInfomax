import time
import numpy as np
import matplotlib.pyplot as plt
import feedforwardmodel
import figplot

t1 = time.time() 

np.random.seed(4)
g_average_list = np.load('data/glist.npy')
  
def calculate_stdp_preonly(g_index, w_initial, pre_neuron):
    """
    Calculating synaptic weight changes after pre-only stimulations 
    in the case that the presynaptic firing rate is pre_fr*1000 [Hz] 
    """
    cal_stdp = feedforwardmodel.FeedforwardSTDPPreonly(*feedforwardmodel.stdp_variables(g_index, g_average_list, w_initial, pre_neuron))
    cal_stdp.change()
    return (np.mean(cal_stdp.w) - cal_stdp.w_initial) / cal_stdp.w_initial * 100

def plot_weight_dependency(file_name):
    """
    calculating the relationship between synaptic weight changes, initial synaptic weights, and the number of the stimulated neurons
    """
    trial_number = 100 #the number of trials 
    w_initial_list = np.array([0.04 * (i+1) for i in range(15)])
    pre_neuron_list = np.array([4 * (i+1) for i in range(15)])
    result_weight = np.zeros((2, np.size(w_initial_list), np.size(pre_neuron_list)))
    for state in range(2):
        g_index = feedforwardmodel.up_down_parameters(state=['down','up'][state])
        for i in range(np.size(w_initial_list)):
            for j in range(np.size(pre_neuron_list)):
                for trial in range(trial_number):
                    result_weight[state, i, j] += calculate_stdp_preonly(g_index=g_index, w_initial=w_initial_list[i], pre_neuron=pre_neuron_list[j])
    result_weight /= trial_number
    
    fig = plt.figure(figsize=(10, 10))
    for j in range(2):
        ax = fig.add_subplot(2, 1, j+1)
        ax.set_xticks([0, np.size(w_initial_list)-1])
        ax.set_xticklabels([w_initial_list[0], w_initial_list[-1]])
        ax.set_yticks([0, np.size(pre_neuron_list)-1])
        ax.set_yticklabels([pre_neuron_list[0], pre_neuron_list[-1]])
        ax.set_aspect('equal', adjustable='box')
        vminmax = max([np.abs(np.amax(result_weight[j, :, :])), np.abs(np.amin(result_weight[j, :, :]))])
        image = ax.imshow((result_weight[j, :, :]).T, vmin = -vminmax, vmax = vminmax, cmap='bwr', origin='lower')
        fig.colorbar(image, ax=ax)
    plt.savefig(file_name)

plot_weight_dependency(file_name='figure/feedforward/weight2d.pdf')

t2 = time.time()
elapsed_time = t2-t1
print(f"Total time：{elapsed_time}")
