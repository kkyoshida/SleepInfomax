import time
import numpy as np
import matplotlib.pyplot as plt
import feedforwardmodel
import figplot

t1 = time.time() 

np.random.seed(4)
g_average_list = np.load('data/glist.npy')
    
def calculate_stdp(g_index, w_initial, pre_neuron, delta_t):
    """
    Calculating synaptic weight changes after STDP stimulations 
    in the case that the presynaptic firing rate is pre_fr*1000 [Hz] 
    and the time difference of the STDP stimulation is delta_t [msec]
    """
    cal_stdp = feedforwardmodel.FeedforwardSTDP(*feedforwardmodel.stdp_variables(g_index, g_average_list, w_initial, pre_neuron), delta_t=delta_t)
    cal_stdp.change()
    return (np.mean(cal_stdp.w) - cal_stdp.w_initial) / cal_stdp.w_initial * 100
    
def calculate_stdp_preonly(g_index, w_initial, pre_neuron):
    """
    Calculating synaptic weight changes after pre-only stimulations 
    in the case that the presynaptic firing rate is pre_fr*1000 [Hz] 
    """
    cal_stdp = feedforwardmodel.FeedforwardSTDPPreonly(*feedforwardmodel.stdp_variables(g_index, g_average_list, w_initial, pre_neuron))
    cal_stdp.change()
    return (np.mean(cal_stdp.w) - cal_stdp.w_initial) / cal_stdp.w_initial * 100

def plot_stdp(file_name):
    """
    Calculating STDP curves and the relationship between synaptic weight changes and the average value of g(u(t))
    """
    fig = plt.figure(figsize=(12,20))
    trial_number = 100 #the number of trials 
    
    #A: calculating STDP curve
    delta_t_list = np.concatenate([np.array([-80+2*i for i in range(40)]), np.array([2*i+2 for i in range(40)])]) #the list of the time difference of STDP stimulations
    for condition in range(5):
        result_stdp = np.zeros((2,np.size(delta_t_list)))
        result_stdp_pre = np.zeros(2)
        w_initial = np.array([0.4, 0.5, 0.6, 0.5, 0.5])[condition]  # initial value of synaptic weights of stimulated neurons
        pre_neuron = np.array([20, 20, 20, 10, 40])[condition]  # the number of stimulated neurons
        for i in range(2):
            for trial in range(trial_number):
                g_index = feedforwardmodel.up_down_parameters(state=['down','up'][i])
                result_stdp_pre[i] += calculate_stdp_preonly(g_index=g_index, w_initial=w_initial, pre_neuron=pre_neuron)
                for j in range(np.size(delta_t_list)):
                    result_stdp[i,j] += calculate_stdp(g_index=g_index, w_initial=w_initial, pre_neuron=pre_neuron, delta_t=delta_t_list[j])
            result_stdp[i, :] /= trial_number
            result_stdp_pre[i] /= trial_number
            figplot.plot_stdp_curve(delta_t=delta_t_list, g=result_stdp[i, :], preonly=result_stdp_pre[i], g_min=-0.8, g_max=2.7, t_min=np.min(delta_t_list), t_max=np.max(delta_t_list), aspect=0.75, fig=fig, plot_x=5, plot_y=2, plot_area=2*condition+i+1)
    plt.savefig(file_name)

def plot_fr_dependency(file_name):
    """
    calculating the relationship between synaptic weight changes and the g_average value
    """
    fig = plt.figure(figsize=(12,20))
    trial_number = 100 #the number of trials 
    
    result_fr = np.zeros((3,np.size(g_average_list[:,0])))

    for j in range(np.size(g_average_list[:,0])):
        for trial in range(trial_number):
            result_fr[0,j] += calculate_stdp(g_index=j, w_initial=0.5, pre_neuron=20, delta_t=10)  #the case of delta_t = 10msec
            result_fr[1,j] += calculate_stdp(g_index=j, w_initial=0.5, pre_neuron=20, delta_t=-10)  #the case of delta_t = -10msec
            result_fr[2,j] += calculate_stdp_preonly(g_index=j, w_initial=0.5, pre_neuron=20)  #the case of pre-only stimulations
    
    result_fr /= trial_number
    for i in range(3):
        figplot.plot_scatter(x=1000*g_average_list[:,1], y=result_fr[i,:], x_min=0.0, x_max=np.max(1000*g_average_list), y_min=-0.8, y_max=1.6, aspect=0.8, fig=fig, plot_x=1, plot_y=3, plot_area=i+1)
    plt.savefig(file_name)

def plot_weight_dependency(file_name):
    """
    calculating the relationship between synaptic weight changes, initial synaptic weights, and the number of the stimulated neurons
    """
    fig = plt.figure(figsize=(10,10))
    trial_number = 200 #the number of trials 
    
    w_initial_list = np.array([0.02 * (i+5) for i in range(26)])
    pre_neuron_list = np.array([10, 20, 40])
    result_weight = np.zeros((2, np.size(w_initial_list), np.size(pre_neuron_list)))
    for state in range(2):
        g_index = feedforwardmodel.up_down_parameters(state=['down','up'][state])
        for i in range(np.size(w_initial_list)):
            for j in range(np.size(pre_neuron_list)):
                for trial in range(trial_number):
                    result_weight[state, i, j] += calculate_stdp_preonly(g_index=g_index, w_initial=w_initial_list[i], pre_neuron=pre_neuron_list[j])
    
    result_weight /= trial_number
    
    for state in range(2):
        for j in range(np.size(pre_neuron_list)):
            figplot.plot_scatter_weight(x=w_initial_list, y=result_weight[state,:,j], x_min=np.min(w_initial_list), x_max=np.max(w_initial_list), y_min=-0.8, y_max=2.7, aspect=0.8, fig=fig, plot_x=3, plot_y=2, plot_area=state+2*j+1)
    plt.savefig(file_name)


plot_stdp(file_name='figure/feedforward/stdp.pdf')
plot_fr_dependency(file_name='figure/feedforward/firingrate.pdf')
plot_weight_dependency(file_name='figure/feedforward/weight.pdf')



t2 = time.time()
elapsed_time = t2-t1
print(f"Total time：{elapsed_time}")

