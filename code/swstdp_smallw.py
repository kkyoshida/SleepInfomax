import time
import numpy as np
import matplotlib.pyplot as plt
import slowwavemodel
import figplot

t1 = time.time()

np.random.seed(4)

def calculate_stdp_slowwave(delta_t, stim_state):
    """
    calculate synaptic weight changes after the STDP stimulation with time difference delta_t in stim_state
    """
    success = False
    while not success:
        cal_stdp = slowwavemodel.SlowwaveSTDP(epoch_length=50000, pre_w_initial=0.09, delta_t=delta_t, stim_state=stim_state, stim_neuron=40)
        cal_stdp.simulate()
        if cal_stdp.stim_count == cal_stdp.stim_num_total:
            success = True
    return (np.mean(cal_stdp.w_pre[0:cal_stdp.stim_neuron]) - cal_stdp.pre_w_initial) / cal_stdp.pre_w_initial * 100

def calculate_stdp_preonly_slowwave(stim_state):
    """
    calculate synaptic weight changes after pre-only stimulation in stim_state
    """
    success = False
    while not success:
        cal_stdp = slowwavemodel.SlowwaveSTDPPreonly(epoch_length=50000, pre_w_initial=0.09, stim_state=stim_state, stim_neuron=40)
        cal_stdp.simulate()
        if cal_stdp.stim_count == cal_stdp.stim_num_total:
            success = True
    return (np.mean(cal_stdp.w_pre[0:cal_stdp.stim_neuron]) - cal_stdp.pre_w_initial) / cal_stdp.pre_w_initial * 100


def calculate_and_plot_stdp(file_name):
    """
    plot the STDP curve
    """
    
    fig = plt.figure(figsize=(20, 20))
    trial_number = 20
    
    delta_t_list = np.concatenate([np.array([-80+2*i for i in range(40)]), np.array([2*i+2 for i in range(40)])])
    result_stdp = np.zeros((4,np.size(delta_t_list)))
    result_stdp_pre = np.zeros(4)
    state_list = np.array([0,2,6,8]) #0: global down, 2:local down, 6: global up, 8: local up
    
    for i in range(4):
        for trial in range(trial_number):
            result_stdp_pre[i] += calculate_stdp_preonly_slowwave(stim_state=state_list[i])  # pre-only stimulations
            for j in range(np.size(delta_t_list)):
                result_stdp[i,j] += calculate_stdp_slowwave(delta_t=delta_t_list[j], stim_state=state_list[i])  # STDP stimulations            
    result_stdp /= trial_number
    result_stdp_pre /= trial_number
    
    #plot STDP curve
    for i in range(4):
        figplot.plot_stdp_curve(delta_t=delta_t_list, g=result_stdp[i,:], preonly=result_stdp_pre[i], g_min=-10.0, g_max=120.0, t_min=np.min(delta_t_list), t_max=np.max(delta_t_list), aspect=0.75, fig=fig, plot_x=3, plot_y=4, plot_area=i+1)
        figplot.plot_stdp_curve(delta_t=delta_t_list, g=result_stdp[i,:], preonly=result_stdp_pre[i], g_min=-10.0, g_max=150.0, t_min=np.min(delta_t_list), t_max=np.max(delta_t_list), aspect=0.75, fig=fig, plot_x=3, plot_y=4, plot_area=i+5)
        figplot.plot_stdp_curve(delta_t=delta_t_list, g=result_stdp[i,:], preonly=result_stdp_pre[i], g_min=-15.0, g_max=120.0, t_min=np.min(delta_t_list), t_max=np.max(delta_t_list), aspect=0.75, fig=fig, plot_x=3, plot_y=4, plot_area=i+9)
    plt.savefig(file_name)

calculate_and_plot_stdp('figure/slowwave/stdp-smallw.pdf')

t2 = time.time()
elapsed_time = t2-t1
print(f"Total time：{elapsed_time}")

