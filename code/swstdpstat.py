import time
import numpy as np
import matplotlib.pyplot as plt
import slowwavemodel
import figplot

t1 = time.time()

np.random.seed(3)

def calculate_stdp_preonly_slowwave(stim_state, pre_w_initial, stim_neuron):
    """
    calculate synaptic weight changes after pre-only stimulation in stim_state
    """
    success = False
    while not success:
        cal_stdp = slowwavemodel.SlowwaveSTDPPreonly(epoch_length=50000, pre_w_initial=pre_w_initial, stim_state=stim_state, stim_neuron=stim_neuron)
        cal_stdp.simulate()
        if cal_stdp.stim_count == cal_stdp.stim_num_total:
            success = True
    return (np.mean(cal_stdp.w_pre[0:cal_stdp.stim_neuron]) - cal_stdp.pre_w_initial) / cal_stdp.pre_w_initial * 100

trial_number = 1600 
prew_list = np.array([0.5, 0.09])
stimneuron_list = np.array([20, 40])
result_stdp_pre = np.zeros((trial_number, 4, np.size(prew_list)))
state_list = np.array([0,2,6,8]) #0: global down, 2:local down, 6: global up, 8: local up

for prew_stim in range(np.size(prew_list)):
    for i in range(4):
        for trial in range(trial_number):
            result_stdp_pre[trial, i, prew_stim] = calculate_stdp_preonly_slowwave(stim_state=state_list[i], pre_w_initial=prew_list[prew_stim], stim_neuron=stimneuron_list[prew_stim])  # pre-only stimulations

np.save('data/stdpstat', result_stdp_pre)

t2 = time.time()
elapsed_time = t2-t1
print(f"Total time：{elapsed_time}")