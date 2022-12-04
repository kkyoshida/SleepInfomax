import time
import numpy as np
import matplotlib.pyplot as plt
import swtaskreactivation
import figplot

t1 = time.time() 

np.random.seed(5)

trial_number = 100
state_list = ['allup', 'globalup', 'localup']

epolen = 50000  # time length per one epoch
eponum = 18  # the number of total epoch 
epowake = 0  # the number of awake epoch
reactivation_neuron = 40  # the number of presynaptic neurons

def cal_perf(w_list, reac_neuron, trial_number):
    # calculate base firing rates
    result_performance = np.zeros(trial_number)
    for i in range(trial_number):
        result_performance[i] = swtaskreactivation.calculate_performance_twopre(w_pre_list_initial=w_list, reac_neuron=reac_neuron)
    return result_performance

base_performance = np.mean(cal_perf(w_list=np.zeros(100), reac_neuron=reactivation_neuron, trial_number=trial_number))
print(base_performance)

pre_w_initial = 0.09  
result_perf_beforesleep = cal_perf(w_list=pre_w_initial*np.ones(100), reac_neuron=reactivation_neuron, trial_number=trial_number)

result_perf = np.zeros((trial_number, 3))
result_weight_globalpre = np.zeros((trial_number, 3, epolen * eponum))
result_weight_localpre = np.zeros((trial_number, 3, epolen * eponum))
conv_time = 200000
v = np.ones(conv_time) / conv_time
#result_reactivation_global = np.zeros((trial_number, 3, epolen * eponum - conv_time + 1))
#result_reactivation_local = np.zeros((trial_number, 3, epolen * eponum - conv_time + 1))

for i in range(3):
    for trial in range(trial_number):
        sw = swtaskreactivation.SlowwaveReactivationDecreaseDown(epoch_length=epolen, epoch_number=eponum, pre_w_initial=pre_w_initial, reac_state=state_list[i], reac_neuron=reactivation_neuron, reac_rate=0.0075, wake_epoch_number=epowake, wake_rate=0.005, w_pre_list_initial=pre_w_initial*np.ones(100), plasticity_option=True)
        sw.simulate()

        result_weight_globalpre[trial, i, :] = np.mean(sw.w_pre_plot[:, 0 : sw.reac_neuron], axis=1)   
        result_weight_localpre[trial, i, :] = np.mean(sw.w_pre_plot[:, sw.reac_neuron : 2 * sw.reac_neuron], axis=1)  

        #state_global = np.where(sw.state_plot == 6, 1, 0)
        #state_local = np.where(sw.state_plot == 8, 1, 0)
        #result_reactivation_global[trial, i, :] = np.convolve(sw.y_e_plot[:, 0] * state_global, v, mode='valid') / np.convolve(state_global, v, mode='valid') * 1000
        #result_reactivation_local[trial, i, :] = np.convolve(sw.y_e_plot[:, 0] * state_local, v, mode='valid') / np.convolve(state_local, v, mode='valid') * 1000

        # calculate performance
        performance = swtaskreactivation.calculate_performance_twopre(w_pre_list_initial=sw.w_pre_plot[epolen*eponum - 1], reac_neuron=sw.reac_neuron)  
        result_perf[trial, i] = performance  

np.save('data/result_weight_globalpre_down', result_weight_globalpre)
np.save('data/result_weight_localpre_down', result_weight_localpre)
#np.save('data/result_reactivation_global_down', result_reactivation_global)
#np.save('data/result_reactivation_local_down', result_reactivation_local)
np.save('data/result_perf_down', result_perf)
np.save('data/result_perf_beforesleep_down', result_perf_beforesleep)
np.save('data/base_performance_down', base_performance)

t2 = time.time()
elapsed_time = t2-t1
print(f"Total time：{elapsed_time}")
