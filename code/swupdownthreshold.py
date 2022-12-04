import time
import numpy as np
import matplotlib.pyplot as plt
import slowwavemodel
import figplot

t1 = time.time()

np.random.seed(3)

class Udthreshold(slowwavemodel.Slowwave): 
    def __init__(self, epoch_length, epoch_number, pre_w_initial, thre_ud, thre_du, plasticity_option=True, plot_option=True, connect_prob_ee=0.05, connect_prob_ie=0.3):
        super().__init__(epoch_length, epoch_number, pre_w_initial, plasticity_option=True, plot_option=True, connect_prob_ee=0.05, connect_prob_ie=0.3) 
        self.threshold_to_down = thre_ud  # transition threshold to down states [mV]
        self.threshold_to_up = thre_du  # transition threshold to up states [mV]

def calculate_state_fr(thre_ud, thre_du):
    """
    Calculate mean firing rates of excitatory populations in each state
    """
    
    sw = Udthreshold(epoch_length=100000, epoch_number=1, pre_w_initial=0.0, thre_ud=thre_ud, thre_du=thre_du, plasticity_option=False, plot_option=True, connect_prob_ee=0.05, connect_prob_ie=0.3) 
    sw.simulate()
    sw.calculate_fr()
    if sw.fr_gso_down == -1:
        print('error')
    else:
        return np.array([sw.fr_gso_down, sw.fr_ldw_down, sw.fr_gso_up, sw.fr_ldw_up])

def plot_state_fr(file_name):
    """
    Plot and calculate mean excitatory firing rates by simulating and averaging many trials
    """
    
    thre_ud_list = np.array([-70.3+0.09*i for i in range(21)])
    thre_du_list = np.array([-68.5+0.105*i for i in range(21)])
    result = np.zeros((np.size(thre_ud_list), np.size(thre_du_list), 2))
    for j in range(np.size(thre_ud_list)):
        for k in range(np.size(thre_du_list)):
            state_fr = np.zeros(4)
            trial_number = 4 
            for i in range(trial_number):
                state_fr += calculate_state_fr(thre_ud_list[j], thre_du_list[k])
            state_fr /= trial_number
            result[j, k, 0] = state_fr[3] - state_fr[2]
            result[j, k, 1] = state_fr[1] - state_fr[0]
    fig = plt.figure(figsize=(8,8))
    for j in range(2):
        ax = fig.add_subplot(2, 1, j+1)
        ax.set_yticks([0, np.size(thre_ud_list)-1])
        ax.set_yticklabels([thre_ud_list[0], thre_ud_list[-1]])
        ax.set_xticks([0, np.size(thre_du_list)-1])
        ax.set_xticklabels([thre_du_list[0], thre_du_list[-1]])
        image = ax.imshow(result[:, :, j], origin='lower')
        fig.colorbar(image, ax=ax)
    plt.savefig(file_name)
plot_state_fr('figure/slowwave/updownthreshold.pdf')


t2 = time.time()
elapsed_time = t2-t1
print(f"Total time：{elapsed_time}")
