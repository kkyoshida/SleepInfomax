import time
import numpy as np
import matplotlib.pyplot as plt
import slowwavemodel
import figplot

t1 = time.time()

np.random.seed(3)

def calculate_state_fr():
    """
    Calculate mean firing rates of excitatory populations in each state
    """
    
    sw = slowwavemodel.Slowwave(epoch_length=100000, epoch_number=1, pre_w_initial=0.0, plasticity_option=False, plot_option=True, connect_prob_ee=0.05, connect_prob_ie=0.3) 
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
    
    state_fr = np.zeros(4)
    trial_number = 30
    for i in range(trial_number):
        state_fr += calculate_state_fr()
    fig = plt.figure(figsize=(4,4))
    state_fr /= trial_number
    figplot.plot_bar(g=state_fr, g_min=0.0, g_max=8.5, aspect=1.0, labels=['Global down', 'Local down', 'Global up', 'Local up'], fig = fig, plot_x=1, plot_y=1, plot_area=1)
    plt.savefig(file_name)

def calculate_u_distribution():
    """
    Calculate the distribution of membrane potential
    """
    sw = slowwavemodel.Slowwave(epoch_length=100000, epoch_number=1, pre_w_initial=0.0, plasticity_option=False, plot_option=True, connect_prob_ee=0.05, connect_prob_ie=0.3) 
    sw.simulate()
    return np.mean(sw.u_e_plot[10000:100000, 0:sw.exc_popu], axis=1)
    
def plot_u_distribution(file_name):
    """
    Plot the distribution of membrane potential and the threshold of transitions between up and down states
    """
    
    fig = plt.figure(figsize=(4,4))
    sw = slowwavemodel.Slowwave(epoch_length=100000, epoch_number=1, pre_w_initial=0.0, plasticity_option=False, plot_option=True, connect_prob_ee=0.05, connect_prob_ie=0.3)
    trial_number = 30
    u_distribution = np.zeros((trial_number, 90000))
    for i in range(trial_number):
        u_distribution[i] = calculate_u_distribution()
    figplot.plot_histogram(np.ravel(u_distribution), sw.threshold_to_down, sw.threshold_to_up, fig, 1, 1, 1)
    plt.savefig(file_name)

    
plot_state_fr('figure/slowwave/firingrate.pdf')
plot_u_distribution('figure/slowwave/udistribution.pdf')

t2 = time.time()
elapsed_time = t2-t1
print(f"Total time：{elapsed_time}")

