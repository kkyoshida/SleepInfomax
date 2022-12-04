import numpy as np
import common
import slowwavemodel


class SlowwaveReactivationDecrease(slowwavemodel.Slowwave):
    """
    Used for the task simulation in which reactivation decreases
    """
    
    def __init__(self, epoch_length, epoch_number, pre_w_initial, reac_state, reac_neuron, reac_rate, wake_epoch_number, wake_rate, w_pre_list_initial, plasticity_option):
        self.w_pre_list_initial = w_pre_list_initial  # the list of initial synaptic weight
        super().__init__(epoch_length, epoch_number, pre_w_initial, plasticity_option, plot_option=True, connect_prob_ee=0.05, connect_prob_ie=0.3)
        self.reac_state = reac_state  # state in which reactivation occurs
        self.reac_neuron = reac_neuron  # the number of presynaptic neurons given reactivated input
        self.reac_rate = reac_rate  # the firing rate of presynaptic inputs during sleep
        self.wake_epoch_number = wake_epoch_number  # the number of awake epochs
        self.wake_rate = wake_rate  # the firing rate of presynaptic inputs during awake

    def pregenerator(self, section):
        """
        generate presynaptic inputs
        """
        super().pregenerator(section)
        reac_rand = np.random.rand(self.epoch_length)
        reac_rate_list = np.array([self.reac_rate - 0.0025 / 900000 * (section * self.epoch_length + i) for i in range(self.epoch_length)])
        if section < self.wake_epoch_number:
            self.reac = np.where(reac_rand < self.wake_rate, 1, 0)  # inputs during awake
        else:
            self.reac = np.where(reac_rand < reac_rate_list, 1, 0)  # inputs during sleep
    
    def learning_rate(self, t, section, current_state):
        """
        set the learning rate of synaptic changes
        """
        if t + self.epoch_length*section >= 200000: 
            if section < self.wake_epoch_number:
                return 0.0
            else:
                recent = self.state_plot[t + self.epoch_length*section - 50 : t + self.epoch_length*section]
                if (self.reac_state == 'globalup' and all(recent != 8)) or (self.reac_state == 'localup' and all(recent != 6)) or (self.reac_state == 'allup'):
                    return self.params.alpha
                else:
                    return 0.0 
        else:
            return 0.0

    def stdp_operation(self, t, section):
        if section < self.wake_epoch_number:
            if self.reac[t] == 1:
                self.x[t, 0 : 2 * self.reac_neuron] = self.reac[t]
        else:
            if (self.reac[t] == 1) and (self.state == 6):
                self.x[t, 0 : self.reac_neuron] = self.reac[t]
            if (self.reac[t] == 1) and (self.state == 8):
                self.x[t, self.reac_neuron : 2 * self.reac_neuron] = self.reac[t]

    def calculate_adapt(self, t, section, adapt_previous, y_e_now):
        """
        During awake, adaptation current does not change. 
        """
        if section < self.wake_epoch_number:
            return adapt_previous
        else:
            return adapt_previous*np.exp(-1./self.tau_adapt) + self.beta*y_e_now

    def set_sleepwake_variables(self):
        self.w_pre = self.w_pre_list_initial
        self.u_e = (-67.0) * np.ones(self.exc_total)  # initial membrane potential is set to high value for generating the awake firing pattern
        self.adapt = 0.025 * np.ones(self.exc_total)  # adaptation current during awake is fixed to this value

class SlowwaveReactivationDecreaseDown(SlowwaveReactivationDecrease):
    def learning_rate(self, t, section, current_state):
        """
        set the learning rate of synaptic changes
        """
        if t + self.epoch_length*section >= 200000:
            if section < self.wake_epoch_number:
                return 0.0
            else:
                recent = self.state_plot[t + self.epoch_length*section-50 : t + self.epoch_length*section]
                if (self.reac_state == 'globalup' and all(recent != 8) and all(recent != 2)) or (self.reac_state == 'localup' and all(recent != 6) and all(recent != 0)) or (self.reac_state == 'allup'):
                    return self.params.alpha
                else:
                    return 0.0 
        else:
            return 0.0
        
    def stdp_operation(self, t, section):
        if section < self.wake_epoch_number:
            if self.reac[t] == 1:
                self.x[t, 0 : 2 * self.reac_neuron] = self.reac[t]
        else:
            if (self.reac[t] == 1) and (self.state == 6 or self.state == 0):
                self.x[t, 0 : self.reac_neuron] = self.reac[t]
            if (self.reac[t] == 1) and (self.state == 8 or self.state == 2):
                self.x[t, self.reac_neuron : 2 * self.reac_neuron] = self.reac[t]
        
def calculate_performance_twopre(w_pre_list_initial, reac_neuron):
    """
    calculate the mean firing rate of the postsynaptic neuron during the task for given presynaptic weights 'w_pre_list_initial'
    """
    total_epoch = 20
    sw = SlowwaveReactivationDecrease(epoch_length=50000, epoch_number=total_epoch, pre_w_initial=0.0, reac_state=0, reac_neuron=reac_neuron, reac_rate=0,\
        wake_epoch_number=total_epoch, wake_rate=0.005, w_pre_list_initial=w_pre_list_initial, plasticity_option=False)
    sw.simulate()
    return 1000 * np.mean(sw.y_e_plot[10000:, 0])



