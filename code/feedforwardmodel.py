import numpy as np
import common

def pre_fr_list():
    return np.array([0.0001*i for i in range(50)])  #the list of presynaptic firing rates

def up_down_parameters(state):
    """
    set the frequency of up and down states
    """
    if state == 'down':
        g_index = 1
    elif state == 'up':
        g_index = 20 
    return g_index

def set_basic_parameters():
    """
    set the basic parameters used in the model
    """
    w_other = 0.5  # synaptic weights of non-stimulated neuron
    other_pre_neuron = 100  # the number of non-stimulated neuron
    return w_other, other_pre_neuron

def stdp_variables(g_index, g_average_list, w_initial, pre_neuron):
    """
    set the parameters of STDP stimulations
    """
    pre_fr = g_average_list[g_index, 0]  # firing rates of non-stimulated neurons [1/msec]
    pre_neuron = pre_neuron  # the number of stimulated neurons
    simulation_time = 20000  # total simulation time [msec]
    w_initial = w_initial  # initial synaptic weights of stimulated neurons
    g_average = g_average_list[g_index, 1]  # set the value of g_average
    stimulation_rate = 0.0002  # frequency of STDP stimulations
    return pre_fr, pre_neuron, simulation_time, w_initial, g_average, stimulation_rate

def set_g_average(pre_fr):
    """
    calculate the g_average values for each firing rate of non-stimulated neurons
    """
    params = common.ConstantsAndFunctions()
    w_other, other_pre_neuron = set_basic_parameters()
    trial_prelim = 100000
    g_ave = np.zeros(trial_prelim)
    for trial in range(trial_prelim):  # calculate the average of trial_prelim trials
        total_t = 1000  # simulation length [msec]
        pre_input = np.random.binomial(other_pre_neuron, pre_fr, total_t)  # generate synaptic inputs of non-stimulated neuron in each time-bin
        h_other_prelim = 0
        for t in range(total_t):
            h_other_prelim = params.calculate_trace(h_other_prelim, pre_input[t], params.tau_m_e)
        g_ave[trial] = params.g_e(params.u_r + h_other_prelim * w_other)  # calculate the g value after 1000 msec
    return np.mean(g_ave)

class Feedforward:
    """
    Basic feedforward model. The synaptic weight changes according to the infomax rule. 
    """
    def __init__(self, pre_fr, pre_neuron, simulation_time, w_initial, g_average):
        self.params = common.ConstantsAndFunctions()
        self.pre_fr = pre_fr  # firing rates of non-stimulated neurons [1/msec]
        self.pre_neuron = pre_neuron  # the number of stimulated neurons
        self.simulation_time = simulation_time  # total simulation time [msec]
        self.w_initial = w_initial  # initial synaptic weights of stimulated neurons    
        self.g_average = g_average  # set the value of \overline{g}
        self.w_other, self.other_pre_neuron = set_basic_parameters()  # synaptic weights of non-stimulated neurons, and the number of non-stimulated neurons
        self.set_variables()  # set the initial value of the variables
    
    def set_variables(self): 
        """
        set initial values
        """
        self.previous_spike = -10000  # the time difference of the most recent spike and now [msec]  
        self.h = np.zeros(self.pre_neuron)  # integration of synaptic inputs of stimulated neuron
        self.h_other = 0.0  # integration of synaptic inputs of non-stimulated neuron
        self.large_c = np.zeros(self.pre_neuron)  # auxiliary value in the infomax rule
        self.w = np.ones(self.pre_neuron) * self.w_initial  # synaptic weights of stimulated neurons
        self.didt = np.ones(self.pre_neuron) * self.w_initial  # synaptic changes by the information term
        self.dphidt = np.ones(self.pre_neuron) * self.w_initial  # synaptic changes by the synaptic cost term
            
    def input_generator(self):
        """
        Generating spikes of stimulated neuron. No spontaneous spikes in normal cases. 
        """
        return np.zeros((self.simulation_time, self.pre_neuron))
    
    def other_input_generator(self):
        """
        Generating spikes of non-stimulated neuron. 
        """
        return np.random.binomial(self.other_pre_neuron,self.pre_fr, self.simulation_time)
    
    def output_random_generator(self):
        """
        Random numbers used for generating postsynaptic spikes 
        """
        return np.random.rand(self.simulation_time)

    def outputgenerator(self, r, t_now):
        """
        Generating postsynaptic spikes according to firing probability in each time bin
        """
        if self.random_numbers[t_now] < r:
            return 1
        else:
            return 0

    def learning_rate(self,t):
        """
        learning rate of synaptic weight
        """
        if t >= self.params.plasticity_start_t: #Synaptic changes start at time of self.params.plasticity_start_t
            return self.params.alpha
        else:
            return 0.0

    def g_e_feedforward(self, u_now): 
        return self.params.g_e(u_now)
        
    def small_cj_feedforward(self, u_now, rho_now, y_now, h_j_now): 
        return self.params.small_cj(u_now, rho_now, y_now, h_j_now)
       
    def changeonetime(self, t):
        self.h = self.params.calculate_trace(self.h, self.x[t], self.params.tau_m_e)
        self.h_other = self.params.calculate_trace(self.h_other, self.x_other[t], self.params.tau_m_e)
        self.u = self.params.u_r + np.dot(self.h, self.w) + self.h_other * self.w_other  # membrane potential changes according to inputs
        self.rho = self.g_e_feedforward(self.u) * self.params.r(t, self.previous_spike)  # firing probability 
        self.y = self.outputgenerator(self.rho, t)  # generate postsynaptic spikes
        self.bpost = self.params.bpost(self.rho, self.g_average * self.params.r(t, self.previous_spike), self.y)
        self.small_c = self.small_cj_feedforward(self.u, self.rho, self.y, self.h)
        self.large_c = self.params.calculate_trace(self.large_c, self.small_c, self.params.tau_c)
        self.w = self.w + self.params.synaptic_change(self.learning_rate(t), self.large_c, self.bpost, self.w, self.x[t])  # synaptic weight change
        self.didt = self.didt + self.params.synaptic_change_info(self.learning_rate(t), self.large_c, self.bpost, self.w, self.x[t])
        self.dphidt = self.dphidt + self.params.synaptic_change_cost(self.learning_rate(t), self.large_c, self.bpost, self.w, self.x[t])
        if self.y == 1:
            self.previous_spike = t  # set the time of last spikes
        
    def change(self, plot_option=False):
        if plot_option:
            self.set_plots()
        self.x = self.input_generator()  # generating spikes of stimulated neuron
        self.x_other = self.other_input_generator()  # generating spikes of non-stimulated neuron
        self.random_numbers = self.output_random_generator() 
        for t in range(self.simulation_time):
            self.changeonetime(t)
            if plot_option:
                self.x_plot[t] = self.x[t]
                self.y_plot[t] = self.y
                self.large_c_plot[t] = self.large_c
                self.bpost_plot[t] = self.bpost
                self.w_plot[t] = self.w / self.w_initial
                self.g_average_plot[t] = self.g_average
                self.g_plot[t] = self.g_e_feedforward(self.u)
                self.rho_plot[t] = self.rho
                self.didt_plot[t] = self.didt / self.w_initial
                self.dphidt_plot[t] = self.dphidt / self.w_initial
       
    def set_plots(self):
        """
        set variables for plotting
        """
        self.x_plot = np.zeros((self.simulation_time, self.pre_neuron), dtype=int)
        self.y_plot = np.zeros(self.simulation_time, dtype=int)
        self.large_c_plot = np.zeros((self.simulation_time, self.pre_neuron))
        self.bpost_plot = np.zeros(self.simulation_time)
        self.w_plot = np.zeros((self.simulation_time, self.pre_neuron))
        self.g_average_plot = np.zeros(self.simulation_time)
        self.g_plot = np.zeros(self.simulation_time)
        self.rho_plot = np.zeros(self.simulation_time)
        self.didt_plot = np.zeros((self.simulation_time, self.pre_neuron))
        self.dphidt_plot = np.zeros((self.simulation_time, self.pre_neuron))

class FeedforwardSTDPPreonly(Feedforward):
    """
    Simulating pre-only stimulations in the feedforward model. 
    """
    def __init__(self, pre_fr, pre_neuron, simulation_time, w_initial, g_average, stimulation_rate):
        super().__init__(pre_fr, pre_neuron, simulation_time, w_initial, g_average)
        self.stimulation_rate = stimulation_rate  # frequency of STDP stimulations

    def input_generator(self):
        """
        generate stimulations for stimulated-neuron
        """
        stimulation_input = np.array([[np.where(j % (int(1.0 / self.stimulation_rate)) == 100 and j >= self.params.plasticity_start_t, 1, 0) for i in range(self.pre_neuron)] for j in range(self.simulation_time)])
        return stimulation_input
        
class FeedforwardSTDP(FeedforwardSTDPPreonly):
    """
    Simulating STDP stimulations in the feedforward model. 
    """
    def __init__(self, pre_fr, pre_neuron, simulation_time, w_initial, g_average, stimulation_rate, delta_t):
        super().__init__(pre_fr, pre_neuron, simulation_time, w_initial, g_average, stimulation_rate)
        self.delta_t = delta_t  # time difference between presynaptic and postsynaptic spikes in STDP stimulations        
    
    def outputgenerator(self, r, t_now):
        """
        generate postsynaptic spikes including STDP stimulations and spontaneous spikes
        """
        if (self.random_numbers[t_now] < r) or (t_now % (int(1.0 / self.stimulation_rate)) == (100 + self.delta_t) and t_now >= self.params.plasticity_start_t):
            return 1
        else:
            return 0




            


            
