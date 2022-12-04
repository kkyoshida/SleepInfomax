import numpy as np
import common

class Slowwave:
    """
    The basic model generating slow waves. 
    """

    def __init__(self, epoch_length, epoch_number, pre_w_initial, plasticity_option=True, plot_option=True, connect_prob_ee=0.05, connect_prob_ie=0.3):
        self.params = common.ConstantsAndFunctions()
        self.epoch_length = epoch_length  # the length of a epoch [msec]
        self.epoch_number = epoch_number  # the number of total epochs
        self.pre_w_initial = pre_w_initial  # the initial value of feedforward synaptic weights
        self.plasticity_option = plasticity_option  # whether feedforward synaptic weights are plastic (True) or not plastic (False)
        self.plot_option = plot_option  # whether prepare plotting or not
        self.connect_prob_ee = connect_prob_ee  # the connecting probability from an excitatory to an excitatory neuron in different populations
        self.connect_prob_ie = connect_prob_ie  # the connecting probability from an excitatory to an inhibitory neuron in different populations
        self.exc_popu = 200  # the number of excitatory neurons in one population
        self.inh_popu = 50  # the number of inhibitory neurons in one population
        self.population = 4  # the number of populations
        self.exc_total = self.exc_popu*self.population  # the number of total excitatory neurons
        self.inh_total = self.inh_popu*self.population  # the number of total inhibitory neurons
        self.tau_adapt = 1500.0  # the time constant of adapdation currents [msec]
        self.beta = 0.0077  # constant related to adaptation currents [mV/msec]
        self.pre_neuron = 100  # We prepare 100 presynaptic neurons for convenience, but a part of neurons does not emit spikes at all. 
        self.threshold_to_down = -69.75  # transition threshold to down states [mV]
        self.threshold_to_up = -68.25  # transition threshold to up states [mV]

        # recurrent synaptic weight
        self.w_rec_ee = 0.16  # excitatory to excitatory [mV]
        self.w_rec_ie = 0.66  # excitatory to inhibitory [mV]
        self.w_rec_ei = -0.14  # inhibitory to excitatory [mV]
        self.w_rec_ii = 0.0  # no connections from inhibitory to inhibitory

        # set the connections of recurrent networks between different populations
        self.w_rec_matrix_ee = np.where(np.random.rand(self.exc_total, self.exc_total) < self.connect_prob_ee, self.w_rec_ee, 0.0)  #w_rec_matrix_ee[i,j] is synapic weight from excitatory i to excitatory j
        self.w_rec_matrix_ie = np.where(np.random.rand(self.exc_total, self.inh_total) < self.connect_prob_ie, self.w_rec_ie, 0.0)  #w_rec_matrix_ie[i,j] is synapic weight from excitatory i to inhibitory j
        self.w_rec_matrix_ei = np.zeros((self.inh_total, self.exc_total))
        self.w_rec_matrix_ii = np.zeros((self.inh_total, self.inh_total))

        # set the connections of recurrent networks within a population
        for i in range(self.population):
            self.w_rec_matrix_ee[self.exc_popu * i : self.exc_popu * (i+1), self.exc_popu * i : self.exc_popu * (i+1)] = self.w_rec_ee * (np.ones((self.exc_popu, self.exc_popu)) - np.identity(self.exc_popu))
            self.w_rec_matrix_ie[self.exc_popu * i : self.exc_popu * (i+1), self.inh_popu * i : self.inh_popu * (i+1)] = self.w_rec_ie * np.ones((self.exc_popu, self.inh_popu)) 
            self.w_rec_matrix_ei[self.inh_popu * i : self.inh_popu * (i+1), self.exc_popu * i : self.exc_popu * (i+1)] = self.w_rec_ei * np.ones((self.inh_popu, self.exc_popu))
            self.w_rec_matrix_ii[self.inh_popu * i : self.inh_popu * (i+1), self.inh_popu * i : self.inh_popu * (i+1)] = self.w_rec_ii * (np.ones((self.inh_popu, self.inh_popu)) - np.identity(self.inh_popu))

        self.set_variables()

    def set_variables(self):
        self.y_e = np.zeros(self.exc_total, dtype=int)  # spikes of excitatory neurons
        self.y_i = np.zeros(self.inh_total, dtype=int)  # spikes of inhibitory neurons
        self.u_i = self.params.u_r * np.ones(self.inh_total)  # membrane potential of inhibitory neurons [mV]
        self.state = -1  # state of population 1
        self.up_or_down = np.ones(self.population)  # up/down states of each population
        self.previous_spike_e = -10000*np.ones(self.exc_total)  # previous spikes of excitatory neurons
        self.previous_spike_i = -10000*np.ones(self.inh_total)  # previous spikes of inhibitory neurons
        self.set_sleepwake_variables() 
        if self.plasticity_option:
            self.h = np.zeros(self.pre_neuron)  # EPSP time-course of presynaptic neurons
            self.small_c = np.zeros(self.pre_neuron)  
            self.large_c = np.zeros(self.pre_neuron)

    def set_sleepwake_variables(self):
        self.w_pre = self.pre_w_initial * np.ones(self.pre_neuron)  # feedforward synaptic weights [mV]
        self.u_e = self.params.u_r * np.ones(self.exc_total)  # membrane potential of excitatory neurons [mV]
        self.adapt = np.zeros(self.exc_total)  # adaptation currents [mV/msec]
        
    def pregenerator(self, section):
        """
        no spontaneous presynaptic spikes except for STDP stimulations
        """
        self.x = np.zeros((self.epoch_length, self.pre_neuron)) 
    
    def postgenerator(self):
        """
        Random numbers used for generating postsynaptic spikes 
        """
        self.y_e_rand = np.random.rand(self.epoch_length, self.exc_total)
        self.y_i_rand = np.random.rand(self.epoch_length, self.inh_total)
    
    def learning_rate(self, t, section, current_state):
        """
        learning rate of synaptic weight
        """
        if t + self.epoch_length*section >= self.params.plasticity_start_t:
            return self.params.alpha
        else:
            return 0.0

    def classify_up_down(self, t, previous_state):  
        """
        Classifying each population into up state (7), down state (1)
        """
        current_u_mean = np.mean(self.u_e.reshape(self.population, self.exc_popu), axis=1)  # mean membrane potential of each excitatory population
        state_candidate = np.where(previous_state == 1, 1, 0) * np.where(current_u_mean > self.threshold_to_up, 7, 1)\
        +np.where(previous_state == 7, 1, 0) * np.where(current_u_mean < self.threshold_to_down, 1, 7)  # determine the current states (up or down) according to the previous states
        return state_candidate 
        
    def stageclassify(self, t):
        """
        Classifying excitatory population 1 into global down (0), local down (2), global up (6), or local up states (8)
        """
        if self.up_or_down[0] == 1:
            if np.count_nonzero(self.up_or_down == 1) <= 2:  # the number of the populations in down states <=2
                return 2
            else:
                return 0
        else:
            if np.count_nonzero(self.up_or_down == 7) <= 2:  # the number of the populations in up states <=2
                return 8
            else:
                return 6

    def calculate_fr(self):
        """
        calculate the mean firing rates of excitatory population 1 in each state
        """
        self.time_start = 10000
        self.gso_down = np.where(self.state_plot[self.time_start : self.epoch_number*self.epoch_length] == 0, 1, 0)
        self.ldw_down = np.where(self.state_plot[self.time_start : self.epoch_number*self.epoch_length] == 2, 1, 0)
        self.gso_up = np.where(self.state_plot[self.time_start : self.epoch_number*self.epoch_length] == 6, 1, 0)
        self.ldw_up = np.where(self.state_plot[self.time_start : self.epoch_number*self.epoch_length] == 8, 1, 0)
        if np.sum(self.gso_down) != 0 and np.sum(self.ldw_down) != 0 and np.sum(self.gso_up) != 0 and np.sum(self.ldw_up) != 0: 
            self.fr_gso_down = 1000 * np.dot(self.gso_down, np.mean(self.y_e_plot[self.time_start : self.epoch_number*self.epoch_length, 0 : self.exc_popu], axis=1)) / np.sum(self.gso_down)
            self.fr_ldw_down = 1000 * np.dot(self.ldw_down, np.mean(self.y_e_plot[self.time_start : self.epoch_number*self.epoch_length, 0 : self.exc_popu], axis=1)) / np.sum(self.ldw_down)
            self.fr_gso_up = 1000 * np.dot(self.gso_up, np.mean(self.y_e_plot[self.time_start : self.epoch_number*self.epoch_length, 0 : self.exc_popu], axis=1)) / np.sum(self.gso_up)
            self.fr_ldw_up = 1000 * np.dot(self.ldw_up, np.mean(self.y_e_plot[self.time_start : self.epoch_number*self.epoch_length, 0 : self.exc_popu], axis=1)) / np.sum(self.ldw_up)
        else:
            self.fr_gso_down = -1
            self.fr_gso_up = -1
            self.fr_ldw_down = -1
            self.fr_ldw_up = -1

    def calculate_adapt(self, t, section, adapt_previous, y_e_now):
        """
        calculate adaptation currents
        """
        return adapt_previous*np.exp(-1./self.tau_adapt) + self.beta*y_e_now

    def next_g_average(self, t, section):
        """
        calculate the expected firing intensity g_average by calculating the population average of g(u)
        """
        return np.mean(self.params.g_e(self.u_e[0 : self.exc_popu]))

    def stdp_operation(self, t, section): 
        """
        the function used in STDP simulations
        """
        pass
    
    def calculate_next(self, t, section):
        """ 
        calculate the values of next timestep
        """
        self.u_e = self.params.u_r + (self.u_e - self.params.u_r) * (np.exp(-1./self.params.tau_m_e)) + self.w_rec_matrix_ee.T@self.y_e + self.w_rec_matrix_ei.T@self.y_i - self.adapt
        self.u_e[0] += np.dot(self.x[t], self.w_pre)  # the neuron 0 receives feedforward inputs
        self.u_i = self.params.u_r + (self.u_i - self.params.u_r) * (np.exp(-1./self.params.tau_m_i)) + self.w_rec_matrix_ie.T@self.y_e + self.w_rec_matrix_ii.T@self.y_i
        self.rho_e = self.params.g_e(self.u_e) * self.params.r(t, self.previous_spike_e)
        self.rho_i = self.params.g_i(self.u_i) * self.params.r(t, self.previous_spike_i)
        self.y_e = np.where(self.y_e_rand[t,:] < self.rho_e, 1, 0)
        self.y_i = np.where(self.y_i_rand[t,:] < self.rho_i, 1, 0)
        self.adapt = self.calculate_adapt(t, section, self.adapt, self.y_e)
        self.up_or_down = self.classify_up_down(t, self.up_or_down) 
        self.state = self.stageclassify(t) 
        
        if self.plasticity_option:
            self.h = self.params.calculate_trace(self.h, self.x[t], self.params.tau_m_e)
            self.g_average = self.next_g_average(t, section)
            self.bpost = self.params.bpost(self.rho_e[0], self.g_average * self.params.r(t, self.previous_spike_e[0]), self.y_e[0])
            self.small_c = self.params.small_cj(self.u_e[0], self.rho_e[0], self.y_e[0], self.h)
            self.large_c = self.params.calculate_trace(self.large_c, self.small_c, self.params.tau_c)
            self.w_pre = self.w_pre + self.learning_rate(t, section, self.state) * (self.large_c*self.bpost - self.params.lamb*self.w_pre*self.x[t])
        
        self.previous_spike_e = np.where(self.y_e == 1, t, self.previous_spike_e)
        self.previous_spike_i = np.where(self.y_i == 1, t, self.previous_spike_i)

    def set_plots(self): 
        """
        preparing variables for plotting
        """
        self.y_e_plot = np.zeros((self.epoch_length * self.epoch_number, self.exc_total))
        self.y_i_plot = np.zeros((self.epoch_length * self.epoch_number, self.inh_total))
        self.u_e_plot = self.params.u_r * np.ones((self.epoch_length * self.epoch_number, self.exc_total))
        self.u_i_plot = self.params.u_r * np.ones((self.epoch_length * self.epoch_number, self.inh_total))
        self.adapt_plot = np.zeros((self.epoch_length * self.epoch_number, self.exc_total))
        self.up_or_down_plot = -1 * np.ones(self.epoch_length * self.epoch_number) 
        self.state_plot = -1 * np.ones(self.epoch_length * self.epoch_number)
        if self.plasticity_option:
            self.g_average_plot = np.zeros((self.epoch_length * self.epoch_number))
            self.w_pre_plot = self.pre_w_initial * np.ones((self.epoch_length * self.epoch_number, self.pre_neuron))

    def simulate(self):
        """
        Conduct the simulation
        """
        if self.plot_option:
            self.set_plots()
        for section in range(self.epoch_number):
            self.pregenerator(section)
            self.postgenerator()
            for t in range(self.epoch_length):
                self.stdp_operation(t, section)
                self.calculate_next(t, section)
                if self.plot_option:
                    self.y_e_plot[section*self.epoch_length + t] = self.y_e
                    self.y_i_plot[section*self.epoch_length + t] = self.y_i
                    self.u_e_plot[section*self.epoch_length + t] = self.u_e
                    self.u_i_plot[section*self.epoch_length + t] = self.u_i
                    self.adapt_plot[section*self.epoch_length + t] = self.adapt
                    self.up_or_down_plot[section*self.epoch_length + t] = np.where(self.up_or_down[0] == 7, 8, 0)
                    self.state_plot[section*self.epoch_length + t] = self.state
                    if self.plasticity_option:
                        self.g_average_plot[section*self.epoch_length + t] = self.g_average
                        self.w_pre_plot[section*self.epoch_length + t] = self.w_pre

            if section < self.epoch_number-1:
                self.previous_spike_e -= self.epoch_length
                self.previous_spike_i -= self.epoch_length

class SlowwaveSTDPPreonly(Slowwave):
    """
    Used for simulating synaptic changes by the infomax rule in the pre-only stimulations in the slow wave model. 
    """

    def __init__(self, epoch_length, pre_w_initial, stim_state, stim_neuron):
        super().__init__(epoch_length=epoch_length, epoch_number=1, pre_w_initial=pre_w_initial, plasticity_option=True, plot_option=False, connect_prob_ee=0.05, connect_prob_ie=0.3)
        self.stim_state = stim_state
        self.stim_neuron = stim_neuron
        self.stim_num_total = 10  # the total number of STDP stimulations
        self.set_variables_stdp()

    def set_variables_stdp(self):
        self.stim_count = 0  # the number of STDP stimulations already given
        self.stim_interval = 500  # this variable counts the interval from the previous stimulation [msec]
        self.state_count = 0  # self.state_count counts the successive number of target states

    def stdp_stim(self, t):
        """
        STDP stimulations
        """
        self.x[t, 0 : self.stim_neuron] = 1  

    def stdp_operation(self, t, section):
        """
        Giving STDP stimulations whether the current situation meets the criteria. 
        """
        if self.state == self.stim_state:
            self.state_count += 1  # counting the number of successive number of target states
        else:
            self.state_count = 0

        if self.params.plasticity_start_t <= t < self.epoch_length - 80:
            self.stim_interval += 1
            if self.stim_interval >= 500 and self.state_count > 200 and self.stim_count < self.stim_num_total:  # the criteria is interval >= 500 msec, target state's duration > 200 msec, total stim <= 10 times
                self.stdp_stim(t)
                self.stim_count += 1
                self.stim_interval = 0

class SlowwaveSTDP(SlowwaveSTDPPreonly):
    """
    Used for simulating synaptic changes by the infomax rule in the STDP stimulations in the slow wave model. 
    """
    def __init__(self, epoch_length, pre_w_initial, delta_t, stim_state, stim_neuron):
        super().__init__(epoch_length, pre_w_initial, stim_state, stim_neuron)
        self.delta_t = delta_t

    def stdp_stim(self, t):
        """
        STDP stimulations
        """
        if self.delta_t < 0:
            self.x[t - self.delta_t, 0 : self.stim_neuron] = 1  # the presynaptic neurons emit a spike at time t - self.delta_t
            self.y_e_rand[t, 0] = -1.0  # the postsynaptic neuron 0 emits a spike at time t

        else:
            self.x[t, 0 : self.stim_neuron] = 1  # the presynaptic neurons emit a spike at time t
            self.y_e_rand[t + self.delta_t, 0] = -1.0  # the postsynaptic neuron 0 emits a spike at time t + self.delta_t

