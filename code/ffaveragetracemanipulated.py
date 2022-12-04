import time
import numpy as np
import matplotlib.pyplot as plt
import feedforwardmodel
import figplot

t1 = time.time() 

np.random.seed(5)
g_average_list=np.load('data/glist.npy')

def plot_timeseries_points_arrow(g, g_min, g_max, t_min, t_max, aspect, fig, plot_x, plot_y, plot_area): 
    time_trace=np.arange(t_min, t_max)
    ax = fig.add_subplot(plot_x, plot_y, plot_area)
    ax.plot(time_trace, g, marker='o', markersize=3, color='tab:blue')
    ax.set_aspect(aspect * (t_max - t_min) / (g_max - g_min))
    ax.set_xlim(t_min, t_max)
    ax.set_ylim(g_min, g_max)
    ax.arrow(25, 0.03, 0, -0.011, width=0.2, head_width=4.0, head_length=0.004, color='black')

class FeedforwardSTDPmanipulate(feedforwardmodel.FeedforwardSTDPPreonly):
    """
    Simulating pre-only stimulations when the postsynaptic membrane potential is manipulated in the feedforward model. 
    """
    def changeonetime(self, t):
        self.h = self.params.calculate_trace(self.h, self.x[t], self.params.tau_m_e)
        self.h_other = self.params.calculate_trace(self.h_other, self.x_other[t], self.params.tau_m_e)
        self.u = self.params.u_r + np.dot(self.h, self.w) + self.h_other * self.w_other  # membrane potential changes according to inputs
        if t >= 10075:
            self.u += (0.001 * 2.1 - 2.0 * self.pre_fr) * self.other_pre_neuron * self.w_other * self.params.tau_m_e
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

def calculate_single_trace(g_index, w_initial, pre_neuron, mode):
    """
    Calculating the time series of synaptic weights and postsynaptic spikes 
    in two modes: post-pre stimulations or pre-only stimulations
    """
    cal_stdp = FeedforwardSTDPmanipulate(*feedforwardmodel.stdp_variables(g_index, g_average_list, w_initial, pre_neuron))
    cal_stdp.change(plot_option=True)
    return np.mean(cal_stdp.w_plot[10050:10200,:], axis=1), cal_stdp.y_plot[10050:10200]

def calculate_average_trace(state, w_initial, pre_neuron, mode):
    """
    Calculating the average time series of synaptic weights and postsynaptic spikes 
    in two modes: post-pre stimulations or pre-only stimulations
    """
    g_index = feedforwardmodel.up_down_parameters(state=state)
    trial_number = 20000
    result_w = np.zeros((150, trial_number))
    result_y = np.zeros((150, trial_number))
    for trial in range(trial_number):
        result_w[:,trial], result_y[:,trial] = calculate_single_trace(g_index=g_index, w_initial=w_initial, pre_neuron=pre_neuron, mode=mode)
    return result_w, result_y

fig = plt.figure(figsize=(16,10))
for i in range(2):
    w_trace, y_trace = calculate_average_trace(state=['up', 'down'][i], w_initial=0.5, pre_neuron=20, mode='preonly')
    w_average = np.mean(w_trace, axis=1)
    w_sd = np.std(w_trace, axis=1)
    y_average = np.mean(y_trace, axis=1)
    figplot.plot_timeseries_sd(g=w_average, g_sd=w_sd, g_min=0.995, g_max=1.004, t_min=0, t_max=150, aspect=0.5, fig=fig, plot_x=2, plot_y=4, plot_area=1+4*i)
    plot_timeseries_points_arrow(g=y_average, g_min = -0.0025, g_max = 0.05, t_min=0, t_max=150, aspect=0.5, fig=fig, plot_x=2, plot_y=4, plot_area=2+4*i)

plt.savefig('figure/feedforward/averagetrace-manipulated.pdf')


t2 = time.time()
elapsed_time = t2-t1
print(f"Total time：{elapsed_time}")
