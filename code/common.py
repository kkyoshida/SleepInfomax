import numpy as np

class ConstantsAndFunctions:
    """
    Constant values and functions needed for the models. 
    """
    def __init__(self):
        self.u_r = -70.0  # resting membrane potential [mV]
        self.tau_m_e = 25.0  # membrane time constant of excitatory neurons [msec]
        self.tau_m_i = 5.0  # membrane time constant of inhibitory neurons [msec]
        self.tau_c = 100.0  # time constant of calculating C_j from c_j [msec]
        self.plasticity_start_t = 10000  # time when synaptic weights start to change in each simulation [msec]
        self.lamb = 0.32  # constant value determining synaptic change
        self.alpha = 0.01  # learning rate

    def g_e(self, u_now): 
        """
        the relationship between the firing probability and the membrane potential of excitatory neurons [1/msec]
        """
        return 0.001 * 1.5 * np.log(1.0 + np.exp((u_now - (-69.4)) / 0.5))

    def g_i(self, u_now):
        """
        the relationship between the firing probability and the membrane potential of inhibitory neurons [1/msec]
        """
        return 0.001 * 6.0 * np.log(1.0 + np.exp((u_now - (-62.5)) / 0.5))
        
    def r(self, t_now, t_previousspike): 
        """
        refractory factor
        t_now: scalar, t_previousspike:ndarray
        """
        return (t_now-t_previousspike)**4 / (30.0**4 + (t_now-t_previousspike)**4)

    def small_cj(self, u_now, rho_now, y_now, h_j_now):
        """
        auxiliary function in the infomax rule
        """
        return 0.001 * 3.0 * np.exp((u_now - (-69.4)) / 0.5) / (1 + np.exp((u_now - (-69.4)) / 0.5)) * (y_now - rho_now) / self.g_e(u_now) * h_j_now 
    
    def bpost(self, rho_now, rho_average_now, y_now):
        """
        auxiliary function in the infomax rule
        """
        return (y_now * np.log(rho_now / rho_average_now) - (rho_now - rho_average_now)) 

    def calculate_trace(self, x_convolve, x_now, tau_x):
        """
        the function for calculating convolution 
        """
        return x_now + x_convolve*np.exp(-1./tau_x)

    def synaptic_change(self, learning_rate, large_c, bpost, w_now, x_now):
        """
        the synaptic change following the infomax rule
        """
        return learning_rate * (large_c * bpost - self.lamb * w_now * x_now)

    def synaptic_change_info(self, learning_rate, large_c, bpost, w_now, x_now):
        """
        the synaptic change derived from the mutual information term
        """
        return learning_rate * (large_c * bpost)

    def synaptic_change_cost(self, learning_rate, large_c, bpost, w_now, x_now):
        """
        the synaptic change derived from the synaptic weight cost term
        """
        return learning_rate * (- self.lamb * w_now * x_now)
