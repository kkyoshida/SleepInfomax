import time
import numpy as np
import matplotlib.pyplot as plt
import slowwavemodel
import figplot
import common
from matplotlib.colors import Normalize
from scipy import optimize
from functools import partial

t1 = time.time() 

sw = slowwavemodel.Slowwave(epoch_length=50000, epoch_number=1, pre_w_initial=0.0, plasticity_option=False, plot_option=False)
adapt_list = np.array([0, -0.01, -0.05, -0.18])

def diff(u_e, u_i, adapt, input_exc, input_inh):
    """
    Calculate the differentiation of excitatory and inhibitory membrane potentials
    """
    duedt = 1/sw.params.tau_m_e*(-(u_e-sw.params.u_r)) + sw.w_rec_ee*(sw.exc_popu-1)*sw.params.g_e(u_e) + sw.w_rec_ei*(sw.inh_popu)*sw.params.g_i(u_i) + adapt + input_exc
    duidt = 1/sw.params.tau_m_i*(-(u_i-sw.params.u_r)) + sw.w_rec_ie*(sw.exc_popu)*sw.params.g_e(u_e) + input_inh
    return duedt, duidt

def set_xyrange():
    x_min, x_max, y_min, y_max = -75, -61, -71, -57
    return x_min, x_max, y_min, y_max

def calculate_flow(delta, adapt , input_exc, input_inh):
    """
    Calculate the differentiation of excitatory and inhibitory membrane potentials in each point
    """
    x_min, x_max, y_min, y_max = set_xyrange()
    x_range = np.arange(x_min, x_max, delta)
    y_range = np.arange(y_min, y_max, delta)
    x, y = np.meshgrid(x_range, y_range)
    dx, dy = diff(x, y, adapt , input_exc, input_inh)
    return x, y, dx, dy

def calculate_intersect(x, adapt, input_exc, input_inh):
    """
    functions used for calculating the intersections of the nullclines
    """
    return list(diff(x[0], x[1], adapt, input_exc, input_inh))

def plot_phaseplane(filename, other_fr):
    """
    Plot phase plane
    """
    fig = plt.figure(figsize=(18,3))
    for i in range(4):
        population = 4
        
        # external currents from other excitatory populations
        input_exc = (population-1)*sw.connect_prob_ee*sw.exc_popu*sw.w_rec_ee*0.001*other_fr
        input_inh = (population-1)*sw.connect_prob_ie*sw.exc_popu*sw.w_rec_ie*0.001*other_fr
        
        adapt = adapt_list[i]

        # plot the nullclines
        x_dense, y_dense, dx_dense, dy_dense = calculate_flow(0.01, adapt, input_exc, input_inh)
        ax = fig.add_subplot(1,4,1+i)
        ax.axis(set_xyrange())
        ax.set_aspect('equal', adjustable='box')
        ax.contour(x_dense, y_dense, dx_dense, [0], colors='darkorange')  # excitatory nullcline
        ax.contour(x_dense, y_dense, dy_dense, [0], colors='navy')  # inhibitory nullcline
        
        # plot the intersections of the nullclines
        for j in range(2):
            intersect = optimize.root(partial(calculate_intersect, adapt=adapt, input_exc=input_exc,                                                    input_inh=input_inh), [[-70.0, -70.0],[-60.0, -60.0]][j])
            ax.plot(intersect.x[0], intersect.x[1], marker='s', markersize=10, fillstyle='none', markeredgewidth=2)
        
        # plot vector field
        x_sparse, y_sparse, dx_sparse, dy_sparse = calculate_flow(1.5, adapt, input_exc, input_inh)
        flow_norm = np.hypot(dx_sparse, dy_sparse)   
        dx_sparse /= flow_norm  # normalize each arrow                                    
        dy_sparse /= flow_norm  # normalize each arrow
        vectorfield = ax.quiver(x_sparse, y_sparse, dx_sparse, dy_sparse, flow_norm,                       norm = Normalize(vmin=0.0, vmax=4.1), cmap='viridis', alpha=0.6)
        fig.colorbar(vectorfield, ax=ax)
    plt.savefig(filename)
    
plot_phaseplane("figure/slowwave/phaseplane_otherdown.pdf", 0.0)
plot_phaseplane("figure/slowwave/phaseplane_otherup.pdf", 6.0)


t2 = time.time()
elapsed_time = t2-t1
print(f"Total time：{elapsed_time}")

