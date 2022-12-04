import time
import numpy as np
import matplotlib.pyplot as plt
import common
import figplot
from scipy import integrate, special

t1 = time.time()

params = common.ConstantsAndFunctions()

def plot_g(x, y_1, y_2, y_3, y_4, aspect, fig, position):
    ax = fig.add_subplot(2, 1, position)
    ax.plot(x, y_4, color = 'red', label = 'Exponential')
    ax.plot(x, y_1, color = 'darkorange', label = 'Power')
    ax.plot(x, y_2, color = 'green', label = 'Softplus')
    ax.plot(x, y_3, color = 'navy', label = 'LIF')
    x_min = np.min(x) - 0.1
    x_max = np.max(x) + 0.1
    y_min = -1.0
    y_max = np.max(y_1) + 1.0
    ax.set_xlim(x_min, x_max)
    ax.set_ylim(y_min, y_max)
    ax.set_aspect(aspect * (x_max - x_min) / (y_max - y_min))
    ax.legend()

# power function    
def winfo_power(u, a):
    return 0.1 * (u- (params.u_r - 1.0)) ** a, (a / (u - (params.u_r - 1.0)))**2

# softplus
def winfo_softplus(u):
    return 1.5 * np.log(1.0 + np.exp((u - (-69.4)) / 0.5)), (1 / 0.5 * np.exp((u - (-69.4)) / 0.5) / (1.0 + np.exp((u - (-69.4)) / 0.5)) / np.log(1.0 + np.exp((u - (-69.4)) / 0.5)))**2  

# LIF
def lif_integrated(x):
    return np.exp(x ** 2) * (1 + special.erf(x))

def winfo_lif(h_0):
    thre = -64.0
    u_reset = -70.0
    sigma = 2.0
    integ = integrate.quad(lif_integrated, (u_reset - h_0) /sigma, (thre - h_0) /sigma)
    gu = 1 / (params.tau_m_e * np.sqrt(np.pi) * integ[0])
    dgdu = 1 / (params.tau_m_e * np.sqrt(np.pi)) * (-1.0 / (integ[0] ** 2)) * (-1 / sigma) * (lif_integrated((thre - h_0) /sigma) - lif_integrated((u_reset - h_0) /sigma))
    return 1000 * gu, (dgdu/gu) ** 2

# Exponential
def winfo_exp(u):
    return 0.05 * np.exp(u- (params.u_r - 1.0)), 1.0

u_list = np.array([-70 + 0.1 * i for i in range(50)])
g_list = np.zeros((4, np.size(u_list)))
winfo_list = np.zeros((4, np.size(u_list)))

g_list[0], winfo_list[0] = winfo_power(u_list, 3)
g_list[1], winfo_list[1] = winfo_softplus(u_list) 
g_list[3], winfo_list[3] = winfo_exp(u_list) 
for j in range(np.size(u_list)):
    g_list[2, j], winfo_list[2, j] = winfo_lif(u_list[j])

fig = plt.figure(figsize=(8, 8))
plot_g(u_list, g_list[0], g_list[1], g_list[2], g_list[3], 0.8, fig, 1)
plot_g(u_list, winfo_list[0], winfo_list[1], winfo_list[2], winfo_list[3], 0.8, fig, 2)
plt.savefig('figure/feedforward/gdependency.pdf')

t2 = time.time()
elapsed_time = t2-t1
print(f"Total time：{elapsed_time}")
