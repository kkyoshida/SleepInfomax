import time
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import swtaskreactivation
import figplot

def statecolor(i):
    if i == 0:
        return 'white'
    elif i == 2:
        return 'white'
    elif i == 6:
        return 'red'
    else:
        return 'purple'
    
def statecolor_down(i):
    if i == 0:
        return 'green'
    elif i == 2:
        return 'saddlebrown'
    elif i == 6:
        return 'red'
    else:
        return 'purple'

t1 = time.time() 

epolen = 50000
eponum = 2
epowake = 0

np.random.seed(5)

w_other = 0.0
sw = swtaskreactivation.SlowwaveReactivationDecrease(epoch_length=epolen, epoch_number=eponum, pre_w_initial=w_other, reac_state='allup', reac_neuron=40, reac_rate=0.005, wake_epoch_number=epowake, wake_rate=0.005, w_pre_list_initial=w_other*np.ones(100), plasticity_option=False)
sw.simulate()

fig = plt.figure(figsize=(8, 8))

t_start = 50000
t_end = t_start + 20000
state_array= sw.state_plot[t_start : t_end]
u_array = np.mean(sw.u_e_plot[t_start : t_end, 0 : sw.exc_popu], axis=1)

color_array = []
color_array_down = []
for i in range(t_end - t_start):
    color_array.append(statecolor(state_array[i]))
    color_array_down.append(statecolor_down(state_array[i]))

for protocol in range(2):
    ax = fig.add_subplot(2, 1, 1 + protocol)
    ax.plot(u_array)
    ax.set_aspect(0.5 * 10000 / 10)

    for i in range(t_end - t_start):
        if protocol == 0:
            r1 = patches.Rectangle(xy=(1.0*i, -74.5), width=1.0, height=3.0, angle=0, color=color_array[i], linewidth=0)
        else:
            r1 = patches.Rectangle(xy=(1.0*i, -74.5), width=1.0, height=3.0, angle=0, color=color_array_down[i], linewidth=0)
        ax.add_patch(r1)
    ax.set_xlim(0, t_end - t_start)
    ax.set_ylim(-74.5,-61)
    ax.axis("off")
    
plt.savefig('figure/slowwave/task_sleep_color.pdf')

t2 = time.time()
elapsed_time = t2-t1
print(f"Total time：{elapsed_time}")
