import time
import numpy as np
import matplotlib.pyplot as plt
import feedforwardmodel
import figplot

t1 = time.time() 

np.random.seed(2)
# Calculate the g_average value for each firing rate of non-stimulated neuron
pre_fr_list = feedforwardmodel.pre_fr_list()  
g_average_list = np.zeros((np.size(pre_fr_list), 2))

for i in range(np.size(pre_fr_list)):
    g_average_list[i,0] = pre_fr_list[i]  # firing rate of non-stimulated neuron
    g_average_list[i,1] = feedforwardmodel.set_g_average(pre_fr_list[i])  # calculate the g_average value

np.save('data/glist',g_average_list)
print(1000 * g_average_list[feedforwardmodel.up_down_parameters('down')])
print(1000 * g_average_list[feedforwardmodel.up_down_parameters('up')])

t2 = time.time()
elapsed_time = t2-t1
print(f"Total time：{elapsed_time}")

