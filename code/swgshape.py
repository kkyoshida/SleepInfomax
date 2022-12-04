import time
import numpy as np
import matplotlib.pyplot as plt
import slowwavemodel
import figplot
import common

t1 = time.time() 

#Plot the relationship between the firing probability and the membrane potential in excitatory and inhibitory neurons
params = common.ConstantsAndFunctions()
x = np.arange(-75,-58.9,0.1) 
y_e = 1000 * params.g_e(x)
y_i = 1000 * params.g_i(x)
fig = plt.figure(figsize=(4,4))
figplot.plot_function(x = x, y_1 = y_e, y_2 = y_i, aspect = 1.0, fig = fig)
plt.savefig('figure/slowwave/gshape.pdf')

t2 = time.time()
elapsed_time = t2-t1
print(f"Total time：{elapsed_time}")
