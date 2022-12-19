# SleepInfomax

## Overview
These codes are accompanying the manuscript: 

Kensuke Yoshida and Taro Toyoizumi, "Information maximization explains state-dependent synaptic plasticity and memory reorganization during non-rapid eye movement sleep", PNAS Nexus, 2022. https://doi.org/10.1093/pnasnexus/pgac286 

## Requirements
Simulations for the paper above were conducted in the following setup: 

iMac Pro (2017)

CPU: 2.3 GHz 18 core Intel Xeon W

RAM: 128 GB

python 3.6.13, numpy 1.19.2, scipy 1.5.2, matplotlib 3.3.2                


## Usage
For obtaining all figures in the paper, run run_code_all.py. Note that it takes long time to run all codes. 

The basic descriptions of single-neuron and slow-wave models are in feedforwardmodel.py and slowwavemodel.py, respectively. 

More detailed correspondance between codes and figures in the paper is as follows: 

### Single-neuron model
First, you need to run ffcalculateg.py. 

Figure 1: 
ffrepresentativetrace.py

Figure 2, S2-S4: 
ffaveragetrace.py, ffstdp.py, ffstdptwod.py, ffaveragetracemanipulated.py

Figure S1: 
gdependency.py

### Slow-wave model
Figure 3, S5, S6: 
swwaveform.py, swgshape.py, swphaseplane.py, swfiringrate.py, swudist.py, swupdownthreshold.py

Figure 4: 
swstdp.py, swstdp_smallw.py, swstdpstat.py, swstdpstatplot.py 

Figure 5, S7: 
swtaskup.py, swtaskup2.py, swtaskup3.py, swtaskdown.py, swtaskwaveformupdown.py, swtaskplotweight.py, swtaskplotreactivation.py, swtaskplotdown.py

## License
This project is licensed under the MIT License (see LICENSE.txt for details).
