import subprocess
import os

outputpath = r'figure/feedforward' 
if not os.path.exists(outputpath):
    os.makedirs(outputpath)

outputpath = r'figure/slowwave' 
if not os.path.exists(outputpath):
    os.makedirs(outputpath)

outputpath = r'data' 
if not os.path.exists(outputpath):
    os.makedirs(outputpath)

subprocess.run('python code/ffcalculateg.py', shell=True)
subprocess.run('python code/ffrepresentativetrace.py', shell=True)
subprocess.run('python code/ffaveragetrace.py', shell=True)
subprocess.run('python code/ffstdp.py', shell=True)
subprocess.run('python code/ffstdptwod.py', shell=True)
subprocess.run('python code/ffaveragetracemanipulated.py', shell=True)
subprocess.run('python code/gdependency.py', shell=True)

subprocess.run('python code/swwaveform.py', shell=True)
subprocess.run('python code/swgshape.py', shell=True)
subprocess.run('python code/swphaseplane.py', shell=True)
subprocess.run('python code/swfiringrate.py', shell=True)
subprocess.run('python code/swudist.py', shell=True)
subprocess.run('python code/swupdownthreshold.py', shell=True)

subprocess.run('python code/swstdp.py', shell=True)
subprocess.run('python code/swstdp_smallw.py', shell=True)
subprocess.run('python code/swstdpstat.py', shell=True)
subprocess.run('python code/swstdpstatplot.py', shell=True)

subprocess.run('python code/swtaskup.py', shell=True)
subprocess.run('python code/swtaskup2.py', shell=True)
subprocess.run('python code/swtaskup3.py', shell=True)
subprocess.run('python code/swtaskdown.py', shell=True)
subprocess.run('python code/swtaskwaveformupdown.py', shell=True)
subprocess.run('python code/swtaskplotweight.py', shell=True)
subprocess.run('python code/swtaskplotreactivation.py', shell=True)
subprocess.run('python code/swtaskplotdown.py', shell=True)

