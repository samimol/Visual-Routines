# -*- coding: utf-8 -*-
"""
Created on Wed Nov 24 17:11:08 2021

@author: Sami
"""

import numpy as np
import matplotlib.pyplot as plt
from IPython import display
np.seterr(all='ignore')
import warnings
warnings.filterwarnings("ignore")
import gc
import time
from Layers import *
from Network import *
from Task import *

import pickle
with open('TraceTask.pkl', 'rb') as input:
    TraceTask = pickle.load(input)

grid_size = 7

n = Network(4)
n.beta = 0.002
n.recordTraces = False

t=Task(4,'trace')
t.onestep = True
t.onlyTrace = True
t.max_dur = 0
t.CheckDistance = 'no'
t.onlyblue = True
t.delay = 1

reward = 0
trialEnd = False
trials = 150000
tac = 0
average = 0
trial_corrects = []
average_all = []
Converged_for_Five = False
Generalization = []
action = np.zeros((1,49));
action[0] = 1;
device = None
total_length = 7
dataset = TraceTask['onlyblue']
#tt = time.time()
for i in range(trials):

    trial_running = True
    new_input, reward, trialEnd= t.doStep(action,dataset)

    while trial_running:
      action = n.doStep(new_input,reward,trialEnd,device)
      new_input, reward, trialEnd = t.doStep(action,dataset)
      n.doLearn(reward)
      if trialEnd:
            #n.doErase()
            trial_running = False
            t.feature_target = np.random.randint(2)
            curve_length = np.random.randint(max_length-min_length+1) + min_length
            dataset = TraceTask[str(curve_length)]
            if reward != 0:
                trial_corrects.append(1)
            else:
                trial_corrects.append(0)
    if (i-tac)%2000 == 0 and i > 0:
        (n,average) = CheckNetwork(n,max_length,t.onlyblue,device)
        average_all.append(average)
        print(average)
    if (i-tac) > 1000 and average >= 0.85 and t.onlyblue:
        t.onlyblue = False
        tac = i
        max_length = 3
        min_length = 3
        #print('max length now 3')
    if not t.onlyblue:
      if max_length < total_length:
        if (i-tac) > 2000 and average >= 0.85:
          max_length += 1
          #(n,gen) = CheckNetwork(n,max_length,t.onlyblue,device)
          #Generalization.append(gen)
          tac = i
      else:
          #(n,gen) = CheckNetwork(n,max_length+1,t.onlyblue,device)
          #Generalization.append(gen)
          break      
    if i==0:
        fig,ax = plt.subplots(1,1)
        hdisplay = display.display("", display_id=True)
    if i>499 and i%500 == 0:
          ax.clear()
          ax.plot(np.convolve(trial_corrects, np.ones(1000), 'valid') / 1000)
          hdisplay.update(fig)
#print(time.time() - tt)