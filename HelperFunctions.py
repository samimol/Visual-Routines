# -*- coding: utf-8 -*-
"""
Created on Tue Dec 14 09:15:10 2021

@author: Sami
"""
import numpy as np
import pickle
import scipy.interpolate
from scipy.ndimage.filters import uniform_filter1d

def RunTrials(t,CurveLength,TrialNumber,n,device,verbose=True):
    n.exploit_prob = 1
    n.dosave = True
    corrects = []
    target_history = []
    distr_history = []
    feature_history = []
    position_history = []
    display = []
    t.curve_length = CurveLength
    action = np.zeros((1,49));
    n.saveX = [[]]
    n.saveXmod = [[]]
    n.saveY1 = [[]]
    n.saveY1mod = [[]]
    n.saveY2 = [[]]
    n.saveQ = [[]]
    
    n.saveX_disk = [[]]
    n.saveXmod_disk = [[]]
    n.saveY1_disk = [[]]
    n.saveY1mod_disk = [[]]
    n.saveY2_disk = [[]]
    n.saveQ_disk = [[]]
    
    for p in range(TrialNumber):
      trial_running = True
      display.append([])
      i=0
      new_input, reward, trialEnd= t.doStep(action)
      feature_history.append(t.feature_target) 
      position_history.append(t.position)
      while trial_running:
        action = n.doStep(new_input,reward,trialEnd,device)
        new_input, reward, trialEnd = t.doStep(action)
        display[p].append(n.X.detach())
        if trialEnd:
          trial_running = False
          if reward == 0:
            corrects.append(0)
          else:
            corrects.append(1)
        i = i + 1
      target_history.append(t.trialTarget)
      distr_history.append(t.trialDistr) 
    if verbose:
        print(np.mean(corrects))
    return(n,corrects,feature_history,target_history,distr_history,position_history,display)  


def open_base(filename):
    with open(filename+'.pkl', 'rb') as input:
      networks_base = pickle.load(input)
    return networks_base


def AverageTraces(n,CurveLength,TrialNumber,max_dur,target_history,distr_history,position_history,feature_history,corrects,Target,Distractor,CountTarget,CountDistractor):
    saved_states_list_disk = [n.saveXmod_disk,n.saveY1_disk,n.saveY1mod_disk,n.saveY2_disk] 
    saved_states_list = [n.saveXmod,n.saveY1,n.saveY1mod,n.saveY2] 
    for trial in range(1,TrialNumber):
            dur = len(n.saveXmod[trial]) 
            target_hist = target_history[trial-1]
            distr_hist = distr_history[trial-1]
            position = position_history[trial-1]
            feature = feature_history[trial-1]
            if feature == 0:
              dis = 1
            else:
              dis = 0
            if corrects[trial-1] == 1:
                      for f in range(n.n_hidden_features):
                        for layer in range(4):
                          for timestep in range(dur):
                              for l in range(CurveLength):
                                Target[target_hist[l]+2,layer,f,l,timestep]+=saved_states_list[layer][trial][timestep][0,f,target_hist[l] % n.grid_size,target_hist[l] // n.grid_size]
                                CountTarget[target_hist[l]+2,layer,f,l,timestep] += 1
    
                                Distractor[distr_hist[l]+2,layer,f,l,timestep]+=saved_states_list[layer][trial][timestep][0,f,distr_hist[l] % n.grid_size,distr_hist[l] // n.grid_size]
                                CountDistractor[distr_hist[l]+2,layer,f,l,timestep] += 1
                              
                              Target[position[feature],layer,f,CurveLength,timestep]+=saved_states_list_disk[layer][trial][timestep][0,f,position[feature]]
                              CountTarget[position[feature],layer,f,CurveLength,timestep] += 1
    
                              Distractor[position[dis],layer,f,CurveLength,timestep]+=saved_states_list_disk[layer][trial][timestep][0,f,position[dis]]
                              CountDistractor[position[dis],layer,f,CurveLength,timestep] += 1
                          if timestep < max_dur:
                                for l in range(CurveLength):
                                  Target[target_hist[l]+2,layer,f,l,dur:max_dur]+=np.array([saved_states_list[layer][trial][dur-1][0,f,target_hist[l] % n.grid_size,target_hist[l] // n.grid_size] for i in range(max_dur - dur)])
                                  CountTarget[target_hist[l]+2,layer,f,l,dur:max_dur] += np.array([1  for i in range(max_dur - dur)])
      
                                  Distractor[distr_hist[l]+2,layer,f,l,dur:max_dur]+=np.array([saved_states_list[layer][trial][dur-1][0,f,distr_hist[l] % n.grid_size,distr_hist[l] // n.grid_size]  for i in range(max_dur - dur)])
                                  CountDistractor[distr_hist[l]+2,layer,f,l,dur:max_dur] += np.array([1 for i in range(max_dur - dur)])
                                
                                Target[position[feature],layer,f,CurveLength,dur:max_dur]+=np.array([saved_states_list_disk[layer][trial][dur-1][0,f,position[feature]]  for i in range(max_dur - dur)])
                                CountTarget[position[feature],layer,f,CurveLength,dur:max_dur] += np.array([1  for i in range(max_dur - dur)])
      
                                Distractor[position[dis],layer,f,CurveLength,dur:max_dur]+=np.array([saved_states_list_disk[layer][trial][dur-1][0,f,position[dis]]  for i in range(max_dur - dur)])
                                CountDistractor[position[dis],layer,f,CurveLength,dur:max_dur] += np.array([1  for i in range(max_dur - dur)])

    return Target,Distractor,CountTarget,CountDistractor



def latency_operation(TD,operation,TASK,CurveLength,number_interpolation_points=500,dur=39,criterion=0.3):
    
    if operation == 'search':
        neurons = [0,1] #first two neurons correspond to the search operation
        position_on_curve = -1 #we arbitrarily put them at the end of the curve_length dimension in TD
    elif operation == 'trace':
        neurons = np.arange(2,TD.shape[0])
    else:
        raise Exception("Operation is either search or trace")
        
    if operation == 'trace' and TASK == 'searchtrace':
        position_on_curve = 0 #in the search then trace task, the trace follows the search
    elif operation == 'trace' and TASK == 'tracesearch':
        position_on_curve = CurveLength - 1 #in the trace then searcg task, the trace precedes the search
    
    #Interpolating the response curve to have non-integer latency of modulation   
    normalized_TD = TD[neurons,:,:,position_on_curve,:]/np.expand_dims(np.max(np.abs(TD[neurons,:,:,position_on_curve,:]),axis = -1),axis=-1)
    interpollation_function = scipy.interpolate.interp1d(np.arange(0,dur), normalized_TD, kind='linear',axis=-1)
    interpolated_normalized_TD = interpollation_function(np.linspace(0, dur-1, num=number_interpolation_points))
    smoothed_interpolated_normalized_TD = uniform_filter1d(interpolated_normalized_TD, size=100,axis = -1)
    
    
    #Modulation condition: criterion * difference between beginning and end
    condition = np.abs(np.max(smoothed_interpolated_normalized_TD,axis=-1)-np.min(smoothed_interpolated_normalized_TD,axis=-1)) * criterion + np.min(smoothed_interpolated_normalized_TD,axis=-1)
    condition = np.expand_dims(condition,axis=-1)
    
    #Getting the timestep when the modulation is greater than criterion%
    latency = np.nanargmin(smoothed_interpolated_normalized_TD[:,:,:,::-1]>condition,axis=-1)
    latency = latency.astype('float')
    latency = 499 - latency
    
    #Removing recording site with non-positive mor non monotonous modulation
    latency[np.isnan(smoothed_interpolated_normalized_TD[:,:,:,0])] = np.nan
    
    latency[smoothed_interpolated_normalized_TD[:,:,:,-1] < 0.5] = np.nan
    latency[smoothed_interpolated_normalized_TD[:,:,:,0]>=0.5] = np.nan
    latency[np.abs(np.max(smoothed_interpolated_normalized_TD,axis=-1)-np.min(smoothed_interpolated_normalized_TD,axis=-1)) < 0.05] = np.nan #constant modulation
    latency[np.max(smoothed_interpolated_normalized_TD,axis=-1)<=0] = np.nan #modulation is negative
    latency[smoothed_interpolated_normalized_TD[:,:,:,-1] < smoothed_interpolated_normalized_TD[:,:,:,0]] = np.nan #modulation at the beginning is greater than at the end


    #Keeping non nan entries
    latency = latency[~np.isnan(latency)]
    latency = latency.astype('int')
    latency = np.linspace(0, dur-1, num=number_interpolation_points)[latency] 
    
    return latency


    