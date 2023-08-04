# -*- coding: utf-8 -*-
"""
Created on Tue Dec 14 09:15:10 2021

@author: Sami
"""
import numpy as np
import pickle
import scipy.interpolate
from scipy.ndimage.filters import uniform_filter1d
from Task import *
from Network import *
import matplotlib.pyplot as plt
from IPython import display
np.seterr(all='ignore')
import warnings
warnings.filterwarnings("ignore")


def train(grid_size=15,task='trace',max_trials=50000,max_length=9,verbose=False):
    
    device = None
    n = Network(4,grid_size)

    current_max_length = 2
    min_length = 2
    
    if task == 'trace':
        t=Trace(4,grid_size)
    elif task == 'search_trace':
        t=SearchTrace(4,grid_size)
    elif task == 'trace_search':
        t=TraceSearch(4,grid_size)
        t.only_trace_curve = True
    else:
        raise Exception("TASK should be 'trace', 'search_trace' or 'trace_search'")

    average = 0
    performance_track = []
    average_all = []
    generalization = np.zeros((max_length-2,max_length+1))
    
    action = 0
    tracker = 0

    for i in range(max_trials):
        
        # The learning loop 
        trial_running = True
        new_input, reward, end_of_trial= t.do_step(action)
        while trial_running:
          action = n.do_step(new_input,reward,end_of_trial,device)
          new_input, reward, end_of_trial = t.do_step(action)
          n.do_learn(reward)
          if end_of_trial:
                trial_running = False
                t.curve_length = np.random.randint(current_max_length-min_length+1) + min_length
                if reward != 0:
                    performance_track.append(1)
                else:
                    performance_track.append(0)
                    
        # Every 2000 trials, we evaluate performance without learning or exploration            
        if (i-tracker)%2000 == 0 and i > 0:
            (n,average) = test(n,task,grid_size,current_max_length,t.no_curves,t.only_trace_curve,device)
            average_all.append(average)
            if verbose:
                print(average)
                
        # If the model learns to select blue pixels, we start presenting curves
        if (i-tracker) > 2000 and average >= 0.85 and t.no_curves:
            t.no_curves = False
            tracker = i
            current_max_length = 3
            min_length = 3
            if task == 'search_trace':
                n.beta = 0.09

        # Each time the network achieve good accuracy for a given length, we
        # test generalization accuracy and then add 1 pixel to the curves
        if not t.no_curves:
          if current_max_length < max_length:
            if (i-tracker) > 2000 and average >= 0.85:
                if t.only_trace_curve == False:
                  for k in range(current_max_length+1,current_max_length+5):
                        (n,gen) = test(n,task,grid_size,k,t.no_curves,t.only_trace_curve,device)
                        generalization[current_max_length-3,k-4] = gen
                current_max_length += 1
                tracker = i
                if task == 'search_trace':
                    n.beta = 0.005  
                    
          # In the trace then search task, when the network has learned to trace
          # we add the search part of the task
          elif current_max_length == max_length and (i-tracker) > 2000 and average >= 0.85 and t.only_trace_curve == True:
              t.only_trace_curve = False
              t.no_curves = True
              current_max_length = 2
              min_length = 2
              tracker = i
             
          # Enf of training
          elif current_max_length == max_length and (i-tracker) > 2000 and average >= 0.85 and t.only_trace_curve == False:
            for k in range(current_max_length+1,current_max_length+5):
                (n,gen) = test(n,task,grid_size,k,t.no_curves,t.only_trace_curve,device)
                generalization[current_max_length-3,k-4] = gen
            break
                  
        if verbose:
            # Plotting the accuracy every 500 trials
            if i==0:
                fig,ax = plt.subplots(1,1)
                hdisplay = display.display("", display_id=True)
            if i>499 and i%500 == 0:
                  ax.clear()
                  ax.plot(np.convolve(performance_track, np.ones(1000), 'valid') / 1000)
                  hdisplay.update(fig)
                  
    if i < max_trials:
        n.converged = True
    else:
        n.converged = False
    return(n,performance_track,generalization)


def test(n,task,grid_size,current_max_length,no_curves,only_trace_curve,device):

    prev_exploitation_probability = n.exploitation_probability
    n.exploitation_probability = 1
    
    if task == 'trace':
        t=Trace(4,grid_size)
    elif task == 'search_trace':
        t=SearchTrace(4,grid_size)
    elif task == 'trace_search':
        t=TraceSearch(4,grid_size)
        t.only_trace_curve = only_trace_curve
    t.curve_length = current_max_length
    t.no_curves = no_curves
    
    performance = []
    action = 0
   
    for p in range(500):
      trial_running = True
      new_input, reward, end_of_trial= t.do_step(action)
      while trial_running:
          action = n.do_step(new_input,reward,end_of_trial,device)
          new_input, reward, end_of_trial = t.do_step(action)
          if end_of_trial:
            trial_running = False
            t.feature_target = np.random.randint(2)
            if reward != 0:
                performance.append(1)
            else:
                performance.append(0)
    n.exploitation_probability = prev_exploitation_probability
    return(n,np.mean(performance))


def open_base(filename):
    input = open(filename+'.pkl', 'rb')
    networks_base = pickle.load(input)
    input.close()
    return networks_base

def run_and_save(n,curve_length=9,max_trials=15000,task='trace',grid_size=15):

    if task == 'trace':
        t=Trace(4,grid_size)
    elif task == 'search_trace':
        t=SearchTrace(4,grid_size)
    elif task == 'trace_search':
        t=TraceSearch(4,grid_size)

    t.no_curves = False
    t.only_trace_curve = False
    max_dur = 50


    if n.converged:
        target_activations = np.zeros((n.grid_size ** 2+2,5,n.n_hidden_features,curve_length+1,max_dur))  #pixels, hidden layers, features, position on the curve, timestep
        distractor_activations = np.zeros((n.grid_size ** 2+2,5,n.n_hidden_features,curve_length+1,max_dur))
        count_target = np.zeros((n.grid_size ** 2+2,5,n.n_hidden_features,curve_length+1,max_dur))
        count_distractor = np.zeros((n.grid_size ** 2+2,5,n.n_hidden_features,curve_length+1,max_dur))

        n,corrects,colour_disk_history,target_history,distr_history,position_disk_history,display,target_curve_history,distractor_curve_history,count_target,count_distractor = run_and_average(t,curve_length,max_trials,n,device,target_activations,distractor_activations,count_target,count_distractor)
       
        #averaging for each unit
        target_activations = target_activations/count_target
        distractor_activations = distractor_activations/count_distractor

        return(target_activations,distractor_activations,np.mean(corrects),colour_disk_history,target_curve_history,distractor_curve_history,max_trials,curve_length)


def run_and_average(t,curve_length,number_of_trials,n,device,target,distractor,count_target,count_distractor,verbose=True):
    n.exploitation_probability = 1
    n.save_activities = True
    corrects = []
    target_history = []
    distr_history = []
    feature_history = []
    position_history = []
    display = []
    t.curve_length = curve_length
    action = 0
    max_dur = 50    
    for p in range(number_of_trials):
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
      display.append([])
      
      trial_running = True
      i=0
      new_input, reward, end_of_trial= t.do_step(action)
      feature_history.append(t.feature_target) 
      position_history.append(t.position_markers)
      while trial_running:
        action = n.do_step(new_input,reward,end_of_trial,device)
        new_input, reward, end_of_trial = t.do_step(action)
        display[p].append(n.X.detach())
        if end_of_trial:
          trial_running = False
          if reward == 0:
            corrects.append(0)
          else:
            corrects.append(1)
        i = i + 1
    
      if corrects[-1] == 1:
          feature_history.append(t.feature_target) 
          position_history.append(t.position_markers)            
          target_history.append(t.target_curve)
          distr_history.append(t.distractor_curve)
          target,count_target,distractor,count_distractor = running_average(n,t,curve_length,max_dur,target,count_target,distractor,count_distractor)
    if verbose:
        print(np.mean(corrects))
    return(n,corrects,feature_history,target_history,distr_history,position_history,display,target,distractor,count_target,count_distractor)  

def running_average(n,t,curve_length,max_dur,target,count_target,distractor,count_distractor):
    dur = len(n.saveXmod[1])
    if t.feature_target == 0:
        dis = 1
    else:
        dis = 0   
    saved_states_list_disk = [n.saveXmod_disk[1],n.saveY1mod_disk[1],n.saveY2_disk[1],n.saveY1_disk[1],n.saveQ_disk[1]] 
    saved_states_list = [n.saveXmod[1],n.saveY1mod[1],n.saveY2[1],n.saveY1[1],n.saveQ[1]] 
    for f in range(n.n_hidden_features):
      for layer in range(5):
        if layer == 4:
          feature = 0
        else:
          feature = f
        for timestep in range(dur):
            for l in range(curve_length):
              target[t.target_curve[l]+2,layer,feature,l,timestep]+=saved_states_list[layer][timestep][0,feature,t.target_curve[l] % n.grid_size,t.target_curve[l] // n.grid_size]
              count_target[t.target_curve[l]+2,layer,feature,l,timestep] += 1
    
              distractor[t.distractor_curve[l]+2,layer,feature,l,timestep]+=saved_states_list[layer][timestep][0,feature,t.distractor_curve[l] % n.grid_size,t.distractor_curve[l] // n.grid_size]
              count_distractor[t.distractor_curve[l]+2,layer,feature,l,timestep] += 1
    
            target[t.position_markers[t.feature_target],layer,feature,curve_length,timestep]+=saved_states_list_disk[layer][timestep][0,feature,t.position_markers[t.feature_target]]
            count_target[t.position_markers[t.feature_target],layer,feature,curve_length,timestep] += 1
    
            distractor[t.position_markers[dis],layer,feature,curve_length,timestep]+=saved_states_list_disk[layer][timestep][0,feature,t.position_markers[dis]]
            count_distractor[t.position_markers[dis],layer,feature,curve_length,timestep] += 1
        if timestep < max_dur:
              for l in range(curve_length):
                target[t.target_curve[l]+2,layer,feature,l,dur:max_dur]+=np.array([saved_states_list[layer][dur-1][0,feature,t.target_curve[l] % n.grid_size,t.target_curve[l] // n.grid_size] for i in range(max_dur - dur)])
                count_target[t.target_curve[l]+2,layer,feature,l,dur:max_dur] += np.array([1  for i in range(max_dur - dur)])
    
                distractor[t.distractor_curve[l]+2,layer,feature,l,dur:max_dur]+=np.array([saved_states_list[layer][dur-1][0,feature,t.distractor_curve[l] % n.grid_size,t.distractor_curve[l] // n.grid_size]  for i in range(max_dur - dur)])
                count_distractor[t.distractor_curve[l]+2,layer,feature,l,dur:max_dur] += np.array([1 for i in range(max_dur - dur)])
    
              target[t.position_markers[t.feature_target],layer,feature,curve_length,dur:max_dur]+=np.array([saved_states_list_disk[layer][dur-1][0,feature,t.position_markers[t.feature_target]]  for i in range(max_dur - dur)])
              count_target[t.position_markers[t.feature_target],layer,feature,curve_length,dur:max_dur] += np.array([1  for i in range(max_dur - dur)])
    
              distractor[t.position_markers[dis],layer,feature,curve_length,dur:max_dur]+=np.array([saved_states_list_disk[layer][dur-1][0,feature,t.position_markers[dis]]  for i in range(max_dur - dur)])
              count_distractor[t.position_markers[dis],layer,feature,curve_length,dur:max_dur] += np.array([1  for i in range(max_dur - dur)])
      
    return(target,count_target,distractor,count_distractor)    
                    
                    

def latency_operation(TD,operation,TASK,CurveLength,number_interpolation_points=500,dur=39,criterion=0.3,position_on_curve=0):
    
    if operation == 'search':
        neurons = [0,1] #first two neurons correspond to the search operation
        position_on_curve = -1 #we arbitrarily put them at the end of the curve_length dimension in TD
    elif operation == 'trace':
        neurons = np.arange(2,TD.shape[0])
    else:
        raise Exception("Operation is either search or trace")
        
    if operation == 'trace' and TASK == 'search_trace':
        position_on_curve = 0 #in the search then trace task, the trace follows the search
    elif operation == 'trace' and TASK == 'trace_search':
        position_on_curve = CurveLength - 1 #in the trace then searcg task, the trace precedes the search
    elif TASK == 'trace':
        position_on_curve = position_on_curve
    print(position_on_curve)
        
    

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
    #latency[smoothed_interpolated_normalized_TD[:,:,:,:,-1] < 0.5] = np.nan #modulation at the end is lower than half of the max
    #latency[smoothed_interpolated_normalized_TD[:,:,:,:,0]>=0.5] = np.nan #modulation at the beggining is greater than half of the max
    latency[np.abs(np.
                   max(smoothed_interpolated_normalized_TD,axis=-1)-np.min(smoothed_interpolated_normalized_TD,axis=-1)) < 0.05] = np.nan #constant modulation


    latency[np.max(smoothed_interpolated_normalized_TD,axis=-1)<=0] = np.nan #modulation is negative
    latency[smoothed_interpolated_normalized_TD[:,:,:,-1] < smoothed_interpolated_normalized_TD[:,:,:,0]] = np.nan #modulation at the beginning is greater than at the end


    #Keeping non nan entries
    latency = latency[~np.isnan(latency)]
    latency = latency.astype('int')
    latency = np.linspace(0, dur-1, num=number_interpolation_points)[latency] 
    
    return latency


def compute_z(n,t,target_hist,distr_hist,grid_size,timesteps):
  if n.winner_disk:
      init = [torch.zeros_like(n.Y2), torch.zeros_like(n.Y1mod), torch.zeros_like(n.Y1), torch.zeros_like(n.Xmod)]
      init_disk = torch.autograd.grad(n.Z_disk[n.action], [n.Y2_disk, n.Y1mod_disk, n.Y1_disk, n.Xmod_disk], retain_graph=True, allow_unused=True)
  else:
      init = torch.autograd.grad(n.Z[n.action], [n.Y2, n.Y1mod, n.Y1, n.Xmod], retain_graph=True, allow_unused=True)
      init_disk = [torch.zeros_like(n.Y2_disk), torch.zeros_like(n.Y1mod_disk), torch.zeros_like(n.Y1_disk), torch.zeros_like(n.Xmod_disk)]


  Zy2 = init[0]
  Zy1mod = init[1]
  Zy1 = init[2]
  Zxmod = init[3]

  Zy2_disk = init_disk[0]
  Zy1mod_disk = init_disk[1]
  Zy1_disk = init_disk[2]
  Zxmod_disk = init_disk[3]

  target_activations_accessory = np.zeros((timesteps,t.curve_length))
  distractor_activations_accessory = np.zeros((timesteps,t.curve_length))

  for i in range(timesteps):
      ZXmod_prev = Zxmod
      Zxmod_disk_prev = Zxmod_disk

      Zy2 = torch.autograd.grad(n.Y1mod, n.Y2test, grad_outputs=Zy1mod, retain_graph=True, allow_unused=True)[0]
      tt = torch.autograd.grad(n.Y1mod, n.Y2test, grad_outputs=Zy1mod, retain_graph=True, allow_unused=True)[0]
      Zy2 = Zy2 + init[0]
      

      Zy1mod = torch.autograd.grad(n.Y2, n.Y1mod, grad_outputs=Zy2, retain_graph=True, allow_unused=True)[0]
      Zy1mod = Zy1mod + torch.autograd.grad(n.Xmod, n.Y1modtest, grad_outputs=Zxmod, retain_graph=True, allow_unused=True)[0]
      Zy1mod = Zy1mod + init[1]

      Zy1 = torch.autograd.grad(n.Y2, n.Y1, grad_outputs=Zy2, retain_graph=True, allow_unused=True)[0]
      Zy1 = Zy1 + init[2]

 
      Zxmod = torch.autograd.grad(n.Y1mod, n.Xmod, grad_outputs=Zy1mod, retain_graph=True, allow_unused=True)[0]
      Zxmod = Zxmod + torch.autograd.grad(n.Xmod, n.Xmodtest, grad_outputs=ZXmod_prev, retain_graph=True, allow_unused=True)[0]
      Zxmod = Zxmod + torch.autograd.grad(n.Xmod_disk, n.Xmodtest, grad_outputs=Zxmod_disk_prev, retain_graph=True, allow_unused=True)[0]
      Zxmod = Zxmod + init[3]
      for j in range(t.curve_length):
        target_activations_accessory[i,j] = Zy1mod[0,0,target_hist[j] % grid_size, target_hist[j] //grid_size]
        distractor_activations_accessory[i,j] = Zy1mod[0,0,distr_hist[j] % grid_size, distr_hist[j] //grid_size]


      Zy2_disk = torch.autograd.grad(n.Y1mod_disk, n.Y2test_disk, grad_outputs=Zy1mod_disk, retain_graph=True, allow_unused=True)[0]
      Zy2_disk = Zy2_disk + init_disk[0]

      Zy1mod_disk = torch.autograd.grad(n.Y2_disk, n.Y1mod_disk, grad_outputs=Zy2_disk, retain_graph=True, allow_unused=True)[0]
      Zy1mod_disk = Zy1mod_disk + torch.autograd.grad(n.Xmod_disk, n.Y1modtest_disk, grad_outputs=Zxmod_disk, retain_graph=True, allow_unused=True)[0]
      Zy1mod_disk = Zy1mod_disk + init_disk[1]

      Zy1_disk = torch.autograd.grad(n.Y2_disk, n.Y1_disk, grad_outputs=Zy2_disk, retain_graph=True, allow_unused=True)[0]
      Zy1_disk = Zy1_disk + init_disk[2]

      Zxmod_disk = Zxmod_disk + torch.autograd.grad(n.Y1mod_disk, n.Xmod_disk, grad_outputs=Zy1mod_disk, retain_graph=True, allow_unused=True)[0]
      Zxmod_disk = Zxmod_disk + torch.autograd.grad(n.Xmod_disk, n.Xmodtest_disk, grad_outputs=Zxmod_disk_prev, retain_graph=True, allow_unused=True)[0]
      Zxmod_disk = Zxmod_disk + torch.autograd.grad(n.Xmod, n.Xmodtest_disk, grad_outputs=ZXmod_prev, retain_graph=True, allow_unused=True)[0]
      Zxmod_disk = Zxmod_disk + init_disk[3]
  return(target_activations_accessory,distractor_activations_accessory)


    