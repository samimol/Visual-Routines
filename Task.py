# -*- coding: utf-8 -*-
"""
Created on Fri Oct 29 13:13:17 2021

@author: Sami
"""

import numpy as np
import torch
import random


class Task():

    def __init__(self, n_hidden_features):

        self.grid_size = 20

        self.state = 0
        self.trial_ended = False

        self.only_trace_curve = True # For the first part of the training of the trace then search task, we only do the curve tracing task

        self.state = 'intertrial'

        self.no_curves = True # First part of the curicculum is no curve, only the fixation point and a blue pixel

        self.current_reward = 0
        self.final_reward = 1.5
        
        
        self.n_hidden_features = n_hidden_features

        self.flowcontrol = {'intertrial': self.do_intertrial,
                            'go': self.do_go}

        if self.task_type != 'searchtrace' and self.task_type != 'trace' and self.task_type != 'tracesearch':
            raise Exception('Task type must be either searchtrace or trace or tracesearch')

    def stateReset(self):
        self.trial_ended = True
        self.state = 'intertrial'
        self.input = [self.display, self.display_disk]

    def doStep(self, action):
        self.trial_ended = False
        self.flowcontrol[self.state](action)
        reward = self.current_reward
        input = self.input
        trial_ended = self.trial_ended
        self.current_reward = 0
        return(input, reward, trial_ended)

    def do_intertrial(self, action):
            self.pickTrialType()
            self.input = [self.display, self.display_disk]
            self.state = 'go'


    def do_go(self, action):
            if (self.task_type == 'tracesearch' and self.only_trace_curve is False):
                pixel_chosen = torch.where(action == 1)[-1]
            else:
                pixel_chosen = torch.where(action == 1)[-1] - 2
            if pixel_chosen == self.target_curve[-1]:
                self.current_reward = self.current_reward + self.final_reward * 0.8
                self.stateReset()
            else:
                self.stateReset()

    def pickTrialType(self):
        position_red_marker = np.random.randint(2)
        position_yellow_marker = 1 if position_red_marker == 0 else 0
        self.position_markers = [position_red_marker, position_yellow_marker]
        self.feature_target = np.random.randint(2) 
        curve1,curve2 = self.PickCurve()
        self.DrawStimulus(curve1,curve2)
        
    def PickCurve(self):
        if self.no_curves:
            red_yellow_position = np.random.randint(self.grid_size**2)
            blue_position = np.random.randint(self.grid_size**2)
            while blue_position == red_yellow_position:  ########CHANGE THAT
                blue_position = np.random.randint(self.grid_size**2)
            curve1 = [red_yellow_position, blue_position]
            curve2 = []
        else:
            while True:
                try:
                    mask = np.zeros((self.grid_size, self.grid_size))
                    curve1, mask1 = self.make_curves([], mask)
                    curve2, mask2 = self.make_curves([], mask1)
                    break
                except IndexError:
                    pass
        return (curve1, curve2)
                
    def make_curves(self,curve, mask_original):
        mask = mask_original.copy()
        if len(curve) == 0:
            # Getting the pixels available
            x, y = np.where(mask == 0)
            
            # Chossing a random pixel among the one availables
            ind = np.random.randint(len(x))
            first_elem = x[ind] + y[ind] * self.grid_size
            
            #We mask this pixel, and call the function again
            mask[first_elem % self.grid_size, first_elem // self.grid_size] = 1
            curve.append(first_elem)
            return self.make_curves(curve, mask)
        else:
            # We look at the last pixel of the curve and get the coordinate of 
            # its neihbours
            x_end = curve[-1] % self.grid_size
            y_end = curve[-1] // self.grid_size
            consecutives_x = [x_end - 1, x_end + 1]
            consecutives_x = [i for i in consecutives_x if i >= 0 and i < self.grid_size]
            consecutives_y = [y_end - 1, y_end + 1]
            consecutives_y = [i for i in consecutives_y if i >= 0 and i < self.grid_size]
            
            while len(curve) < self.curve_length:  
                possible_next_value = []
                
                # If a neighbouring pixel is available, we record it as a
                # a possible value
                for i in range(len(consecutives_x)):
                    if mask[consecutives_x[i], y_end] == 0:
                        possible_next_value.append((consecutives_x[i], y_end))
                for i in range(len(consecutives_y)):
                    if mask[x_end, consecutives_y[i]] == 0:
                        possible_next_value.append((x_end, consecutives_y[i]))
                
                # We randomly choose the next pixel among the available ones
                # and mask the nehbours 
                next_value = random.choice(possible_next_value)
                curve.append(next_value[0] + next_value[1] * self.grid_size)
                for i in range(len(possible_next_value)):
                    mask[possible_next_value[i][0], possible_next_value[i][1]] = 1
                return self.make_curves(curve, mask)
            else:
                
                # If we reached the length of the curve, we only mask the nehbours
                for i in range(len(consecutives_x)):
                    mask[consecutives_x[i], y_end] = 1
                for i in range(len(consecutives_y)):
                    mask[x_end, consecutives_y[i]] = 1
                return curve, mask

        
class Trace(Task):
    
    def __init__(self,n_hidden_features):
        self.task_type = 'trace'
        super().__init__(n_hidden_features)
        
    def DrawStimulus(self,curve1,curve2):
        display = torch.zeros((1, self.n_hidden_features, self.grid_size, self.grid_size))
        display_disk = torch.zeros((1, self.n_hidden_features, 2))
        if self.no_curves:
                display[:, 0, curve1[0] % self.grid_size, curve1[0]//self.grid_size] = 1
                display[:, 3, curve1[1] % self.grid_size, curve1[1]//self.grid_size] = 1
        else:
                for i in range(len(curve1)):
                    if i == 0:  # red
                        display[:, 0, curve1[0] % self.grid_size, curve1[0]//self.grid_size] = 1
                    elif i == len(curve1)-1:  # blue
                        display[:, 3, curve1[i] % self.grid_size, curve1[i]//self.grid_size] = 1
                    else:  # green
                        display[:, 2, curve1[i] % self.grid_size, curve1[i]//self.grid_size] = 1
                for i in range(len(curve2)):
                    if i == 0:
                        display[:, 2, curve2[0] % self.grid_size, curve2[0]//self.grid_size] = 1
                    elif i == len(curve2)-1:
                        display[:, 3, curve2[i] % self.grid_size, curve2[i]//self.grid_size] = 1
                    else:
                        display[:, 2, curve2[i] % self.grid_size, curve2[i]//self.grid_size] = 1

        self.target_curve = curve1
        self.distractor_curve = curve2
        self.display = display
        self.display_disk = display_disk     
        
class SearchTrace(Task):
    
    def __init__(self,n_hidden_features):
        self.task_type = 'searchtrace'
        super().__init__(n_hidden_features)
        
    def DrawStimulus(self,curve1,curve2):
        display = torch.zeros((1, self.n_hidden_features, self.grid_size, self.grid_size))
        display_disk = torch.zeros((1, self.n_hidden_features, 2))
        if self.no_curves:
          display[:, self.feature_target, curve1[0] % self.grid_size, curve1[0]//self.grid_size] = 1
          display[:, 3, curve1[1] % self.grid_size, curve1[1]//self.grid_size] = 1
          display_disk[:, self.feature_target, self.position_markers[self.feature_target]] = 1 
        else:
          for i in range(len(curve1)):
              if i == 0:  # red
                  display[:, 0, curve1[0] % self.grid_size, curve1[0]//self.grid_size] = 1
              elif i == len(curve1)-1:  # blue
                  display[:, 3, curve1[i] % self.grid_size, curve1[i]//self.grid_size] = 1
              else:  # green
                  display[:, 2, curve1[i] % self.grid_size, curve1[i]//self.grid_size] = 1
          for i in range(len(curve2)):
              if i == 0:
                  display[:, 1, curve2[0] % self.grid_size, curve2[0]//self.grid_size] = 1
              elif i == len(curve2)-1:
                  display[:, 3, curve2[i] % self.grid_size, curve2[i]//self.grid_size] = 1
              else:
                  display[:, 2, curve2[i] % self.grid_size, curve2[i]//self.grid_size] = 1
          display_disk[:, self.feature_target, self.position_markers[self.feature_target]] = 1
        if self.no_curves:
            self.target_curve = curve1
            self.distractor_curve = curve2
        else:
            if self.feature_target == 0:
                self.target_curve = curve1
                self.distractor_curve = curve2
            else:
                self.target_curve = curve2
                self.distractor_curve = curve1
        self.display = display
        self.display_disk = display_disk
        
class TraceSearch(Task):      
    
    def __init__(self,n_hidden_features):
        self.task_type = 'tracesearch'
        super().__init__(n_hidden_features)
        
    def DrawStimulus(self,curve1,curve2):
        
        # The eye movement target is the red or yellow pixel so we reverse the
        # curves
        curve1.reverse()
        curve2.reverse()
        if not self.only_trace_curve:
            if self.no_curves:
                curve1.append(self.position_markers[self.feature_target])
            else:
                curve1.append(self.position_markers[0])
                curve2.append(self.position_markers[1])       
        display = torch.zeros((1, self.n_hidden_features, self.grid_size, self.grid_size))
        display_disk = torch.zeros((1, self.n_hidden_features, 2))
        if self.no_curves:
            display[:, 3, curve1[0] % self.grid_size, curve1[0]//self.grid_size] = 1
            display[:, self.feature_target, curve1[1] % self.grid_size, curve1[1]//self.grid_size] = 1
            if not self.only_trace_curve:
                display_disk[:, 0, self.position_markers[0]] = 1
                display_disk[:, 1, self.position_markers[1]] = 1
        else:
            if self.only_trace_curve:
                display[:, 0, curve1[len(curve1)-1] % self.grid_size, curve1[len(curve1)-1]//self.grid_size] = 1
                display[:, 1, curve2[len(curve2)-1] % self.grid_size, curve2[len(curve2)-1]//self.grid_size] = 1
                range1 = len(curve1) - 1
                range2 = len(curve2) - 1
            else:
                display[:, 0, curve1[len(curve1)-2] % self.grid_size, curve1[len(curve1)-2]//self.grid_size] = 1
                display[:, 1, curve2[len(curve2)-2] % self.grid_size, curve2[len(curve2)-2]//self.grid_size] = 1
                range1 = len(curve1) - 2
                range2 = len(curve2) - 2
            for i in range(range1):
                display[:, 2, curve1[i] % self.grid_size, curve1[i]//self.grid_size] = 1
            for i in range(range2):
                display[:, 2, curve2[i] % self.grid_size, curve2[i]//self.grid_size] = 1
            if self.feature_target == 0:
                display[:, 3, curve1[0] % self.grid_size, curve1[0]//self.grid_size] = 1
                display[:, 2, curve1[0] % self.grid_size, curve1[0]//self.grid_size] = 0
            else:
                display[:, 3, curve2[0] % self.grid_size, curve2[0]//self.grid_size] = 1
                display[:, 2, curve2[0] % self.grid_size, curve2[0]//self.grid_size] = 0
            if not self.only_trace_curve:
                display_disk[:, 0, self.position_markers[0]] = 1
                display_disk[:, 1, self.position_markers[1]] = 1
        if self.no_curves:
            self.target_curve = curve1
            self.distractor_curve = curve2
        else:
            if self.feature_target == 0:
                self.target_curve = curve1
                self.distractor_curve = curve2
            else:
                self.target_curve = curve2
                self.distractor_curve = curve1
        self.display = display
        self.display_disk = display_disk
    
        
    
