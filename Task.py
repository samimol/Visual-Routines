# -*- coding: utf-8 -*-
"""
Created on Fri Oct 29 13:13:17 2021

@author: Sami
"""

import numpy as np
import torch
import random


class Task():

    def __init__(self, n_hidden_features, tasktype):

        self.grid_size = 7

        self.state = 0
        self.counter = 0
        self.trialEnd = False

        self.onlyTrace = True

        self.state = 'intertrial'

        self.onlyblue = True

        self.cur_reward = 0
        self.fix_reward = 0.2
        self.fin_reward = 1.5
        
        self.n_hidden_features = n_hidden_features
        
        self.targ_display = torch.zeros((1, self.n_hidden_features, self.grid_size, self.grid_size))
        self.targ_display_disk = torch.zeros((1, self.n_hidden_features, 2))

        self.flowcontrol = {'intertrial': self.do_intertrial,
                            'go': self.do_go}

        self.tasktype = tasktype
        if self.tasktype != 'searchtrace' and self.tasktype != 'trace' and self.tasktype != 'tracesearch':
            raise Exception('Task type must be either searchtrace or trace or tracesearch')

    def stateReset(self):
        self.trialEnd = True
        self.counter = 0
        self.state = 'intertrial'
        self.nwInput = [self.target_display, self.target_display_disk]

    def doStep(self, action):
        self.trialEnd = False
        self.flowcontrol[self.state](action)
        reward = self.cur_reward
        nwInput = self.nwInput
        trialEnd = self.trialEnd
        self.cur_reward = 0
        return(nwInput, reward, trialEnd)

    def do_intertrial(self, action):
        if self.counter == self.intertrial_dur:
            self.pickTrialType()
            self.nwInput = [self.target_display, self.target_display_disk]
            self.state = 'go'
            self.counter = 0
        else:
            self.counter = self.counter + 1

    def do_go(self, action):
        # Only the last saccade, see matlab for step by step tracing
        if self.counter <= 0:
            if (self.tasktype == 'tracesearch' and self.onlyTrace is False):
                cond = torch.where(action == 1)[-1]
            else:
                cond = torch.where(action == 1)[-1] - 2
            if cond == self.trialTarget[-1] and (self.counter >= 0 or self.force_wait == 0):
                self.cur_reward = self.cur_reward + self.fin_reward * 0.8
                self.stateReset()
            else:
                self.stateReset()
        else:
            self.stateReset()

    def pickTrialType(self):
        positionRed = np.random.randint(2)
        positionYellow = 1 if positionRed == 0 else 0
        self.position = [positionRed, positionYellow]
        self.feature_target = np.random.randint(2) 
        curve1,curve2 = self.PickCurve()
        self.DrawStimulus(curve1,curve2)
        
    def PickCurve(self):
        if self.onlyblue:
            target_pos = np.random.randint(self.grid_size**2)
            blue_pos = np.random.randint(self.grid_size**2)
            while blue_pos == target_pos:
                blue_pos = np.random.randint(self.grid_size**2)
            curve1 = [target_pos, blue_pos]
            curve2 = []
        else:
            while True:
                try:
                    mask = np.zeros((self.grid_size, self.grid_size))
                    curve1, mask1 = self.make_curves([], mask)
                    curve2, mask2 = self.make_curves([], mask)
                    break
                except IndexError:
                    pass
        return (curve1, curve2)
                
    def make_curves(self,curve, mask_original):
        mask = mask_original.copy()
        if len(curve) == 0:
            x, y = np.where(mask == 0)
            ind = np.random.randint(len(x))
            first_elem = x[ind] + y[ind] * self.grid_size
            mask[first_elem % self.grid_size, first_elem // self.grid_size] = 1
            curve.append(first_elem)
            return self.make_curves(curve, mask)
        else:
            xend = curve[-1] % self.grid_size
            yend = curve[-1] // self.grid_size
            consecutivesx = [xend - 1, xend + 1]
            consecutivesx = [i for i in consecutivesx if i >= 0 and i < self.grid_size]
            consecutivesy = [yend - 1, yend + 1]
            consecutivesy = [i for i in consecutivesy if i >= 0 and i < self.grid_size]
            while len(curve) < self.curve_length:  # We append one at the end
                possible_next_value = []
                for i in range(len(consecutivesx)):
                    if mask[consecutivesx[i], yend] == 0:
                        possible_next_value.append((consecutivesx[i], yend))
                for i in range(len(consecutivesy)):
                    if mask[xend, consecutivesy[i]] == 0:
                        possible_next_value.append((xend, consecutivesy[i]))
                next_value = random.choice(possible_next_value)
                curve.append(next_value[0] + next_value[1] * self.grid_size)
                for i in range(len(possible_next_value)):
                    mask[possible_next_value[i][0], possible_next_value[i][1]] = 1
                return self.make_curves(curve, mask)
            else:
                for i in range(len(consecutivesx)):
                    mask[consecutivesx[i], yend] = 1
                for i in range(len(consecutivesy)):
                    mask[xend, consecutivesy[i]] = 1
                return curve, mask

        
class Trace(Task):
    
    def __init__(self):
        super().__init__()
        
    def DrawStimulus(self,curve1,curve2):
        targ_display = torch.zeros((1, self.n_hidden_features, self.grid_size, self.grid_size))
        targ_display_disk = torch.zeros((1, self.n_hidden_features, 2))
        if self.onlyblue:
                targ_display[:, 0, curve1[0] % self.grid_size, curve1[0]//self.grid_size] = 1
                targ_display[:, 3, curve1[1] % self.grid_size, curve1[1]//self.grid_size] = 1
        else:
                for i in range(len(curve1)):
                    if i == 0:  # red
                        targ_display[:, 0, curve1[0] % self.grid_size, curve1[0]//self.grid_size] = 1
                    elif i == len(curve1)-1:  # blue
                        targ_display[:, 3, curve1[i] % self.grid_size, curve1[i]//self.grid_size] = 1
                    else:  # green
                        targ_display[:, 2, curve1[i] % self.grid_size, curve1[i]//self.grid_size] = 1
                for i in range(len(curve2)):
                    if i == 0:
                        targ_display[:, 2, curve2[0] % self.grid_size, curve2[0]//self.grid_size] = 1
                    elif i == len(curve2)-1:
                        targ_display[:, 3, curve2[i] % self.grid_size, curve2[i]//self.grid_size] = 1
                    else:
                        targ_display[:, 2, curve2[i] % self.grid_size, curve2[i]//self.grid_size] = 1

        self.trialTarget = curve1
        self.trialDistr = curve2
        self.target_display = targ_display
        self.target_display_disk = targ_display_disk     
        
class SearchTrace(Task):
    
    def __init__(self):
        super().__init__()
        
    def DrawStimulus(self,curve1,curve2):
        targ_display = torch.zeros((1, self.n_hidden_features, self.grid_size, self.grid_size))
        targ_display_disk = torch.zeros((1, self.n_hidden_features, 2))
        if self.onlyblue:
          targ_display[:, self.feature_target, curve1[0] % self.grid_size, curve1[0]//self.grid_size] = 1
          targ_display[:, 3, curve1[1] % self.grid_size, curve1[1]//self.grid_size] = 1
          targ_display_disk[:, self.feature_target, self.position[self.feature_target]] = 1 
        else:
          for i in range(len(curve1)):
              if i == 0:  # red
                  targ_display[:, 0, curve1[0] % self.grid_size, curve1[0]//self.grid_size] = 1
              elif i == len(curve1)-1:  # blue
                  targ_display[:, 3, curve1[i] % self.grid_size, curve1[i]//self.grid_size] = 1
              else:  # green
                  targ_display[:, 2, curve1[i] % self.grid_size, curve1[i]//self.grid_size] = 1
          for i in range(len(curve2)):
              if i == 0:
                  targ_display[:, 1, curve2[0] % self.grid_size, curve2[0]//self.grid_size] = 1
              elif i == len(curve2)-1:
                  targ_display[:, 3, curve2[i] % self.grid_size, curve2[i]//self.grid_size] = 1
              else:
                  targ_display[:, 2, curve2[i] % self.grid_size, curve2[i]//self.grid_size] = 1
          targ_display_disk[:, self.feature_target, self.position[self.feature_target]] = 1
        if self.onlyblue:
            self.trialTarget = curve1
            self.trialDistr = curve2
        else:
            if self.feature_target == 0:
                self.trialTarget = curve1
                self.trialDistr = curve2
            else:
                self.trialTarget = curve2
                self.trialDistr = curve1
        self.target_display = targ_display
        self.target_display_disk = targ_display_disk
        
class TraceSearch(Task):      
    
    def __init__(self):
        super().__init__()
        
    def DrawStimulus(self,curve1,curve2):
        targ_display = torch.zeros((1, self.n_hidden_features, self.grid_size, self.grid_size))
        targ_display_disk = torch.zeros((1, self.n_hidden_features, 2))
        if self.onlyblue:
            targ_display[:, 3, curve1[0] % self.grid_size, curve1[0]//self.grid_size] = 1
            targ_display[:, self.feature_target, curve1[1] % self.grid_size, curve1[1]//self.grid_size] = 1
            if not self.onlytrace:
                targ_display_disk[:, 0, self.position[0]] = 1
                targ_display_disk[:, 1, self.position[1]] = 1
        else:
            if self.onlytrace:
                targ_display[:, 0, curve1[len(curve1)-1] % self.grid_size, curve1[len(curve1)-1]//self.grid_size] = 1
                targ_display[:, 1, curve2[len(curve2)-1] % self.grid_size, curve2[len(curve2)-1]//self.grid_size] = 1
                range1 = len(curve1) - 1
                range2 = len(curve2) - 1
            else:
                targ_display[:, 0, curve1[len(curve1)-2] % self.grid_size, curve1[len(curve1)-2]//self.grid_size] = 1
                targ_display[:, 1, curve2[len(curve2)-2] % self.grid_size, curve2[len(curve2)-2]//self.grid_size] = 1
                range1 = len(curve1) - 2
                range2 = len(curve2) - 2
            for i in range(range1):
                targ_display[:, 2, curve1[i] % self.grid_size, curve1[i]//self.grid_size] = 1
            for i in range(range2):
                targ_display[:, 2, curve2[i] % self.grid_size, curve2[i]//self.grid_size] = 1
            if self.feature_target == 0:
                targ_display[:, 3, curve1[0] % self.grid_size, curve1[0]//self.grid_size] = 1
                targ_display[:, 2, curve1[0] % self.grid_size, curve1[0]//self.grid_size] = 0
            else:
                targ_display[:, 3, curve2[0] % self.grid_size, curve2[0]//self.grid_size] = 1
                targ_display[:, 2, curve2[0] % self.grid_size, curve2[0]//self.grid_size] = 0
            if not self.onlytrace:
                targ_display_disk[:, 0, self.position[0]] = 1
                targ_display_disk[:, 1, self.position[1]] = 1
        if self.onlyblue:
            self.target_curve = curve1
            self.distractor_curve = curve2
        else:
            if self.feature_target == 0:
                self.target_curve = curve1
                self.distractor_curve = curve2
            else:
                self.target_curve = curve2
                self.distractor_curve = curve1
        self.target_display = targ_display
        self.target_display_disk = targ_display_disk
    
        
    
