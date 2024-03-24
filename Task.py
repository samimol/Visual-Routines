# -*- coding: utf-8 -*-
"""
Created on Fri Oct 29 13:13:17 2021

@author: Sami
"""

import numpy as np
import torch
import random


class Task():

    def __init__(self, n_hidden_features,grid_size):

        self.grid_size = grid_size

        self.state = 0
        self.trial_ended = False

        self.only_trace_curve = False # For the first part of the training of the trace then search task, we only do the curve tracing task

        self.state = 'intertrial'
        
        self.curve_length = 2

        self.no_curves = True # First part of the curicculum is no curve, only the fixation point and a blue pixel

        self.current_reward = 0
        self.final_reward = 1.5
        
        self.add_distractors = False
        
        
        self.n_hidden_features = n_hidden_features

        self.flowcontrol = {'intertrial': self.do_intertrial,
                            'go': self.do_go}

        if self.task_type != 'searchtrace' and self.task_type != 'trace' and self.task_type != 'tracesearch':
            raise Exception('Task type must be either searchtrace or trace or tracesearch')

    def state_reset(self):
        self.trial_ended = True
        self.state = 'intertrial'
        self.input = [self.display, self.display_disk]

    def do_step(self, action):
        self.trial_ended = False
        self.flowcontrol[self.state](action)
        reward = self.current_reward
        input = self.input
        trial_ended = self.trial_ended
        self.current_reward = 0
        return(input, reward, trial_ended)

    def do_intertrial(self, action):
            self.pick_trial_type()
            self.input = [self.display, self.display_disk]
            self.state = 'go'


    def do_go(self, action):
            if (self.task_type == 'tracesearch' and self.only_trace_curve is False):
                pixel_chosen = torch.where(action == 1)[-1]
            else:
                pixel_chosen = torch.where(action == 1)[-1] - 2
            if pixel_chosen == self.target_curve[-1]:
                self.current_reward = self.current_reward + self.final_reward * 0.8
                self.state_reset()
            else:
                self.state_reset()

    def pick_trial_type(self):
        position_red_marker = np.random.randint(2)
        position_yellow_marker = 1 if position_red_marker == 0 else 0
        self.position_markers = [position_red_marker, position_yellow_marker]
        self.feature_target = np.random.randint(2)
        self.t_junction = False
        if np.random.rand() > 10:
            self.t_junction = True
        curve1,curve2 = self.pick_curve()
        self.draw_stimulus(curve1,curve2)
        
         
        
    def pick_curve(self):
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
                    if not self.t_junction or self.curve_length == 3:
                        flag_equidist = False
                        while not flag_equidist:
                            curve1, mask1 = self.make_curves([], mask,self.curve_length)
                            curve2, mask2 = self.make_curves([], mask1,self.curve_length)
                            x1,y1 = curve1[0] % self.grid_size, curve1[0] // self.grid_size
                            x2,y2 = curve1[-1] % self.grid_size, curve1[-1] // self.grid_size
                            x3,y3 = curve2[-1] % self.grid_size, curve2[-1] // self.grid_size
                            flag_equidist = np.linalg.norm(np.array([x1,y1])-np.array([x2,y2])) == np.linalg.norm(np.array([x1,y1])-np.array([x3,y3]))
                            
                        if self.add_distractors:
                            self.distractors = []
                            for p in range(10):
                                curve_distrc, mask2 = self.make_curves([], mask2,3)
                                self.distractors.append(curve_distrc)
                    else:
                        curve1,curve2, mask, temp_masked_indexs = self.make_curves_t_junction(mask)
                        curve1, mask1 = self.make_curves(curve1, mask)
                        if mask1[temp_masked_indexs[0][0],temp_masked_indexs[0][1]] == 1:
                            mask1[temp_masked_indexs[0][0],temp_masked_indexs[0][1]] = 0
                        if mask1[temp_masked_indexs[1][0],temp_masked_indexs[1][1]] == 1:
                            mask1[temp_masked_indexs[1][0],temp_masked_indexs[1][1]] = 0
                      
                        curve2, mask2 = self.make_curves(curve2, mask1)
                    break
                except IndexError:
                    pass
        #curve1 = [456,426,427, 428, 429, 430, 431, 432, 433, 434, 464, 494, 524, 554, 584, 614, 613, 612, 611, 610, 580, 550]
        #curve2 = [552, 522, 492, 491, 490, 489, 488, 518, 548, 578, 608, 638, 668, 669, 670, 671, 672, 673, 674, 675, 676, 646]
        #curve1.reverse()
        #curve2.reverse()
        return (curve1, curve2)
                
        
    def make_curves_t_junction(self,mask_original):
        mask = mask_original.copy()
        curve1 = []
        curve2 = []
        # Getting the pixels available for the center pixel of the junction (not the borders)
        x, y = np.where(mask == 0)
        y = y[np.where((x != 0) & (x != 14))]
        x = x[np.where((x != 0) & (x != 14))]

        x = x[np.where((y != 0) & (y != 14))]
        y = y[np.where((y != 0) & (y != 14))]

        # Choosing a random pixel among the one availables
        ind = np.random.randint(len(x))
        center = x[ind] + y[ind] * self.grid_size
        top = x[ind]-1 + (y[ind]) * self.grid_size
        bottom = x[ind]+1 + (y[ind]) * self.grid_size
        left = x[ind] + (y[ind] - 1) * self.grid_size
        right = x[ind] + (y[ind] + 1) * self.grid_size
        
        # Getting the corners to mask them
        corn_top_left = x[ind] - 1 + (y[ind] - 1) * self.grid_size
        corn_top_right = x[ind] - 1 + (y[ind] + 1) * self.grid_size
        corn_bot_left = x[ind] + 1 + (y[ind] - 1) * self.grid_size
        corn_bot_right = x[ind] + 1 + (y[ind] + 1) * self.grid_size

        #We mask those pixels, and call the function again
        mask[center % self.grid_size, center // self.grid_size] += 1
        mask[top % self.grid_size, top // self.grid_size] += 1
        mask[bottom % self.grid_size, bottom // self.grid_size] += 1
        mask[left % self.grid_size, left // self.grid_size] += 1
        mask[right % self.grid_size, right // self.grid_size] += 1
        
        mask[corn_top_left % self.grid_size, corn_top_left // self.grid_size] += 1
        mask[corn_top_right % self.grid_size, corn_top_right // self.grid_size] += 1
        mask[corn_bot_left % self.grid_size, corn_bot_left // self.grid_size] += 1
        mask[corn_bot_right % self.grid_size, corn_bot_right // self.grid_size] += 1

        #choose curve1 and curve2
        if np.random.rand() < 0.5:
            curve1 = [top,center,bottom]
            curve2 = [left,center,right]
            
            #to not have a snake eating itself
            temp_masked_indexs = [(x[ind],y[ind]-2),(x[ind],y[ind]+2)]
            mask[x[ind],y[ind]-2]  += 1
            mask[x[ind],y[ind]+2]  += 1
        else:
            curve2 = [top,center,bottom]
            curve1 = [left,center,right]  
                 
            #to not have a snake eating itself
            temp_masked_indexs = [(x[ind]-2,y[ind]),(x[ind]+2,y[ind])]
            mask[x[ind]-2,y[ind]] += 1
            mask[x[ind]+2,y[ind]] += 1

        #shuffling randomly start and end of curves
        if np.random.rand() < 0.5:
            curve1.reverse()
        if np.random.rand() < 0.5:
            curve2.reverse()

        return curve1, curve2,mask,temp_masked_indexs


    def make_curves(self,curve, mask_original,curve_length):
        mask = mask_original.copy()
        if len(curve) == 0:
            # Getting the pixels available
            x, y = np.where(mask == 0)
            
            # Chossing a random pixel among the one availables
            ind = np.random.randint(len(x))
            first_elem = x[ind] + y[ind] * self.grid_size
            
            #We mask this pixel, and call the function again
            mask[first_elem % self.grid_size, first_elem // self.grid_size] += 1
            curve.append(first_elem)
            return self.make_curves(curve, mask,curve_length)
        else:
            if self.t_junction:
                #shuffling randomly start and end of curves to propagate in both direction of the junction
                if np.random.rand() < 0.5:
                    curve.reverse()

            # We look at the last pixel of the curve and get the coordinate of 
            # its neihbours
            x_end = curve[-1] % self.grid_size
            y_end = curve[-1] // self.grid_size
            consecutives_x = [x_end - 1, x_end + 1]
            consecutives_x = [i for i in consecutives_x if i >= 0 and i < self.grid_size]
            consecutives_y = [y_end - 1, y_end + 1]
            consecutives_y = [i for i in consecutives_y if i >= 0 and i < self.grid_size]
            
            while len(curve) < curve_length:  
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
                for i in range(len(consecutives_x)):
                    mask[consecutives_x[i], y_end] += 1
                for i in range(len(consecutives_y)):
                    mask[x_end, consecutives_y[i]] += 1
                return self.make_curves(curve, mask,curve_length)
            else:
                for i in range(len(consecutives_x)):
                    mask[consecutives_x[i], y_end] += 1
                for i in range(len(consecutives_y)):
                    mask[x_end, consecutives_y[i]] += 1
                    
                # We also mask begining of the curve
                mask[(curve[0]%self.grid_size)-1, (curve[0]) // self.grid_size] += 1
                mask[(curve[0]%self.grid_size)+1, (curve[0]) // self.grid_size] += 1
                mask[(curve[0]%self.grid_size), (curve[0] // self.grid_size)-1] += 1
                mask[(curve[0]%self.grid_size), (curve[0] // self.grid_size)+1] += 1
                
                mask[(curve[-1]%self.grid_size)-1, (curve[-1]) // self.grid_size] += 1
                mask[(curve[-1]%self.grid_size)+1, (curve[-1]) // self.grid_size] += 1
                mask[(curve[-1])%self.grid_size, (curve[-1] // self.grid_size)-1] += 1
                mask[(curve[-1])%self.grid_size, (curve[-1] // self.grid_size)+1] += 1
                

                return curve, mask

        
class Trace(Task):
    
    def __init__(self,n_hidden_features,grid_size):
        self.task_type = 'trace'
        super().__init__(n_hidden_features,grid_size)
        
    def draw_stimulus(self,curve1,curve2):
        display = torch.zeros((1, self.n_hidden_features, self.grid_size, self.grid_size))
        display_disk = torch.zeros((1, self.n_hidden_features, 2))
        if self.no_curves:
                display[:, 0, curve1[0] % self.grid_size, curve1[0]//self.grid_size] = 0
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
                if self.add_distractors:
                    for distr in self.distractors:
                        for i in range(len(distr)):
                            if i == 0:
                                display[:, 2, distr[0] % self.grid_size, distr[0]//self.grid_size] = 1
                            elif i == 2:
                                display[:, 3, distr[i] % self.grid_size, distr[i]//self.grid_size] = 1
                            else:
                                display[:, 2, distr[i] % self.grid_size, distr[i]//self.grid_size] = 1
        self.target_curve = curve1
        self.distractor_curve = curve2
        self.display = display
        self.display_disk = display_disk     
        
class SearchTrace(Task):
    
    def __init__(self,n_hidden_features,grid_size):
        self.task_type = 'searchtrace'
        self.contrast = 1
        super().__init__(n_hidden_features,grid_size)
        
    def draw_stimulus(self,curve1,curve2):
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
                display[:, 0, curve1[0] % self.grid_size, curve1[0]//self.grid_size] = self.contrast
                self.distractor_curve = curve2
            else:
                self.target_curve = curve2
                display[:, 1, curve2[0] % self.grid_size, curve2[0]//self.grid_size] = self.contrast
                self.distractor_curve = curve1
        self.display = display
        self.display_disk = display_disk
        
class TraceSearch(Task):      
    
    def __init__(self,n_hidden_features,grid_size):
        self.task_type = 'tracesearch'
        super().__init__(n_hidden_features,grid_size)
        
    def draw_stimulus(self,curve1,curve2):
        
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
    
        
    