# -*- coding: utf-8 -*-
"""
Created on Fri Oct 29 13:13:17 2021

@author: Sami
"""

import numpy as np
import torch


class Task():

    def __init__(self, n_hidden_features, tasktype):

        self.grid_size = 7

        self.state = 0
        self.counter = 0
        self.trialEnd = False

        self.onlyTrace = False

        self.state = 'intertrial'

        self.onlyblue = False

        self.intertrial_dur = 0

        self.cur_reward = 0
        self.fix_reward = 0.2
        self.fin_reward = 1.5
        
        self.n_hidden_features = n_hidden_features

        self.flowcontrol = {'intertrial': self.do_intertrial,
                            'go': self.do_go}

        self.tasktype = tasktype
        if self.tasktype != 'searchtrace' and self.tasktype != 'trace' and self.tasktype != 'tracesearch':
            raise Exception('Task type must be either searchtrace or trace or tracesearch')

    def stateReset(self):
        self.trialEnd = True
        self.counter = 0
        self.state = 'intertrial'
        self.nwInput = torch.zeros((1, self.grid_size ** 2 + 2, self.n_hidden_features))

    def doStep(self, action, dataset,indexs):
        self.trialEnd = False
        self.flowcontrol[self.state](action,dataset,indexs)
        reward = self.cur_reward
        nwInput = self.nwInput
        trialEnd = self.trialEnd
        self.cur_reward = 0
        return(nwInput, reward, trialEnd)

    def do_intertrial(self, action, dataset,indexs):
        if self.counter == self.intertrial_dur:
            self.pickTrialType(dataset)
            self.state = 'go'
            self.counter = 0
        else:
            self.counter = self.counter + 1

    def do_go(self, action,dataset,indexs):
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

    def pickTrialType(self, dataset,indexs):
        index = np.random.choice(indexs)
        self.nwInput = [dataset['stimulus'][index], dataset['stimulus_disk'][index]]
        self.trialTarget = dataset['target_curve'][index]
        self.trialDistr = dataset['distractor_curve'][index]
        self.feature_target = dataset['feature_target'][index]
