# -*- coding: utf-8 -*-
"""
Created on Fri Aug  4 16:52:41 2023

@author: Sami
"""

import argparse
import os
import numpy as np
import pickle
import pandas as pd

from Network import *
from Task import *
from HelperFunctions import *

def setup_args():
    parser = argparse.ArgumentParser()
    
    parser.add_argument('--train', action='store_true')
    parser.add_argument('--test', action='store_true')
    parser.add_argument('--task', type=str, default='TRACE')
    parser.add_argument('--max_trials', type=int, default=50000)
    parser.add_argument('--grid_size', type=int, default=15)
    parser.add_argument('--max_length', type=int, default=9)
    parser.add_argument('--path', type=str, default="")
    
    args = parser.parse_args()
    return args

if __name__ == "__main__":
    args = setup_args()
    
    if args.train:
        (n,performance_track,generalization) = train(task=args.task,
                                                     max_trials=args.max_trials,
                                                     grid_size=args.grid_size,
                                                     max_length=args.max_length)
        os.makedirs('results',exist_ok=True)
        
        filename = os.path.join('results','network_'+args.task+'_'+str(args.max_length)+'.pkl')
        with open(filename, 'wb') as output:  # Overwrites any existing file.
            pickle.dump(n, output, pickle.HIGHEST_PROTOCOL)
            
        np.save(os.path.join('results','performance_'+args.task+'_'+str(args.max_length)), np.array(performance_track))
        np.save(os.path.join('results','generalisation_'+args.task+'_'+str(args.max_length)), generalization)

        

    elif args.test:
        with open(args.path, 'rb') as pickle_file:
            n = pickle.load(pickle_file)
        
        (target_activations,distractor_activations,performance,colour_disk_history,target_curve_history,distractor_curve_history,max_trials,curve_length) = run_and_save(n,
                                                                                                                                                                         task=args.task,
                                                                                                                                                                         curve_length=args.max_length,
                                                                                                                                                                         max_trials=args.max_trials,
                                                                                                                                                                         grid_size=arg.grid_size)                                                                                                                                                                         


        resultsdict = {'target_activations':[target_activations], # Activations of the neurons when their RF fall on the target curve
               'distractor_activations':[distractor_activations],  # Activations of the neurons when their RF fall on the distractor curve
               'performance':[performance],
               'colour_disk_history':[colour_disk_history], # Colour of the disk for the search then trace and trace then search task
               'target_curve_history':[target_curve_history], # Coordinates of the target curve pixels
               'distractor_curve_history':[distractor_curve_history], # Coordinates of the distractor curve pixels
               'number_trials':[max_trials],
               'curve_length':[curve_length]}

        df = pd.DataFrame(resultsdict)
        df.to_pickle(os.path.join('results','network_activations_'+args.task+'_'+str(args.max_length)+'.pkl'))
            
        
