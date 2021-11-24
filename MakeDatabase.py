# -*- coding: utf-8 -*-
"""
Created on Wed Nov 24 15:14:09 2021

@author: Sami
"""
from Stimulus import *
import pickle
import matplotlib.pyplot as plt


## Trace Task
TraceTask = {'onlyblue': {}}
Length = list(range(3,7))
Length = [str(length) for length in Length]
TraceTask.update({x: {} for x in Length})

TraceTask['onlyblue'] = GenerateForGivenLength(0, 50, 'trace', True, 7)
for i in range(2,7):
    TraceTask[str(i)] = GenerateForGivenLength(i, 50, 'trace', False, 7)



## Search Trace Task
SearchTraceTask = {'onlyblue': {}}
Length = list(range(3,7))
Length = [str(length) for length in Length]
SearchTraceTask.update({x: {} for x in Length})

SearchTraceTask['onlyblue'] = GenerateForGivenLength(0, 50, 'searchtrace', True, 7)
for i in range(2,7):
    SearchTraceTask[str(i)] = GenerateForGivenLength(i, 50, 'searchtrace', False, 7)


##TraceSearchTask Onlytrace
SearchTraceTaskOnlyTrace = {'onlyblue': {}}
Length = list(range(3,7))
Length = [str(length) for length in Length]
SearchTraceTaskOnlyTrace.update({x: {} for x in Length})

SearchTraceTaskOnlyTrace['onlyblue'] = GenerateForGivenLength(0, 50, 'searchtrace', True, 7,onlytrace=True)
for i in range(2,7):
    SearchTraceTaskOnlyTrace[str(i)] = GenerateForGivenLength(i, 50, 'searchtrace', False, 7,onlytrace=True)

##TraceSearchTask 
SearchTraceTask = {'onlyblue': {}}
Length = list(range(3,7))
Length = [str(length) for length in Length]
SearchTraceTask.update({x: {} for x in Length})

SearchTraceTask['onlyblue'] = GenerateForGivenLength(0, 50, 'searchtrace', True, 7,onlytrace=False)
for i in range(2,7):
    SearchTraceTask[str(i)] = GenerateForGivenLength(i, 50, 'searchtrace', False, 7,onlytrace=False)

