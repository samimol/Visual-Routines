# -*- coding: utf-8 -*-
"""
Created on Wed Nov 24 15:14:09 2021

@author: Sami
"""
from Stimulus import *
import pickle
import matplotlib.pyplot as plt

GRIDSIZE = 20
ITERATION = 3000
MAXLENGTH = 20

def SaveDic(file,filename):
  with open(filename+'.pkl', 'wb') as handle:
      pickle.dump(file, handle, protocol=pickle.HIGHEST_PROTOCOL)  


## Trace Task
TraceTask = {'onlyblue': {}}
Length = list(range(3,MAXLENGTH+1))
Length = [str(length) for length in Length]
TraceTask.update({x: {} for x in Length})

TraceTask['onlyblue'] = GenerateForGivenLength(0, 160, 'trace', True, GRIDSIZE)
for i in range(3,MAXLENGTH):
    TraceTask[str(i)] = GenerateForGivenLength(i, ITERATION, 'trace', False, GRIDSIZE)
    print(i)
TraceTask[str(MAXLENGTH)] = GenerateForGivenLength(i, 10000, 'trace', False, GRIDSIZE)

SaveDic(TraceTask,'TraceTask')
print('Trace Done')

## Search Trace Task
SearchTraceTask = {'onlyblue': {}}
Length = list(range(3,MAXLENGTH+1))
Length = [str(length) for length in Length]
SearchTraceTask.update({x: {} for x in Length})

SearchTraceTask['onlyblue'] = GenerateForGivenLength(0, ITERATION, 'searchtrace', True, GRIDSIZE)
for i in range(3,MAXLENGTH+1):
    SearchTraceTask[str(i)] = GenerateForGivenLength(i, ITERATION, 'searchtrace', False, GRIDSIZE)
    print(i)
    
SaveDic(SearchTraceTask,'SearchTraceTask')
print('SearchTrace Done')

##TraceSearchTask Onlytrace
TraceSearchTaskOnlyTrace = {'onlyblue': {}}
Length = list(range(3,MAXLENGTH+1))
Length = [str(length) for length in Length]
TraceSearchTaskOnlyTrace.update({x: {} for x in Length})

TraceSearchTaskOnlyTrace['onlyblue'] = GenerateForGivenLength(0, ITERATION, 'searchtrace', True, GRIDSIZE,onlytrace=True)
for i in range(3,MAXLENGTH+1):
    TraceSearchTaskOnlyTrace[str(i)] = GenerateForGivenLength(i, ITERATION, 'searchtrace', False, GRIDSIZE,onlytrace=True)
    print(i)
SaveDic(TraceSearchTaskOnlyTrace,'TraceSearchTaskOnlyTrace')
print('TraceSearch1 Done')


##TraceSearchTask 
TraceSearchTask = {'onlyblue': {}}
Length = list(range(3,MAXLENGTH+1))
Length = [str(length) for length in Length]
TraceSearchTask.update({x: {} for x in Length})

TraceSearchTask['onlyblue'] = GenerateForGivenLength(0, ITERATION, 'searchtrace', True, GRIDSIZE,onlytrace=False)
for i in range(3,MAXLENGTH+1):
    TraceSearchTask[str(i)] = GenerateForGivenLength(i, ITERATION, 'searchtrace', False, GRIDSIZE,onlytrace=False)
    print(i)
SaveDic(TraceSearchTask,'TraceSearchTask')
print('TraceSearch2 Done')
