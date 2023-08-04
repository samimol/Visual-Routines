# Recurrent neural networks that learn multi-step visual routines with reinforcement learning

This repository contains the code to reproduce the results of [Recurrent neural networks that learn multi-step visual routines with reinforcement learning](https://www.biorxiv.org/content/10.1101/2023.07.03.547198v3).

## Requirements 

To install requirements:
 ```
 pip install -r requirements.txt
```

## Training the model

To train the model on the trace task, for a maximum number of 50000 trials, on a grid of size 15x15 pixels, for curves up to 9 pixels run the following command:

```
python main.py --train --task trace --max_trials 50000 --grid_size 15 --max_length 9
```
The model will be saved in an output folder made in the directory

## Testing the model

To test the saved model on 15000 trials and save the neural activities that will be used for the analysis, run the following command:
```
python main.py --test --task trace --max_trials 15000 --grid_size 15 --max_length 9 --path results/network_trace_6.pkl
```

## Reproduvcing the figures in the paper

To reproduce the analysis and plots shown in the paper, run the jupyter notebook analysis.ipynb
