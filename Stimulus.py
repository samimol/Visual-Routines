import numpy as np
import torch
import random


def make_curves(curve, mask_original, curvelength,grid_size):
    mask = mask_original.copy()
    if len(curve) == 0:
        x, y = np.where(mask == 0)
        ind = np.random.randint(len(x))
        first_elem = x[ind] + y[ind] * grid_size
        mask[first_elem % grid_size, first_elem // grid_size] = 1
        curve.append(first_elem)
        return make_curves(curve, mask_original, curvelength,grid_size)
    else:
        xend = curve[-1] % grid_size
        yend = curve[-1] // grid_size
        consecutivesx = [xend - 1, xend + 1]
        consecutivesx = [i for i in consecutivesx if i >= 0 and i < grid_size]
        consecutivesy = [yend - 1, yend + 1]
        consecutivesy = [i for i in consecutivesy if i >= 0 and i < grid_size]
        while len(curve) < curvelength:  # We append one at the end
            possible_next_value = []
            for i in range(len(consecutivesx)):
                if mask[consecutivesx[i], yend] == 0:
                    possible_next_value.append((consecutivesx[i], yend))
            for i in range(len(consecutivesy)):
                if mask[xend, consecutivesy[i]] == 0:
                    possible_next_value.append((xend, consecutivesy[i]))
            next_value = random.choice(possible_next_value)
            curve.append(next_value[0] + next_value[1] * grid_size)
            for i in range(len(possible_next_value)):
                mask[possible_next_value[i][0], possible_next_value[i][1]] = 1
            return make_curves(curve, mask_original, curvelength,grid_size)
        else:
            for i in range(len(consecutivesx)):
                mask[consecutivesx[i], yend] = 1
            for i in range(len(consecutivesy)):
                mask[xend, consecutivesy[i]] = 1
            return curve, mask


def MakeTraceStimulus(curve1, curve2, onlyblue, grid_size):
    targ_display = torch.zeros((1, 4, grid_size, grid_size))
    targ_display_disk = torch.zeros((1, 4, 2))
    if onlyblue:
        targ_display[:, 0, curve1[0] % grid_size, curve1[0]//grid_size] = 1
        targ_display[:, 3, curve1[1] % grid_size, curve1[1]//grid_size] = 1
    else:
        for i in range(len(curve1)):
            if i == 0:  # red
                targ_display[:, 0, curve1[0] % grid_size, curve1[0]//grid_size] = 1
            elif i == len(curve1)-1:  # blue
                targ_display[:, 3, curve1[i] % grid_size, curve1[i]//grid_size] = 1
            else:  # green
                targ_display[:, 2, curve1[i] % grid_size, curve1[i]//grid_size] = 1
        for i in range(len(curve2)):
            if i == 0:
                targ_display[:, 2, curve2[0] % grid_size, curve2[0]//grid_size] = 1
            elif i == len(curve2)-1:
                targ_display[:, 3, curve2[i] % grid_size, curve2[i]//grid_size] = 1
            else:
                targ_display[:, 2, curve2[i] % grid_size, curve2[i]//grid_size] = 1
    return [targ_display, targ_display_disk]


def MakeSearchTraceStimulus(curve1, curve2, feature_target, onlyblue, grid_size, position):
    targ_display = torch.zeros((1, 4, grid_size, grid_size))
    targ_display_disk = torch.zeros((1, 4, 2))
    if onlyblue:
        targ_display[:, feature_target, curve1[0] % grid_size, curve1[0]//grid_size] = 1
        targ_display[:, 3, curve1[1] % grid_size, curve1[1]//grid_size] = 1
        targ_display_disk[:, feature_target, position[feature_target]] = 1
    else:
        for i in range(len(curve1)):
            if i == 0:  # red
                targ_display[:, 0, curve1[0] % grid_size, curve1[0]//grid_size] = 1
            elif i == len(curve1)-1:  # blue
                targ_display[:, 3, curve1[i] % grid_size, curve1[i]//grid_size] = 1
            else:  # green
                targ_display[:, 2, curve1[i] % grid_size, curve1[i]//grid_size] = 1
        for i in range(len(curve2)):
            if i == 0:
                targ_display[:, 1, curve2[0] % grid_size, curve2[0]//grid_size] = 1
            elif i == len(curve2)-1:
                targ_display[:, 3, curve2[i] % grid_size, curve2[i]//grid_size] = 1
            else:
                targ_display[:, 2, curve2[i] % grid_size, curve2[i]//grid_size] = 1
        targ_display_disk[:, feature_target, position[feature_target]] = 1

    return [targ_display, targ_display_disk]


def MakeTraceSearchStimulus(curve1, curve2, feature_target, onlyblue, grid_size, position, onlytrace=False):
    targ_display = torch.zeros((1, 4, grid_size, grid_size))
    targ_display_disk = torch.zeros((1, 4, 2))
    if onlyblue:
        targ_display[:, 3, curve1[0] % grid_size, curve1[0]//grid_size] = 1
        targ_display[:, feature_target, curve1[1] % grid_size, curve1[1]//grid_size] = 1
        if not onlytrace:
            targ_display_disk[:, 0, position[0]] = 1
            targ_display_disk[:, 1, position[1]] = 1
    else:
        if onlytrace:
            targ_display[:, 0, curve1[len(curve1)-1] % grid_size, curve1[len(curve1)-1]//grid_size] = 1
            targ_display[:, 1, curve2[len(curve2)-1] % grid_size, curve2[len(curve2)-1]//grid_size] = 1
            range1 = len(curve1) - 1
            range2 = len(curve2) - 1
        else:
            targ_display[:, 0, curve1[len(curve1)-2] % grid_size, curve1[len(curve1)-2]//grid_size] = 1
            targ_display[:, 1, curve2[len(curve2)-2] % grid_size, curve2[len(curve2)-2]//grid_size] = 1
            range1 = len(curve1) - 2
            range2 = len(curve2) - 2
        for i in range(range1):
            targ_display[:, 2, curve1[i] % grid_size, curve1[i]//grid_size] = 1
        for i in range(range2):
            targ_display[:, 2, curve2[i] % grid_size, curve2[i]//grid_size] = 1
        if feature_target == 0:
            targ_display[:, 3, curve1[0] % grid_size, curve1[0]//grid_size] = 1
            targ_display[:, 2, curve1[0] % grid_size, curve1[0]//grid_size] = 0
        else:
            targ_display[:, 3, curve2[0] % grid_size, curve2[0]//grid_size] = 1
            targ_display[:, 2, curve2[0] % grid_size, curve2[0]//grid_size] = 0
        if not onlytrace:
            targ_display_disk[:, 0, position[0]] = 1
            targ_display_disk[:, 1, position[1]] = 1

    return [targ_display, targ_display_disk]


def GenerateForGivenLength(length, NumberIteration, task, onlyblue, grid_size,onlytrace=False):
    dictionnary = {'stimulus': [],
                   'stimulus_disk': [],
                   'feature_target': [],
                   'position': [],
                   'target_curve': [],
                   'distractor_curve': []}
    hashtable = []
    i=0
    while i < NumberIteration:
        positionRed = np.random.randint(2)
        if positionRed == 0:
            positionYellow = 1
        else:
            positionYellow = 0
        position = [positionRed, positionYellow]
        if onlyblue:
            target_pos = np.random.randint(grid_size**2)
            blue_pos = np.random.randint(grid_size**2)
            while blue_pos == target_pos:
                blue_pos = np.random.randint(grid_size**2)
            curve1 = [target_pos, blue_pos]
            curve2 = []
        else:
            while True:
                try:
                    mask = np.zeros((grid_size, grid_size))
                    curve1, mask1 = make_curves([], mask, length,grid_size)
                    curve2, mask2 = make_curves([], mask1, length,grid_size)
                    break
                except IndexError:
                    pass
        if task not in ['trace','searchtrace','tracesearch']:
            raise Exception("TaskType should be trace,searchtrace or tracesearch")
        if task == 'trace':
            feature_target = 0
            [targ_display, targ_display_disk] = MakeTraceStimulus(curve1, curve2, onlyblue, grid_size)
            target_curve = curve1
            distractor_curve = curve2
        elif task == 'searchtrace':
            feature_target = np.random.randint(2)

            [targ_display, targ_display_disk] = MakeSearchTraceStimulus(curve1, curve2, feature_target, onlyblue, grid_size, position)
            if onlyblue:
                target_curve = curve1
                distractor_curve = curve2
            else:
                if feature_target == 0:
                    target_curve = curve1
                    distractor_curve = curve2
                else:
                    target_curve = curve2
                    distractor_curve = curve1
        elif task == 'tracesearch':
            feature_target = np.random.randint(2)
            curve1.reverse()
            curve2.reverse()
            if not onlytrace:
                if onlyblue:
                    curve1.append(position[feature_target])
                else:
                    curve1.append(position[0])
                    curve2.append(position[1])
            [targ_display, targ_display_disk] = MakeTraceSearchStimulus(curve1, curve2, feature_target, onlyblue, grid_size, position, onlytrace=onlytrace)
            if onlyblue:
                target_curve = curve1
                distractor_curve = curve2
            else:
                if feature_target == 0:
                    target_curve = curve1
                    distractor_curve = curve2
                else:
                    target_curve = curve2
                    distractor_curve = curve1
        tt = hash(str([targ_display, targ_display_disk]))
        if tt not in hashtable:
            hashtable.append(tt)
            dictionnary['stimulus'].append(targ_display)
            dictionnary['stimulus_disk'].append(targ_display_disk)
            dictionnary['feature_target'].append(feature_target)
            dictionnary['position'].append(position)
            dictionnary['target_curve'].append(target_curve)
            dictionnary['distractor_curve'].append(distractor_curve)
            i+=1
    return dictionnary
