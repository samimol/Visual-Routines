import numpy as np
import torch
import random


def make_curves(curve, mask_original, curvelength):
    mask = mask_original.copy()
    if len(curve) == 0:
        x, y = np.where(mask == 0)
        ind = np.random.randint(len(x))
        first_elem = x[ind] + y[ind] * 7
        mask[first_elem % 7, first_elem // 7] = 1
        curve.append(first_elem)
        return make_curves(curve, mask, curvelength)
    else:
        xend = curve[-1] % 7
        yend = curve[-1] // 7
        consecutivesx = [xend - 1, xend + 1]
        consecutivesx = [i for i in consecutivesx if i >= 0 and i < 7]
        consecutivesy = [yend - 1, yend + 1]
        consecutivesy = [i for i in consecutivesy if i >= 0 and i < 7]
        while len(curve) < curvelength:  # We append one at the end
            possible_next_value = []
            for i in range(len(consecutivesx)):
                if mask[consecutivesx[i], yend] == 0:
                    possible_next_value.append((consecutivesx[i], yend))
            for i in range(len(consecutivesy)):
                if mask[xend, consecutivesy[i]] == 0:
                    possible_next_value.append((xend, consecutivesy[i]))
            next_value = random.choice(possible_next_value)
            curve.append(next_value[0] + next_value[1] * 7)
            for i in range(len(possible_next_value)):
                mask[possible_next_value[i][0], possible_next_value[i][1]] = 1
            return make_curves(curve, mask, curvelength)
        else:
            for i in range(len(consecutivesx)):
                mask[consecutivesx[i], yend] = 1
            for i in range(len(consecutivesy)):
                mask[xend, consecutivesy[i]] = 1
            return curve, mask


def MakeTraceStimulus(curve1, curve2, onlyblue):
    targ_display = torch.zeros((1, 4, 7, 7))
    targ_display_disk = torch.zeros((1, 4, 2))
    if onlyblue:
        targ_display[:, 0, curve1[0] % 7, curve1[0]//7] = 1
        targ_display[:, 3, curve1[1] % 7, curve1[1]//7] = 1
    else:
        for i in range(len(curve1)):
            if i == 0:  # red
                targ_display[:, 0, curve1[0] % 7, curve1[0]//7] = 1
            elif i == len(curve1)-1:  # blue
                targ_display[:, 3, curve1[i] % 7, curve1[i]//7] = 1
            else:  # green
                targ_display[:, 2, curve1[i] % 7, curve1[i]//7] = 1
        for i in range(len(curve2)):
            if i == 0:
                targ_display[:, 2, curve2[0] % 7, curve2[0]//7] = 1
            elif i == len(curve2)-1:
                targ_display[:, 3, curve2[i] % 7, curve2[i]//7] = 1
            else:
                targ_display[:, 2, curve2[i] % 7, curve2[i]//7] = 1
    return [targ_display, targ_display_disk]


def MakeSearchTraceStimulus(curve1, curve2, feature_target, onlyblue):
    targ_display = torch.zeros((1, 4, 7, 7))
    targ_display_disk = torch.zeros((1, 4, 2))
    if onlyblue:
        targ_display[:, feature_target, curve1[0] % 7, curve1[0]//7] = 1
        targ_display[:, 3, curve1[1] % 7, curve1[1]//7] = 1
        targ_display_disk[:, feature_target, 0] = 1
    else:
        for i in range(len(curve1)):
            if i == 0:  # red
                targ_display[:, 0, curve1[0] % 7, curve1[0]//7] = 1
            elif i == len(curve1)-1:  # blue
                targ_display[:, 3, curve1[i] % 7, curve1[i]//7] = 1
            else:  # green
                targ_display[:, 2, curve1[i] % 7, curve1[i]//7] = 1
        for i in range(len(curve2)):
            if i == 0:
                targ_display[:, 1, curve2[0] % 7, curve2[0]//7] = 1
            elif i == len(curve2)-1:
                targ_display[:, 3, curve2[i] % 7, curve2[i]//7] = 1
            else:
                targ_display[:, 2, curve2[i] % 7, curve2[i]//7] = 1
        if feature_target == 0:
            targ_display_disk[:, 0, 0] = 1
        else:
            targ_display_disk[:, 1, 0] = 1
    return [targ_display, targ_display_disk]


def MakeTraceSearchStimulus(curve1, curve2, feature_target, onlyblue, onlytrace=False):
    targ_display = torch.zeros((1, 4, 7, 7))
    targ_display_disk = torch.zeros((1, 4, 2))
    if onlyblue:
        targ_display[:, 3, curve1[0] % 7, curve1[0]//7] = 1
        targ_display[:, feature_target, curve1[1] % 7, curve1[1]//7] = 1
        if not onlytrace:
            targ_display_disk[:, 0, 0] = 1
            targ_display_disk[:, 1, 1] = 1
    else:
        if onlytrace:
            targ_display[:, 0, curve1[len(curve1)-1] % 7, curve1[len(curve1)-1]//7] = 1
            targ_display[:, 1, curve2[len(curve2)-1] % 7, curve2[len(curve2)-1]//7] = 1
            range1 = len(curve1) - 1
            range2 = len(curve2) - 1
        else:
            targ_display[:, 0, curve1[len(curve1)-2] % 7, curve1[len(curve1)-2]//7] = 1
            targ_display[:, 1, curve2[len(curve2)-2] % 7, curve2[len(curve2)-2]//7] = 1
            range1 = len(curve1) - 2
            range2 = len(curve2) - 2
        for i in range(range1):
            targ_display[:, 2, curve1[i] % 7, curve1[i]//7] = 1
        for i in range(range2):
            targ_display[:, 2, curve2[i] % 7, curve2[i]//7] = 1
        if feature_target == 0:
            targ_display[:, 3, curve1[0] % 7, curve1[0]//7] = 1
            targ_display[:, 2, curve1[0] % 7, curve1[0]//7] = 0
        else:
            targ_display[:, 3, curve2[0] % 7, curve2[0]//7] = 1
            targ_display[:, 2, curve2[0] % 7, curve2[0]//7] = 0
        if not onlytrace:
            targ_display_disk[:, 0, 0] = 1
            targ_display_disk[:, 1, 1] = 1

    return [targ_display, targ_display_disk]


def GenerateForGivenLength(length, NumberIteration, task, onlytrace, onlyblue):
    dictionnary = {'stimulus': [],
                   'stimulus_disk': [],
                   'feature_target': [],
                   'target_curve': [],
                   'distractor_curve': []}
    hashtable = []
    for i in range(NumberIteration):
        if onlyblue:
            target_pos = np.random.randint(7**2)
            blue_pos = np.random.randint(7**2)
            while blue_pos == target_pos:
                blue_pos = np.random.randint(7**2)
            curve1 = [target_pos, blue_pos]
            curve2 = []
        else:
            while True:
                try:
                    mask = np.zeros((7, 7))
                    curve1, mask1 = make_curves([], mask, length)
                    curve2, mask2 = make_curves([], mask1, length)
                    break
                except IndexError:
                    pass
        if task == 'trace':
            feature_target = 0
            [targ_display, targ_display_disk] = MakeTraceStimulus(curve1, curve2, onlyblue)
            target_curve = curve1
            distractor_curve = curve2
        elif task == 'searchtrace':
            feature_target = np.random.randint(2)

            [targ_display, targ_display_disk] = MakeSearchTraceStimulus(curve1, curve2, feature_target, onlyblue)
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
                    curve1.append(feature_target)
                else:
                    curve1.append(0)
                    curve2.append(1)
            [targ_display, targ_display_disk] = MakeTraceSearchStimulus(curve1, curve2, feature_target, onlytrace, onlyblue)
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
            dictionnary['target_curve'].append(target_curve)
            dictionnary['distractor_curve'].append(distractor_curve)
    return dictionnary


Random position marche pour search trace
il faut verifier que ca marche pour tracesearch
puis changer le code
