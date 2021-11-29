# -*- coding: utf-8 -*-
"""
Created on Fri Oct 29 12:29:28 2021

@author: Sami
"""
import numpy as np

import torch
import torch.nn as nn


class CustomLayer(nn.Module):

    def __init__(self):
        super().__init__()

    def MyRelu(self, x):
        thresh = 6
        x[x == 0] = 0
        x = torch.clamp(x, 0)
        x[thresh <= x] = torch.log(1.5 * x[thresh <= x] + 1) + thresh - torch.log(torch.tensor(1.5 * thresh + 1))
        return x

    def AverageTraces(self, traces, mask, diag=None):
        traces *= mask
        traces = torch.clamp(traces, None, 1)
        if self.LayerType != "output" or diag is True:
            m = torch.mean(traces[:, :, [0, 1, 1, 2], [1, 0, 2, 1]], axis=2)
            m = m[:, :, None]
            traces[:, :, [0, 1, 1, 2], [1, 0, 2, 1]] = m
        elif self.LayerType == "output":
            intermmask = torch.ones((2 * self.grid_size - 1, 2 * self.grid_size - 1))
            intermmask[self.grid_size - 1, self.grid_size - 1] = 0
            m = torch.mean(traces[:, :, intermmask == 1], axis=2)
            traces[:, :, intermmask == 1] = m.T
        return(traces)

    def init_weights(self, layer, diag=None, disk=False):
        K = torch.zeros(layer.weight.shape)
        if layer.bias is not None:
            b = torch.zeros(layer.bias.shape)
        for f in range(K.shape[0]):
            for f2 in range(K.shape[1]):
                if not disk:
                    if self.LayerType != "output":
                        K[f, f2, 1, 1] = 0.1 * np.random.rand()
                        K[f, f2, [0, 1, 1, 2], [1, 0, 2, 1]] = 0.1 * np.random.rand()
                    else:
                        if diag is not None:
                            K[f, f2, 1, 1] = 0.1 * np.random.rand()
                        else:
                            K[f, f2, :, :] = - 0.1 * np.random.rand() * torch.ones((K.shape[2], K.shape[3])) / 100
                            K[f, f2, 6, 6] = 0.1 * np.random.rand()
                else:
                    K[f, f2, 0] = 0.1 * np.random.rand()
            if layer.bias is not None:
                b[f] = 0.1 * np.random.rand()
        layer.weight = torch.nn.Parameter(K)
        if layer.bias is not None:
            layer.bias = torch.nn.Parameter(b)

    def UpdateWeight(self, layer, upper, beta, delta, mask=None, z=None, diag=None, disk=False):
        with torch.no_grad():
            if layer.bias is not None:
                trace = torch.autograd.grad(upper, [layer.weight, layer.bias], grad_outputs=z, retain_graph=True, allow_unused=True)
                bias_traces = trace[1]
                bias_traces[:] = torch.mean(bias_traces)
            else:
                trace = torch.autograd.grad(upper, layer.weight, grad_outputs=z, retain_graph=True, allow_unused=True)
            weight_traces = trace[0]
            if not disk:
                weight_traces = self.AverageTraces(weight_traces, mask, diag)
            weight_update = layer.weight + beta * delta * weight_traces
            layer.weight.copy_(weight_update)
            if layer.bias is not None:
                bias_update = layer.bias + beta * delta * bias_traces  # / 49
                layer.bias.copy_(bias_update)
        return(layer)

    def toCustom(self, device):
        for i in range(2):
            if self.LayerType == 'input':
                self.u[i].to(device)
                self.umod[i].to(device)
                self.umask = self.umask.to(device)
            elif self.LayerType == 'hidden':
                self.v[i].to(device)
                self.t[i].to(device)
                self.vtmask = self.vtmask.to(device)
                if self.has_Ymod:
                    self.u[i].to(device)
                    self.umask = self.umask.to(device)
                    if self.upper_ymod:
                        self.umod[i].to(device)
            elif self.LayerType == 'output':
                self.w[i].to(device)
                self.skip[i].to(device)
                self.wmask = self.wmask.to(device)
                self.skipmask = self.skipmask.to(device)


class HiddenLayer(CustomLayer):

    def __init__(self, feature_in, feature_out, has_Ymod=False, upper_ymod=False):
        super().__init__()
        kernel_size = 3
        self.LayerType = 'hidden'
        self.has_Ymod = has_Ymod
        self.upper_ymod = upper_ymod
        self.v = [nn.Conv2d(feature_in, feature_out, kernel_size, stride=1, padding='same'), nn.Conv1d(feature_in, feature_out, 1, stride=1, padding='same')]
        self.t = [nn.Conv2d(feature_in, feature_out, kernel_size, stride=1, padding='same', bias=False), nn.Conv1d(feature_in, feature_out, 1, stride=1, padding='same', bias=False)]
        if has_Ymod:
            self.u = [nn.Conv2d(feature_out, feature_out, kernel_size, stride=1, padding='same'), nn.Conv1d(feature_out, feature_out, 1, stride=1, padding='same')]
            if upper_ymod:
                self.umod = [nn.Conv2d(feature_out, feature_out, kernel_size, stride=1, padding='same', bias=False), nn.Conv1d(feature_out, feature_out, 1, stride=1, padding='same', bias=False)]

        # Initializing weights
        self.init_weights(self.v[0])
        self.init_weights(self.v[1], disk=True)
        self.init_weights(self.t[0])
        self.init_weights(self.t[1], disk=True)
        if has_Ymod:
            self.init_weights(self.u[0])
            self.init_weights(self.u[1], disk=True)
            if upper_ymod:
                self.init_weights(self.umod[0])
                self.init_weights(self.umod[1], disk=True)

        # Setting the mask to have connections only between neighboours
        #self.vtmask = torch.tensor([[0,4/168,0],[4/168,1/49,4/168],[0,4/168,0]],requires_grad = False)
        #self.vtmask = torch.tile(self.vtmask,(feature_out,feature_in,1,1))
        # if has_Ymod:
        #  self.umask = torch.tensor([[0,4/168,0],[4/168,1/49,4/168],[0,4/168,0]],requires_grad = False)
        #  self.umask = torch.tile(self.umask,(feature_out,feature_out,1,1))

        self.vtmask = torch.tensor([[0, 1, 0], [1, 1, 1], [0, 1, 0]], requires_grad=False)
        self.vtmask = torch.tile(self.vtmask, (feature_out, feature_in, 1, 1))
        if has_Ymod:
            self.umask = torch.tensor([[0, 1, 0], [1, 1, 1], [0, 1, 0]], requires_grad=False)
            self.umask = torch.tile(self.umask, (feature_out, feature_out, 1, 1))

    def forward(self, lower_y=None, lower_ymod=None, upper_y=None, upper_ymod=None, det=False):
        current_y = self.MyRelu(self.v[0](lower_y[0]) + self.t[0](lower_ymod[0]))
        current_y_disk = self.MyRelu(self.v[1](lower_y[1]) + self.t[1](lower_ymod[1]))
        if self.has_Ymod:
            if upper_ymod is not None:
                current_ymod = self.MyRelu((self.u[0](upper_y[0]) + self.umod[0](upper_ymod[0])) * current_y)
                current_ymod_disk = self.MyRelu((self.u[1](upper_y[1]) + self.umod[1](upper_ymod[1])) * current_y_disk)
            else:
                current_ymod = self.MyRelu(current_y * self.u[0](upper_y[0]))
                current_ymod_disk = self.MyRelu(current_y_disk * self.u[1](upper_y[1]))
            return([current_y, current_y_disk], [current_ymod, current_ymod_disk])
        return([current_y, current_y_disk])

    def UpdateLayer(self, upper, z, beta, delta):
        self.v[0] = self.UpdateWeight(self.v[0], upper[0][0], beta, delta, self.vtmask, z[0][0])
        self.t[0] = self.UpdateWeight(self.t[0], upper[0][0], beta, delta, self.vtmask, z[0][0])

        self.v[1] = self.UpdateWeight(self.v[1], upper[0][1], beta, delta, z=z[0][1], disk=True)
        self.t[1] = self.UpdateWeight(self.t[1], upper[0][1], beta, delta, z=z[0][1], disk=True)

        if self.has_Ymod:
            self.u[0] = self.UpdateWeight(self.u[0], upper[1][0], beta, delta, self.umask, z[1][0])
            self.u[1] = self.UpdateWeight(self.u[1], upper[1][1], beta, delta, z=z[1][1], disk=True)
            if self.upper_ymod:
                self.umod[0] = self.UpdateWeight(self.umod[0], upper[1][0], beta, delta, self.umask, z[1][0])
                self.umod[1] = self.UpdateWeight(self.umod[1][1], upper[1], beta, delta, z=z[1][1], disk=True)


class InputLayer(CustomLayer):

    def __init__(self, feature_in, feature_out):
        super().__init__()
        self.LayerType = 'input'
        kernel_size = 3
        self.u = [nn.Conv2d(feature_out, feature_in, kernel_size, stride=1, padding='same'), nn.Conv1d(feature_out, feature_in, 1, stride=1, padding='same')]
        self.umod = [nn.Conv2d(feature_out, feature_in, kernel_size, stride=1, padding='same', bias=False), nn.Conv1d(feature_out, feature_in, 1, stride=1, padding='same', bias=False)]

        # Initializing weights
        self.init_weights(self.u[0])
        self.init_weights(self.umod[0])

        self.init_weights(self.u[1], disk=True)
        self.init_weights(self.umod[1], disk=True)

        # Setting the mask to have connections only between neighboours
        #self.umask = torch.tensor([[0,4/168,0],[4/168,1/49,4/168],[0,4/168,0]],requires_grad = False)
        self.umask = torch.tensor([[0, 1, 0], [1, 1, 1], [0, 1, 0]], requires_grad=False)
        self.umask = torch.tile(self.umask, (feature_in, feature_out, 1, 1))

    def forward(self, upper_y, upper_ymod, input):
        Y = input[0]
        Y_disk = input[1]
        Ymod = self.u[0](upper_y[0]) + self.umod[0](upper_ymod[0])
        #Ymod = self.MyRelu(Ymod * Y[0])

        Ymod_disk = self.u[1](upper_y[1]) + self.umod[1](upper_ymod[1])
        #Ymod_disk = self.MyRelu(Ymod_disk * Y_disk)

        return([Y, Y_disk], [Ymod, Ymod_disk])

    def UpdateLayer(self, upper, z, beta, delta):
        self.u[0] = self.UpdateWeight(self.u[0], upper[0], beta, delta, self.umask, z[0])
        self.umod[0] = self.UpdateWeight(self.umod[0], upper[0], beta, delta, self.umask, z[0])

        self.u[1] = self.UpdateWeight(self.u[1], upper[1], beta, delta, z=z[1], disk=True)
        self.umod[1] = self.UpdateWeight(self.umod[1], upper[1], beta, delta, z=z[1], disk=True)


class OutputLayer(CustomLayer):

    def __init__(self, hidden_features, input_features, feature_out,grid_size):
        super().__init__()
        self.grid_size = grid_size
        kernel_size = 2 * self.grid_size - 1
        self.LayerType = 'output'
        self.w = [nn.Conv2d(hidden_features, feature_out, kernel_size, stride=1, padding='same'), nn.Conv1d(hidden_features, feature_out, 1, stride=1, padding='same')]
        self.skip = [nn.Conv2d(input_features, feature_out, 3, stride=1, padding='same', bias=False), nn.Conv1d(input_features, feature_out, 1, stride=1, padding='same', bias=False)]

        self.wmask = (1/14) * torch.ones_like(self.w[0].weight)  # In numpy we do the average over 49*49-49 = 2352 whereas here over 13*13 -1 = 168 so we have to divide by 14 (number of strides)
        #self.wmask = torch.ones_like(self.w[0].weight,requires_grad = False)
        self.wmask[:, :, self.grid_size - 1, self.grid_size - 1] = 1/49

        self.skipmask = torch.zeros_like(self.skip[0].weight, requires_grad=False)
        self.skipmask[:, :, 1, 1] = 1  # /49

        # Initializing weights
        self.init_weights(self.w[0])
        self.init_weights(self.skip[0], diag=True)

        self.init_weights(self.w[1], disk=True)
        self.init_weights(self.skip[1], disk=True)

    def forward(self, lower_y, inputmod):
        Y = self.w[0](lower_y[0]) + self.skip[0](inputmod[0])
        Y_disk = self.w[1](lower_y[1]) + self.skip[1](inputmod[1])
        return(Y, Y_disk)

    def UpdateLayer(self, upper, beta, delta, winner_disk):
        if winner_disk:
            self.w[1] = self.UpdateWeight(self.w[1], upper, beta, delta, disk=True)
            self.skip[1] = self.UpdateWeight(self.skip[1], upper, beta, delta, disk=True)
        else:
            self.w[0] = self.UpdateWeight(self.w[0], upper, beta, delta, self.wmask)
            self.skip[0] = self.UpdateWeight(self.skip[0], upper, beta, delta, self.skipmask, diag=True)


class HorizLayer(CustomLayer):

    def __init__(self, features, grid_size):
        super().__init__()
        self.grid_size = grid_size
        self.features = features
        self.LayerType = 'horizontal'
        self.weight_GridToDisk = torch.nn.Parameter(0.1 * torch.rand((features, 1)))
        self.weight_DiskToGrid = torch.nn.Parameter(0.1 * torch.rand((features, 1)))
        self.weight_GridToGrid = torch.ones((1, features, grid_size, grid_size))
        self.weight_DiskToDisk = torch.ones((1, features, 2))

        for i in range(features):
            self.weight_GridToGrid[0, i, :, :] *= 0.1 * np.random.rand()
            self.weight_DiskToDisk[0, i, :] *= 0.1 * np.random.rand()

        self.weight_GridToGrid = torch.nn.Parameter(self.weight_GridToGrid)
        self.weight_DiskToDisk = torch.nn.Parameter(self.weight_DiskToDisk)

    def forward(self, Y, Y_disk):

        Update_disk = Y_disk * self.weight_DiskToDisk
        Update_disk = Update_disk + torch.sum(Y, axis=(2, 3)).T * self.weight_GridToDisk

        Update_grid = Y * self.weight_GridToGrid
        Update_grid = torch.reshape(Update_grid, [1, self.features, self.grid_size**2])
        Update_grid = Update_grid + torch.sum(Y_disk, axis=2).T * self.weight_DiskToGrid
        Update_grid = torch.reshape(Update_grid, [1, self.features, self.grid_size, self.grid_size])

        return(Update_grid, Update_disk)

    def UpdateLayer(self, upper, z, beta, delta):

        traces_GridToDisk = torch.autograd.grad(upper[0], self.weight_GridToDisk, grad_outputs=z[0], retain_graph=True, allow_unused=True)[0]
        traces_DiskToDisk = torch.autograd.grad(upper[0], self.weight_DiskToDisk, grad_outputs=z[0], retain_graph=True, allow_unused=True)[0]
        traces_DiskToGrid = torch.autograd.grad(upper[1], self.weight_DiskToGrid, grad_outputs=z[1], retain_graph=True, allow_unused=True)[0]
        traces_GridToGrid = torch.autograd.grad(upper[1], self.weight_GridToGrid, grad_outputs=z[1], retain_graph=True, allow_unused=True)[0]

        traces_GridToGrid[:] = torch.mean(traces_GridToGrid, axis=(2, 3), keepdim=True)
        traces_DiskToDisk[:] = torch.mean(traces_DiskToDisk, axis=(2), keepdim=True)

        with torch.no_grad():
            weight_GridToDisk_update = self.weight_GridToDisk + beta * delta * traces_GridToDisk
            weight_DiskToGrid_update = self.weight_DiskToGrid + beta * delta * traces_DiskToGrid
            weight_GridToGrid_update = self.weight_GridToGrid + beta * delta * traces_GridToGrid
            weight_DiskToDisk_update = self.weight_DiskToDisk + beta * delta * traces_DiskToDisk

            self.weight_GridToDisk.copy_(weight_GridToDisk_update)
            self.weight_DiskToGrid.copy_(weight_DiskToGrid_update)
            self.weight_GridToGrid.copy_(weight_GridToGrid_update)
            self.weight_DiskToDisk.copy_(weight_DiskToDisk_update)
