from numpy.core.arrayprint import format_float_scientific
# -*- coding: utf-8 -*-
"""
Created on Fri Oct 29 12:29:28 2021

@author: Sami
"""
import numpy as np
import torch
import torch.nn as nn
from itertools import repeat

class LocallyConnected2d(nn.Module):
    
    #Adapted from https://discuss.pytorch.org/t/locally-connected-layers/26979
    def __init__(self, in_channels, out_channels, output_size, kernel_size, stride, bias=False):
        super(LocallyConnected2d, self).__init__()
        output_size = tuple(repeat(output_size, 2))
        self.weight = torch.zeros(1, out_channels, in_channels, output_size[0], output_size[1], kernel_size**2)
        for f in range(output_size[0]):
            for f2 in range(output_size[1]):
                self.weight[0,0,:,f,f2,8] = 1 
        self.weight = nn.Parameter(self.weight)
        if bias:
            self.bias = nn.Parameter(
                torch.randn(1, out_channels, output_size[0], output_size[1])
            )
        else:
            self.register_parameter('bias', None)
        self.kernel_size = tuple(repeat(kernel_size, 2))
        self.stride =tuple(repeat(stride, 2))
        
    def forward(self, x):
        x = torch.nn.functional.pad(x,pad=(1, 1, 1, 1))
        _, c, h, w = x.size()
        kh, kw = self.kernel_size
        dh, dw = self.stride
        x = x.unfold(2, kh, dh).unfold(3, kw, dw)
        x = x.contiguous().view(*x.size()[:-2], -1)
        # Sum in in_channel and kernel_size dims
        out = (x.unsqueeze(1) * self.weight).sum([2, -1])
        if self.bias is not None:
            out += self.bias
        return out

class CustomLayer(nn.Module):

    def __init__(self,grid_size,weight_sharing=True):
        self.initialisation_range = 0.1
        self.grid_size = grid_size
        self.weight_sharing = weight_sharing
        super().__init__()

    def activation_function(self, x):
        thresh = 6
        x[x == 0] = 0
        x = torch.clamp(x, 0)
        x[thresh <= x] = torch.log(1.5 * x[thresh <= x] + 1) + thresh - torch.log(torch.tensor(1.5 * thresh + 1))
        return x
    
    def gating_function(self,x):
      param = 100
      return param*x/(torch.sqrt(1+(param**2)*(x**2)))
      #return x

    def average_trace(self, traces, mask,disk=False):
        traces *= mask
        if disk:
            m = torch.mean(traces[:,:,[0,2]],axis = 2)
            m = m[:,:,None]
            traces[:,:,[0,2]] = m
        else:
          if self.LayerType != "output":
              m = torch.mean(traces[:, :, [0, 1, 1, 2], [1, 0, 2, 1]], axis=2)
              m = m[:, :, None]
              traces[:, :, [0, 1, 1, 2], [1, 0, 2, 1]] = m
          elif self.LayerType == "output":
              intermmask = torch.ones((2 * self.grid_size - 1, 2 * self.grid_size - 1))
              intermmask[self.grid_size - 1, self.grid_size - 1] = 0
              m = torch.mean(traces[:, :, intermmask == 1], axis=2)
              traces[:, :, intermmask == 1] = m.T
        return(traces)

    def initialise_weights(self, layer, disk=False, connections_to_neighbours=True,marker=False):
        kernel = torch.zeros(layer.weight.shape)
        if layer.bias is not None:
            bias = torch.zeros(layer.bias.shape)
        if not disk:
            if self.weight_sharing:
                if self.LayerType != "output":
                  if marker == False:
                    kernel[:, :, 1, 1] = self.initialisation_range * torch.rand(kernel.shape[0],kernel.shape[1])
                    if connections_to_neighbours:
                        kernel[:, :, [0,1,1,2],[1,0,2,1]] = self.initialisation_range * torch.rand(kernel.shape[0],kernel.shape[1],4)
                  else:
                    for f in range(kernel.shape[0]):
                      for f2 in range(kernel.shape[1]):
                        kernel[f,f2,0] = np.random.rand()*0.1
                        kernel[f,f2,2] = np.random.rand()*0.1
                        kernel[f,f2,1] = np.random.rand()*0.1                    
                else:
                  if marker == False:
                    kernel[:] = - self.initialisation_range * torch.rand(kernel.shape[1],)[None,:,None,None] / 100
                    kernel[:, :, self.grid_size - 1, self.grid_size - 1] = self.initialisation_range * torch.rand(kernel.shape[0],kernel.shape[1])
                  else:
                    for f in range(kernel.shape[1]):
                      kernel[0,f,[0,2]] = -np.random.rand()*0.1/10
                      kernel[0,f,1] = np.random.rand()*0.1
                if layer.bias is not None:
                    bias = self.initialisation_range * torch.rand_like(bias)
            else:
              if marker == False:
                if self.LayerType != "output":
                        kernel[0,:,:,:,:,4] = self.initialisation_range * torch.rand((kernel.shape[1],kernel.shape[2],kernel.shape[3],kernel.shape[4],))
                        if connections_to_neighbours:
                          kernel[0,:,:,:,:,[1,3,4,5,7]] = self.initialisation_range * torch.rand((kernel.shape[1],kernel.shape[2],kernel.shape[3],kernel.shape[4],5))
                else:
                    kernel = - self.initialisation_range * torch.rand_like(kernel)/ 100
                    kernel[:, :, self.grid_size - 1, self.grid_size - 1] = self.initialisation_range * torch.rand(kernel.shape[0],kernel.shape[1])  
              else:
                    for f in range(kernel.shape[0]):
                      for f2 in range(kernel.shape[1]):
                        kernel[f,f2,0] = np.random.rand()*0.1
                        kernel[f,f2,2] = np.random.rand()*0.1
                        kernel[f,f2,1] = np.random.rand()*0.1    
              if layer.bias is not None:
                bias = self.initialisation_range * torch.rand_like(layer.bias)
        else:
            kernel = self.initialisation_range * torch.rand_like(layer.weight)
        layer.weight = torch.nn.Parameter(kernel)
        if layer.bias is not None:
            layer.bias = torch.nn.Parameter(bias)

    def update_weights(self, layer, upper, beta, delta, mask=None, z=None, average=True,average_disk=False):
        with torch.no_grad():
            
            # Getting the derivative of the output unit with respect to the 
            # weight and bias
            if layer.bias is not None:
                delta_weight = torch.autograd.grad(upper, [layer.weight, layer.bias], grad_outputs=z, retain_graph=True, allow_unused=True)
                delta_bias = delta_weight[1]
                delta_bias = torch.clamp(delta_bias, None, 1)
            else:
                delta_weight = torch.autograd.grad(upper, layer.weight, grad_outputs=z, retain_graph=True, allow_unused=True)
            delta_weight = delta_weight[0]
            delta_weight = torch.clamp(delta_weight, None, 1)
            
            # Averaging the weight and bias and updating the weights with RPE
            if (average or average_disk) and self.weight_sharing:
                #if self.LayerType == 'output':
                delta_weight = self.average_trace(delta_weight, mask,disk=average_disk)
                #else:
                    #delta_weight = delta_weight*mask
            elif average and not self.weight_sharing:
                delta_weight = delta_weight*mask
                
            weight_update = layer.weight + beta * delta * delta_weight
            layer.weight.copy_(weight_update)
            if layer.bias is not None:
                bias_update = layer.bias + beta * delta * delta_bias
                layer.bias.copy_(bias_update)
        return(layer)

    def to(self, device):
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
                
                
class InputLayer(CustomLayer):

    def __init__(self, feature_in, feature_out,grid_size,weight_sharing):
        super().__init__(grid_size,weight_sharing)
        self.LayerType = 'input'
        kernel_size = 3
        
        # First element is grid to grid, second element is disk to disk
        if self.weight_sharing:
            self.umod = [nn.Conv2d(feature_out, feature_in, kernel_size, stride=1, padding='same', bias=True), nn.Conv1d(feature_out, feature_in, 3, stride=1, padding='same', bias=True)]
        else:
            self.umod = [LocallyConnected2d(feature_out, feature_in,output_size=self.grid_size, kernel_size=kernel_size, stride=1, bias=True), nn.Conv1d(feature_out, feature_in, 3, stride=1, padding='same', bias=True)]

        # Initializing weights
        self.initialise_weights(self.umod[0])
        self.initialise_weights(self.umod[1], disk=False,marker=True)

        # Setting the mask to have connections only between neighboours
        self.umask = torch.clone(self.umod[0].weight.detach())
        self.umask[self.umask != 0] = 1    

        self.u_disk_mask = torch.clone(self.umod[1].weight.detach())
        self.u_disk_mask[self.u_disk_mask != 0] = 1  
             
    def register_params(self):
        self.register_parameter(name='u0_weight',param=self.umod[0].weight)
        self.register_parameter(name='u0_bias',param=self.umod[0].bias)
        
        self.register_parameter(name='u1_weight',param=self.umod[1].weight)
        self.register_parameter(name='u1_bias',param=self.umod[1].bias)
        
    def forward(self, upper_y, upper_ymod, input):
        Y = input[0]
        Y_disk = input[1]
        
        Ymod = self.umod[0](upper_ymod[0])
        Ymod_disk = self.umod[1](upper_ymod[1])

        return([Y, Y_disk], [Ymod, Ymod_disk])

    def update_layer(self, upper, z, beta, delta):
        self.umod[0] = self.update_weights(self.umod[0], upper[0], beta, delta, self.umask, z[0],average=self.weight_sharing)
        self.umod[1] = self.update_weights(self.umod[1], upper[1], beta, delta,self.u_disk_mask, z=z[1], average=False,average_disk=True)

    def mask_weights(self):
        with torch.no_grad():
            self.umod[0].weight *= self.umask
            self.umod[1].weight *= self.u_disk_mask


class HiddenLayer(CustomLayer):

    def __init__(self, feature_in, feature_out,grid_size,weight_sharing=True, has_Ymod=False, upper_ymod=False):
        super().__init__(grid_size,weight_sharing)
        kernel_size = 3
        self.LayerType = 'hidden'
        self.has_Ymod = has_Ymod # If the layer has a modulated group
        self.upper_ymod = upper_ymod # If the higher layer has a modulated group
        
        if self.weight_sharing:
            self.v = [nn.Conv2d(feature_in, feature_out, kernel_size, stride=1, padding='same', bias=False), nn.Conv1d(feature_in, feature_out, 1, stride=1, padding='same', bias=False)]
            self.t = [nn.Conv2d(feature_in, feature_out, kernel_size, stride=1, padding='same', bias=True), nn.Conv1d(feature_in, feature_out, 3, stride=1, padding='same', bias=True)]
            if has_Ymod:
                self.u = [nn.Conv2d(feature_out, feature_out, kernel_size, stride=1, padding='same', bias=False), nn.Conv1d(feature_out, feature_out, 3, stride=1, padding='same', bias=False)]
                if upper_ymod:
                    self.umod = [nn.Conv2d(feature_out, feature_out, kernel_size, stride=1, padding='same', bias=False), nn.Conv1d(feature_out, feature_out, 3, stride=1, padding='same', bias=False)]
        else:
            self.v = [LocallyConnected2d(feature_in, feature_out, output_size=self.grid_size, kernel_size=kernel_size, stride=1, bias=False), nn.Conv1d(feature_in, feature_out, 1, stride=1, padding='same', bias=False)]
            self.t = [LocallyConnected2d(feature_in, feature_out, output_size=self.grid_size, kernel_size=kernel_size,stride=1, bias=True), nn.Conv1d(feature_in, feature_out, 3, stride=1, padding='same', bias=True)]
            if has_Ymod:
                self.u = [LocallyConnected2d(feature_out, feature_out, output_size=self.grid_size, kernel_size=kernel_size, stride=1, bias=False), nn.Conv1d(feature_out, feature_out, 3, stride=1, padding='same', bias=False)]
                if upper_ymod:
                    self.umod = [LocallyConnected2d(feature_out, feature_out, output_size=self.grid_size, kernel_size=kernel_size, stride=1, bias=False), nn.Conv1d(feature_out, feature_out, 3, stride=1, padding='same', bias=False)]            

        # Initializing weights
        self.initialise_weights(self.v[0],connections_to_neighbours=False)
        self.initialise_weights(self.v[1], disk=True,connections_to_neighbours=False)
        self.initialise_weights(self.t[0])
        self.initialise_weights(self.t[1], disk=False,marker=True)
        if has_Ymod:
            self.initialise_weights(self.u[0])
            self.initialise_weights(self.u[1], disk=False,marker=True)
            if upper_ymod:
                self.initialise_weights(self.umod[0])
                self.initialise_weights(self.umod[1], disk=False,marker=True)

        # Setting the mask to have connections only between neighboours
        self.vmask = torch.clone(self.v[0].weight.detach())
        self.vmask[self.vmask != 0] = 1  
        self.tmask = torch.clone(self.t[0].weight.detach())
        self.tmask[self.tmask != 0] = 1    

        self.t_disk_mask = torch.clone(self.t[1].weight.detach())
        self.t_disk_mask[self.t_disk_mask != 0] = 1  
        if has_Ymod:
          self.umask = torch.clone(self.u[0].weight.detach())
          self.umask[self.umask != 0] = 1    
          self.u_disk_mask = torch.clone(self.u[1].weight.detach())
          self.u_disk_mask[self.u_disk_mask != 0] = 1  
          
    def register_params(self):
        self.register_parameter(name='v0_weight',param=self.v[0].weight)
        self.register_parameter(name='v0_bias',param=self.v[0].bias)
        
        self.register_parameter(name='v1_weight',param=self.v[1].weight)
        self.register_parameter(name='v1_bias',param=self.v[1].bias)
        
        self.register_parameter(name='t0_weight',param=self.t[0].weight)
        self.register_parameter(name='t0_bias',param=self.t[0].bias)
        
        self.register_parameter(name='t1_weight',param=self.t[1].weight)
        self.register_parameter(name='t1_bias',param=self.t[1].bias)
        
        if self.has_Ymod:
            self.register_parameter(name='u0_weight',param=self.u[0].weight)
            self.register_parameter(name='u0_bias',param=self.u[0].bias)
            
            self.register_parameter(name='u1_weight',param=self.u[1].weight)
            self.register_parameter(name='u1_bias',param=self.u[1].bias)    
            
            if self.upper_ymod:
                self.register_parameter(name='umod0_weight',param=self.umod[0].weight)
                self.register_parameter(name='umod0_bias',param=self.umod[0].bias)
                
                self.register_parameter(name='umod1_weight',param=self.umod[1].weight)
                self.register_parameter(name='umod1_bias',param=self.umod[1].bias)    

    def forward(self, lower_y=None, lower_ymod=None, upper_y=None, upper_ymod=None, det=False):
        if self.has_Ymod:
            current_y = self.activation_function(self.v[0](lower_y[0]))
            current_y_disk = self.activation_function(self.v[1](lower_y[1]))       
        else:
            current_y = self.activation_function(self.v[0](lower_y[0]) + self.t[0](lower_ymod[0]))
            current_y_disk = self.activation_function(self.v[1](lower_y[1]) + self.t[1](lower_ymod[1]))
        if self.has_Ymod:
            if upper_ymod is not None:
                current_ymod = self.activation_function((self.t[0](lower_ymod[0]) + self.umod[0](upper_ymod[0])) * self.gating_function(current_y))
                current_ymod_disk = self.activation_function((self.t[1](lower_ymod[1]) + self.umod[1](upper_ymod[1])) * self.gating_function(current_y_disk))
            else:
                current_ymod = self.activation_function(self.gating_function(current_y) * (self.u[0](upper_y[0])+self.t[0](lower_ymod[0])))
                current_ymod_disk = self.activation_function(self.gating_function(current_y_disk) * (self.u[1](upper_y[1])+self.t[1](lower_ymod[1])))
            return([current_y, current_y_disk], [current_ymod, current_ymod_disk])
        return([current_y, current_y_disk])

    def update_layer(self, upper, z, beta, delta):
        self.v[0] = self.update_weights(self.v[0], upper[0][0], beta, delta, self.vmask, z[0][0])
        self.v[1] = self.update_weights(self.v[1], upper[0][1], beta, delta, z=z[0][1], average=False)
        
        if not self.has_Ymod:
            self.t[0] = self.update_weights(self.t[0], upper[0][0], beta, delta, self.tmask, z[0][0])  
            self.t[1] = self.update_weights(self.t[1], upper[0][1], beta, delta, self.t_disk_mask,z=z[0][1], average=False,average_disk=True)

        if self.has_Ymod:
            self.t[0] = self.update_weights(self.t[0], upper[1][0], beta, delta, self.tmask, z[1][0])  
            self.t[1] = self.update_weights(self.t[1], upper[1][1], beta, delta,self.t_disk_mask, z=z[1][1], average=False,average_disk=True)

            if self.upper_ymod:
                self.umod[0] = self.update_weights(self.umod[0], upper[1][0], beta, delta, self.umask, z[1][0])
                self.umod[1] = self.update_weights(self.umod[1], upper[1][0], beta, delta,self.u_disk_mask, z=z[1][1], average=False,average_disk=True)
            else:
                self.u[0] = self.update_weights(self.u[0], upper[1][0], beta, delta, self.umask, z[1][0])
                self.u[1] = self.update_weights(self.u[1], upper[1][1], beta, delta,self.u_disk_mask, z=z[1][1], average=False,average_disk=True)

    def mask_weights(self):
        with torch.no_grad():
            self.v[0].weight *= self.vmask
            self.t[0].weight *= self.tmask
            self.t[1].weight *= self.t_disk_mask
            
            if self.has_Ymod:
                if self.upper_ymod:
                    self.umod[0].weight *= self.umask
                    self.umod[1].weight *= self.u_disk_mask
                else:
                    self.u[0].weight *= self.umask
                    self.u[1].weight *= self.u_disk_mask


class OutputLayer(CustomLayer):

    def __init__(self, hidden_features, input_features, feature_out,grid_size,weight_sharing):
        super().__init__(grid_size,weight_sharing)
        kernel_size = 2 * self.grid_size - 1
        self.LayerType = 'output'
        self.w = [nn.Conv2d(hidden_features, feature_out, kernel_size, stride=1, padding='same'), nn.Conv1d(hidden_features, feature_out, 3, stride=1, padding='same')]
        self.skip = [nn.Conv2d(input_features, feature_out, 1, stride=1, padding='same', bias=False), nn.Conv1d(input_features, feature_out, 1, stride=1, padding='same', bias=False)]

        self.wmask = (1/14) * torch.ones_like(self.w[0].weight)
        self.wmask[:, :, self.grid_size - 1, self.grid_size - 1] = 1/50

        self.w_disk_mask = torch.clone(self.w[1].weight.detach())
        self.w_disk_mask[self.w_disk_mask != 0] = 1    


        # Initializing weights
        self.initialise_weights(self.w[0])
        self.initialise_weights(self.skip[0], disk=True) #not disk but same structure

        self.initialise_weights(self.w[1], disk=False,marker=True)
        self.initialise_weights(self.skip[1], disk=True)
        
    def register_params(self):
        self.register_parameter(name='w0_weight',param=self.w[0].weight)
        self.register_parameter(name='w0_bias',param=self.w[0].bias)
        
        self.register_parameter(name='w1_weight',param=self.w[1].weight)
        self.register_parameter(name='w1_bias',param=self.w[1].bias)
        
        self.register_parameter(name='skip0_weight',param=self.skip[0].weight)
        self.register_parameter(name='skip0_bias',param=self.skip[0].bias)
        
        self.register_parameter(name='skip1_weight',param=self.skip[1].weight)
        self.register_parameter(name='skip1_bias',param=self.skip[1].bias)
        


    def forward(self, lower_y, inputmod):
        Y = self.w[0](lower_y[0]) + self.skip[0](inputmod[0])
        Y_disk = self.w[1](lower_y[1]) + self.skip[1](inputmod[1])
        return(Y, Y_disk)

    def update_layer(self, upper, beta, delta, winner_disk):
        if winner_disk:
            self.w[1] = self.update_weights(self.w[1], upper, beta, delta,self.w_disk_mask, average=False,average_disk=True)
            self.skip[1] = self.update_weights(self.skip[1], upper, beta, delta, average=False)
        else:
            self.w[0] = self.update_weights(self.w[0], upper, beta, delta, self.wmask)
            self.skip[0] = self.update_weights(self.skip[0], upper, beta, delta, average=False)
            
    def mask_weights(self):
        with torch.no_grad():
            intermmask = torch.ones((2 * self.grid_size - 1, 2 * self.grid_size - 1))
            intermmask[self.grid_size - 1, self.grid_size - 1] = 0
            m = torch.mean(self.w[0].weight[:, :, intermmask == 1], axis=2)
            self.w[0].weight[:, :, intermmask == 1] = m.T
            
class HorizLayer(CustomLayer):

    def __init__(self, features, grid_size):
        super().__init__(grid_size)
        self.features = features
        self.LayerType = 'horizontal'
        self.weight_GridToDisk = torch.nn.Parameter(self.initialisation_range * torch.rand((1,features,2)))  #0.07
        self.weight_DiskToGrid = torch.nn.Parameter(self.initialisation_range * torch.rand((1,features, 2)))
        self.weight_GridToGrid = torch.ones((1, features, grid_size, grid_size))
        self.weight_DiskToDisk =  torch.nn.Parameter(self.initialisation_range * torch.rand((1, features, 2)))

        for i in range(features):
            self.weight_GridToGrid[0, i, :, :] *= self.initialisation_range * np.random.rand() * torch.eye(self.grid_size) 


        self.weight_GridToGrid = torch.nn.Parameter(self.weight_GridToGrid)
        self.weight_DiskToDisk = torch.nn.Parameter(self.weight_DiskToDisk)
        
    def register_params(self):
            self.register_parameter(name='weight_GridToDisk',param=self.weight_GridToDisk)
            self.register_parameter(name='weight_DiskToGrid',param=self.weight_DiskToGrid)
            
            self.register_parameter(name='weight_GridToGrid',param=self.weight_GridToGrid)
            self.register_parameter(name='weight_DiskToDisk',param=self.weight_DiskToDisk)

    def forward(self, Y, Y_disk):

        Update_disk = Y_disk * self.weight_DiskToDisk
        Update_disk = Update_disk + torch.sum(Y, axis=(2, 3))[:,:,None] * self.weight_GridToDisk

        Update_grid = Y * self.weight_GridToGrid
        Update_grid = torch.reshape(Update_grid, [1, self.features, self.grid_size**2])
        Update_grid = Update_grid + torch.sum(Y_disk * self.weight_DiskToGrid, axis=2)[:,:,None]
        Update_grid = torch.reshape(Update_grid, [1, self.features, self.grid_size, self.grid_size])

        return(Update_grid, Update_disk)

    def update_layer(self, upper, z, beta, delta):

        traces_GridToDisk = torch.autograd.grad(upper[0], self.weight_GridToDisk, grad_outputs=z[0], retain_graph=True, allow_unused=True)[0]
        traces_DiskToDisk = torch.autograd.grad(upper[0], self.weight_DiskToDisk, grad_outputs=z[0], retain_graph=True, allow_unused=True)[0]
        traces_DiskToGrid = torch.autograd.grad(upper[1], self.weight_DiskToGrid, grad_outputs=z[1], retain_graph=True, allow_unused=True)[0]
        traces_GridToGrid = torch.autograd.grad(upper[1], self.weight_GridToGrid, grad_outputs=z[1], retain_graph=True, allow_unused=True)[0]

        traces_GridToDisk[:] = torch.mean(traces_GridToDisk, axis=(2), keepdim=True)
        traces_GridToGrid[:] = torch.mean(traces_GridToGrid, axis=(2, 3), keepdim=True) 
        traces_DiskToDisk[:] = torch.mean(traces_DiskToDisk, axis=(2), keepdim=True)
        traces_DiskToGrid[:] = torch.mean(traces_DiskToGrid, axis=(2), keepdim=True)

        for i in range(self.features):
            traces_GridToGrid[0,i,:,:] = traces_GridToGrid[0,i,:,:] * torch.eye(self.grid_size)

        with torch.no_grad():
            weight_GridToDisk_update = self.weight_GridToDisk + beta * delta * traces_GridToDisk
            weight_DiskToGrid_update = self.weight_DiskToGrid + beta * delta * traces_DiskToGrid
            weight_GridToGrid_update = self.weight_GridToGrid + beta * delta * traces_GridToGrid
            weight_DiskToDisk_update = self.weight_DiskToDisk + beta * delta * traces_DiskToDisk

            self.weight_GridToDisk.copy_(weight_GridToDisk_update)
            self.weight_DiskToGrid.copy_(weight_DiskToGrid_update)
            self.weight_GridToGrid.copy_(weight_GridToGrid_update)
            self.weight_DiskToDisk.copy_(weight_DiskToDisk_update)
