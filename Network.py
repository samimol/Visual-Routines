# -*- coding: utf-8 -*-
"""
Created on Fri Oct 29 12:37:14 2021

@author: Sami
"""

from Layers import *


class Network():

    def __init__(self, n_input_features):
        self.beta = 0.02
        self.n_input_features = n_input_features
        self.n_hidden_features = 4

        self.controller = 'max-boltzmann'
        self.exploit_prob = 0.95

        self.grid_size = 7

        self.input_layer = InputLayer(self.n_input_features, 4)
        self.hidden_layer_1 = HiddenLayer(self.n_input_features, 4, has_Ymod=True)
        self.hidden_layer_2 = HiddenLayer(self.n_hidden_features, self.n_hidden_features, has_Ymod=False)
        self.output_layer = OutputLayer(self.n_hidden_features, self.n_input_features, 1)
        self.horizontal = HorizLayer(self.n_input_features, self.grid_size)

        self.dosave = False
        self.saveX = []
        self.saveXmod = []
        self.saveY1 = []
        self.saveY1mod = []
        self.saveY2 = []
        self.saveQ = []

        self.saveX_disk = []
        self.saveXmod_disk = []
        self.saveY1_disk = []
        self.saveY1mod_disk = []
        self.saveY2_disk = []

    def doStep(self, input_env, reward, reset_traces, device):

        if self.dosave:
            self.saveX.append([])
            self.saveXmod.append([])
            self.saveY1.append([])
            self.saveY1mod.append([])
            self.saveY2.append([])
            self.saveQ.append([])

            self.saveX_disk.append([])
            self.saveXmod_disk.append([])
            self.saveY1_disk.append([])
            self.saveY1mod_disk.append([])
            self.saveY2_disk.append([])

        self.Xmod = torch.zeros(1, self.n_input_features, self.grid_size, self.grid_size, device=device)
        self.Xmod_disk = torch.zeros(1, self.n_input_features, 1, device=device)
        self.Y1 = torch.zeros(1, self.n_hidden_features, self.grid_size, self.grid_size, device=device)
        self.Y1_disk = torch.zeros(1, self.n_hidden_features, 1, device=device)
        self.Y1mod = torch.zeros(1, self.n_hidden_features, self.grid_size, self.grid_size, device=device)
        self.Y1mod_disk = torch.zeros(1, self.n_hidden_features, 1, device=device)
        self.Y2 = torch.zeros(1, self.n_hidden_features, self.grid_size, self.grid_size, device=device)
        self.Y2_disk = torch.zeros(1, self.n_hidden_features, 1, device=device)

        i = 0
        norm = 10

        while i < 40 and norm > 0:

            prevY1 = self.Y1.detach()

            [XmodHoriz, Xmod_diskHoriz] = self.horizontal.forward(self.Xmod, self.Xmod_disk)
            ([self.X, self.X_disk], [self.Xmod, self.Xmod_disk]) = self.input_layer.forward([self.Y1, self.Y1_disk], [self.Y1mod, self.Y1mod_disk], input_env)
            self.Xmod = self.input_layer.MyRelu((self.Xmod + XmodHoriz) * self.X)
            self.Xmod_disk = self.input_layer.MyRelu((self.Xmod_disk + Xmod_diskHoriz) * self.X_disk)
            ([self.Y1, self.Y1_disk], [self.Y1mod, self.Y1mod_disk]) = self.hidden_layer_1.forward(lower_y=[self.X, self.X_disk], lower_ymod=[self.Xmod, self.Xmod_disk], upper_y=[self.Y2, self.Y2_disk])
            [self.Y2, self.Y2_disk] = self.hidden_layer_2.forward(lower_y=[self.Y1, self.Y1_disk], lower_ymod=[self.Y1mod, self.Y1mod_disk])

            if self.dosave:
                self.saveX[len(self.saveX) - 1].append(self.X.detach())
                self.saveXmod[len(self.saveXmod) - 1].append(self.Xmod.detach())
                self.saveY1[len(self.saveY1) - 1].append(self.Y1.detach())
                self.saveY1mod[len(self.saveY1mod) - 1].append(self.Y1mod.detach())
                self.saveY2[len(self.saveY2) - 1].append(self.Y2.detach())

                self.saveX_disk[len(self.saveX_disk) - 1].append(self.X_disk.detach())
                self.saveXmod_disk[len(self.saveXmod_disk) - 1].append(self.Xmod_disk.detach())
                self.saveY1_disk[len(self.saveY1_disk) - 1].append(self.Y1_disk.detach())
                self.saveY1mod_disk[len(self.saveY1mod_disk) - 1].append(self.Y1mod_disk.detach())
                self.saveY2_disk[len(self.saveY2_disk) - 1].append(self.Y2_disk.detach())

            with torch.no_grad():
                norm = torch.linalg.norm(self.Y1[0, -1, :]-prevY1[0, -1, :], ord=float('inf'))

            i += 1

        ([self.Xmodtest, self.Y1test, self.Y1modtest, self.Y2test], [self.Xmod, self.Y1, self.Y1mod, self.Y2]) = self.detachAndreattach([self.Xmod, self.Y1, self.Y1mod, self.Y2])
        ([self.Xmodtest_disk, self.Y1test_disk, self.Y1modtest_disk, self.Y2test_disk], [self.Xmod_disk, self.Y1_disk,
                                                                                         self.Y1mod_disk, self.Y2_disk]) = self.detachAndreattach([self.Xmod_disk, self.Y1_disk, self.Y1mod_disk, self.Y2_disk])

        [XmodHoriz, Xmod_diskHoriz] = self.horizontal.forward(self.Xmod, self.Xmod_disk)
        ([self.X, self.X_disk], [self.Xmod, self.Xmod_disk]) = self.input_layer.forward([self.Y1, self.Y1_disk], [self.Y1mod, self.Y1mod_disk], input_env)
        self.Xmod = self.input_layer.MyRelu((self.Xmod + XmodHoriz) * self.X)
        self.Xmod_disk = self.input_layer.MyRelu((self.Xmod_disk + Xmod_diskHoriz) * self.X_disk)
        ([self.Y1, self.Y1_disk], [self.Y1mod, self.Y1mod_disk]) = self.hidden_layer_1.forward(lower_y=[self.X, self.X_disk], lower_ymod=[self.Xmod, self.Xmod_disk], upper_y=[self.Y2, self.Y2_disk])
        [self.Y2, self.Y2_disk] = self.hidden_layer_2.forward(lower_y=[self.Y1, self.Y1_disk], lower_ymod=[self.Y1mod, self.Y1mod_disk])

        self.Z, self.Z_disk = self.calc_Output(device)

        if self.dosave:
            self.saveQ[len(self.saveQ) - 1].append(self.Z.detach())

        with torch.no_grad():
            action_chosen = torch.zeros((1, 1, 2+self.grid_size**2))
            action_chosen[0, 0, self.index_selected] = 1

        return (self.action)

    def calc_Output(self, device):
        Z, Z_disk = self.output_layer.forward([self.Y2, self.Y2_disk], [self.Xmod, self.Xmod_disk])

        with torch.no_grad():
            ZZ = torch.cat((Z_disk, torch.flatten(Z, start_dim=2)), dim=2)

            if self.controller == 'max-boltzmann':
                if np.random.rand() < self.exploit_prob:
                    winner = self.calc_maxQ(ZZ)
                else:
                    ZZint = ZZ.detach()
                    ZZint -= torch.max(ZZint)
                    ZZint = torch.exp(ZZint) / torch.sum(torch.exp(ZZint))
                    winner = self.calc_softWTA(ZZint, device)
            else:
                raise Exception("Wrong controller")
            self.index_selected = winner
            if winner < 2:  # disk
                winner = [torch.tensor([0]), torch.tensor([0]), torch.tensor([winner])]
                self.winner_disk = True
            else:  # grid
                winner = [torch.tensor([0]), torch.tensor([0]), torch.tensor([(winner-2)//self.grid_size]), torch.tensor([(winner-2) % self.grid_size])]
                self.winner_disk = False

            #winner = [torch.tensor([0]),torch.tensor([0]),torch.tensor([2]),torch.tensor([2])]
            self.action = winner

        return Z, Z_disk

    def calc_maxQ(self, Z):
        winner = torch.where(Z == torch.max(Z))[-1]
        # Break ties randomly
        if len(winner) > 1:
            tiebreak = torch.randint(0, len(winner), (1,))
            winner = tiebreak
        return winner

    def calc_softWTA(self, probabilities, device):
        # Create wheel:
        probs = torch.cumsum(probabilities, 2)[0][0]

        # Select from wheel
        rnd = torch.rand((1,), device=device)
        for (i, prob) in enumerate(probs):
            if rnd <= prob:
                return i

    def Accessory(self):

        if self.winner_disk:
            init = [torch.zeros_like(self.Y2), torch.zeros_like(self.Y1mod), torch.zeros_like(self.Y1), torch.zeros_like(self.Xmod)]
            init_disk = torch.autograd.grad(self.Z_disk[self.action], [self.Y2_disk, self.Y1mod_disk, self.Y1_disk, self.Xmod_disk], retain_graph=True, allow_unused=True)
        else:
            init = torch.autograd.grad(self.Z[self.action], [self.Y2, self.Y1mod, self.Y1, self.Xmod], retain_graph=True, allow_unused=True)
            init_disk = [torch.zeros_like(self.Y2_disk), torch.zeros_like(self.Y1mod_disk), torch.zeros_like(self.Y1_disk), torch.zeros_like(self.Xmod_disk)]

        Zy2 = init[0]
        Zy1mod = init[1]
        Zy1 = init[2]
        Zxmod = init[3]

        Zy2_disk = init_disk[0]
        Zy1mod_disk = init_disk[1]
        Zy1_disk = init_disk[2]
        Zxmod_disk = init_disk[3]

        for i in range(7):
            ZXmod_prev = Zxmod
            Zxmod_disk_prev = Zxmod_disk

            Zy2 = torch.autograd.grad(self.Y1mod, self.Y2test, grad_outputs=Zy1mod, retain_graph=True, allow_unused=True)[0]
            Zy2 = Zy2 + init[0]

            Zy1mod = torch.autograd.grad(self.Y2, self.Y1mod, grad_outputs=Zy2, retain_graph=True, allow_unused=True)[0]
            Zy1mod = Zy1mod + torch.autograd.grad(self.Xmod, self.Y1modtest, grad_outputs=Zxmod, retain_graph=True, allow_unused=True)[0]
            Zy1mod = Zy1mod + init[1]

            Zy1 = torch.autograd.grad(self.Y2, self.Y1, grad_outputs=Zy2, retain_graph=True, allow_unused=True)[0]
            Zy1 = Zy1 + torch.autograd.grad(self.Xmod, self.Y1test, grad_outputs=Zxmod, retain_graph=True, allow_unused=True)[0]
            Zy1 = Zy1 + init[2]

            Zxmod = torch.autograd.grad(self.Y1, self.Xmod, grad_outputs=Zy1, retain_graph=True, allow_unused=True)[0]
            Zxmod = Zxmod + torch.autograd.grad(self.Xmod, self.Xmodtest, grad_outputs=ZXmod_prev, retain_graph=True, allow_unused=True)[0]
            Zxmod = Zxmod + torch.autograd.grad(self.Xmod_disk, self.Xmodtest, grad_outputs=Zxmod_disk_prev, retain_graph=True, allow_unused=True)[0]
            Zxmod = Zxmod + init[3]

            Zy2_disk = torch.autograd.grad(self.Y1mod_disk, self.Y2test_disk, grad_outputs=Zy1mod_disk, retain_graph=True, allow_unused=True)[0]
            Zy2_disk = Zy2_disk + init_disk[0]

            Zy1mod_disk = torch.autograd.grad(self.Y2_disk, self.Y1mod_disk, grad_outputs=Zy2_disk, retain_graph=True, allow_unused=True)[0]
            Zy1mod_disk = Zy1mod_disk + torch.autograd.grad(self.Xmod_disk, self.Y1modtest_disk, grad_outputs=Zxmod_disk, retain_graph=True, allow_unused=True)[0]
            Zy1mod_disk = Zy1mod_disk + init_disk[1]

            Zy1_disk = torch.autograd.grad(self.Y2_disk, self.Y1_disk, grad_outputs=Zy2_disk, retain_graph=True, allow_unused=True)[0]
            Zy1_disk = Zy1_disk + torch.autograd.grad(self.Xmod_disk, self.Y1test_disk, grad_outputs=Zxmod_disk, retain_graph=True, allow_unused=True)[0]
            Zy1_disk = Zy1_disk + init_disk[2]

            Zxmod_disk = Zxmod_disk + torch.autograd.grad(self.Y1_disk, self.Xmod_disk, grad_outputs=Zy1_disk, retain_graph=True, allow_unused=True)[0]
            Zxmod_disk = Zxmod_disk + torch.autograd.grad(self.Xmod_disk, self.Xmodtest_disk, grad_outputs=Zxmod_disk_prev, retain_graph=True, allow_unused=True)[0]
            Zxmod_disk = Zxmod_disk + torch.autograd.grad(self.Xmod, self.Xmodtest_disk, grad_outputs=ZXmod_prev, retain_graph=True, allow_unused=True)[0]
            Zxmod_disk = Zxmod_disk + init_disk[3]

        return (Zxmod, Zy1, Zy1mod, Zy2, Zxmod_disk, Zy1_disk, Zy1mod_disk, Zy2_disk)

    def doLearn(self, reward):

        with torch.no_grad():
            if self.winner_disk:
                exp_value = self.Z_disk[self.action]
            else:
                exp_value = self.Z[self.action]
            self.delta = reward - exp_value

        (Zxmod, Zy1, Zy1mod, Zy2, Zxmod_disk, Zy1_disk, Zy1mod_disk, Zy2_disk) = self.Accessory()
        self.input_layer.UpdateLayer([self.Xmod, self.Xmod_disk], [Zxmod, Zxmod_disk], self.beta, self.delta)
        self.horizontal.UpdateLayer([self.Xmod_disk, self.Xmod], [Zxmod_disk, Zxmod], self.beta, self.delta)
        self.hidden_layer_1.UpdateLayer([[self.Y1, self.Y1_disk], [self.Y1mod, self.Y1mod_disk]], [[Zy1, Zy1_disk], [Zy1mod, Zy1mod_disk]], self.beta, self.delta)
        self.hidden_layer_2.UpdateLayer([[self.Y2, self.Y2_disk]], [[Zy2, Zy2_disk]], self.beta, self.delta)
        if self.winner_disk:
            self.output_layer.UpdateLayer(self.Z_disk[self.action], self.beta, self.delta, self.winner_disk)
        else:
            self.output_layer.UpdateLayer(self.Z[self.action], self.beta, self.delta, self.winner_disk)

    def detachAndreattach(self, x):
        detached = [xx.detach() for xx in x]
        for i, xx in enumerate(detached):
            xx.requires_grad = True
            x[i] = xx
        return(x, detached)

    def to(self, device):
        self.input_layer.toCustom(device)
        self.hidden_layer_1.toCustom(device)
        self.hidden_layer_2.toCustom(device)
        self.output_layer.toCustom(device)

        self.horizontal.weight_GridToDisk = self.horizontal.weight_GridToDisk.to(device)
        self.horizontal.weight_DiskToGrid = self.horizontal.weight_DiskToGrid.to(device)
        self.horizontal.weight_GridToGrid = self.horizontal.weight_GridToGrid.to(device)
        self.horizontal.weight_DiskToDisk = self.horizontal.weight_DiskToDisk.to(device)
