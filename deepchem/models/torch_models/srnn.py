import torch
import torch.nn as nn
import numpy as np
from torch.autograd import grad
from typing import Tuple
from deepchem.models.torch_models.layers import MultilayerPerceptron
from deepchem.models.torch_models.torch_model import TorchModel
from deepchem.models.losses import L2Loss

# import leapfrog (once the PR for leapfrog gets merged)


class SRNN(nn.Module):

    def __init__(self, d_input:int =2,
                 d_hidden:Tuple[int, ...] = (128, 128),
                 activation_fn: str = 'tanh',
                 dt:float = 0.1,
                 T: int = 10 ) -> None:
        
        super().__init__()
        self.h_net = MultilayerPerceptron(d_input=d_input,
                                          d_hidden=d_hidden,
                                          d_output=1,
                                          activation_fn=activation_fn)
        
        self.dt = dt 
        self.T = T 
    
    def get_hamiltonian(self, q0, p0):
        x = torch.cat([q0, p0], dim=1)
        H = self.h_net(x)
        return H
    
    def forward(self, z):
        q0 = z[:, 0:1]
        p0 = z[:, 1:2]
        predicted_traj = leapfrog(q0, p0, self.get_hamiltonian, self.dt, self.T, is_hamiltonian=True)
        predicted_traj = predicted_traj.permute(1, 0, 2)
        return predicted_traj


class SRNNModel(TorchModel):

    def __init__(self, d_input:int =2,
                 d_hidden:Tuple[int, ...] = (128, 128),
                 activation_fn: str = 'tanh',
                 dt:float = 0.1,
                 T: int = 10,
                 **kwargs ) -> None:
        model = SRNN(d_input=d_input,
                     d_hidden= d_hidden,
                     activation_fn=activation_fn,
                     dt=dt,
                     T=T)
        super().__init__(model, loss=L2Loss(), **kwargs)