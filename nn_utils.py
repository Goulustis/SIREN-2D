import numpy as np
import torch
from torch.nn import functional as F

def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def relu(x):
    return np.maximum(0, x)

class NeuralNet:
    def __init__(self, input_dim, hidden_dims, output_dim, hidden_fnc=relu, out_fnc=lambda x: x):
        self.input_dim = input_dim
        self.hidden_dims = hidden_dims
        self.output_dim = output_dim
        self.num_layers = len(hidden_dims) + 1
        self.num_params = self.calculate_num_params()
        self.hidden_fnc = hidden_fnc
        self.out_fnc = out_fnc

    def calculate_num_params(self):
        num_params = 0
        prev_dim = self.input_dim
        for hidden_dim in self.hidden_dims:
            num_params += prev_dim * hidden_dim + hidden_dim
            prev_dim = hidden_dim
        num_params += prev_dim * self.output_dim + self.output_dim
        return num_params

    def forward_pass(self, theta, X):
        idx = 0
        prev_activation = X
        prev_dim = self.input_dim

        for hidden_dim in self.hidden_dims:
            W = theta[idx: idx + prev_dim * hidden_dim].reshape(prev_dim, hidden_dim)
            idx += prev_dim * hidden_dim
            b = theta[idx: idx + hidden_dim]
            idx += hidden_dim
            prev_activation = self.hidden_fnc(prev_activation @ W + b)
            prev_dim = hidden_dim

        W = theta[idx: idx + prev_dim * self.output_dim].reshape(self.output_dim, prev_dim)
        idx += prev_dim * self.output_dim
        b = theta[idx: idx + self.output_dim]
        y_pred = self.out_fnc(prev_activation @ W + b)
        y_pred = y_pred #.flatten()
        return y_pred
    
    def get_num_params(self):
        return self.num_params
    
    def init_weights(self, num_chains):
        return [np.random.randn(self.num_params).astype(np.float32) for _ in range(num_chains)]


class NeuralNetTorch(NeuralNet):
    def __init__(self, input_dim, hidden_dims, output_dim, hidden_fnc=F.relu, out_fnc=lambda x: x, device="cpu"):
        super().__init__(input_dim, hidden_dims, output_dim, hidden_fnc, out_fnc)
        self.device = device
    
    def forward_pass(self, theta:torch.Tensor, X:torch.Tensor):
        idx = 0
        prev_activation = X
        prev_dim = self.input_dim

        for hidden_dim in self.hidden_dims:
            W = theta[idx: idx + prev_dim * hidden_dim].reshape(hidden_dim, prev_dim)
            idx += prev_dim * hidden_dim
            b = theta[idx: idx + hidden_dim]
            idx += hidden_dim
            prev_dim = hidden_dim
            
            prev_activation = self.hidden_fnc(F.linear(prev_activation, W, b))

        W = theta[idx: idx + prev_dim * self.output_dim].reshape(self.output_dim, prev_dim)
        idx += prev_dim * self.output_dim
        b = theta[idx: idx + self.output_dim]
        y_pred = self.out_fnc(F.linear(prev_activation, W, b))
        y_pred = y_pred.squeeze()
        return y_pred
    
    def init_weights(self, num_chains):
        weights = [torch.from_numpy(e).to(self.device) for e in super().init_weights(num_chains)]
        return weights



class NNObj(NeuralNet):
    def __init__(self, input_dim, hidden_dims, output_dim, hidden_fnc=relu, out_fnc=lambda x: x):
        super().__init__(input_dim, hidden_dims, output_dim, hidden_fnc, out_fnc)
        self.theta = self.init_weights(1)[0]
    
    def forward_pass(self, X):
        theta = self.theta
        return super().forward_pass(theta, X)