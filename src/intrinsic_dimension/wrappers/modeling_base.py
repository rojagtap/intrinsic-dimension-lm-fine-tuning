import numpy as np
import torch

from .projections.dense import DenseRandomProjection
from .projections.fastfood import FastFoodProjection


class BaseSubspaceWrapper(torch.nn.Module):
    def __init__(self, layer, theta, _lambda=None, layer_index=-1, fastfood=True):
        super(BaseSubspaceWrapper, self).__init__()
        self.layer = layer
        self.theta = theta
        self._lambda = _lambda
        self.layer_index = layer_index

        # intrinsic dimension
        dint = theta.size(0)

        # flat weights
        flat_weight_dim = np.prod(self.layer.weight.size())
        flat_bias_dim = 0
        if self.__use_bias():
            flat_bias_dim = np.prod(self.layer.bias.size())

        if fastfood:
            self.P_weight = FastFoodProjection(flat_weight_dim)
            if self.__use_bias():
                self.P_bias = FastFoodProjection(flat_bias_dim)
        else:
            # will crash for most transformer based architectures
            self.P_weight = DenseRandomProjection(dint, flat_weight_dim)
            if self.__use_bias():
                self.P_bias = DenseRandomProjection(dint, flat_bias_dim)

        self.reset_parameters()

    def __use_bias(self):
        return hasattr(self.layer, "bias") and self.layer.bias is not None

    def reset_parameters(self):
        for parameter in self.layer.parameters():
            parameter.requires_grad_(False)

    def forward(self, x):
        delta_theta_weight = self.P_weight(self.theta)

        if self._lambda is not None:
            # structure-aware intrinsic dimension
            delta_theta_weight = delta_theta_weight * self._lambda[self.layer_index]

        weight = self.layer.weight + delta_theta_weight.view(self.layer.weight.size())

        bias = None
        if self.__use_bias():
            delta_theta_bias = self.P_bias(self.theta)

            if self._lambda is not None:
                # structure-aware intrinsic dimension
                delta_theta_bias = delta_theta_bias * self._lambda[self.layer_index]

            bias = self.layer.bias + delta_theta_bias.view(self.layer.bias.size())

        return weight, bias
