import numpy as np
import torch


class BaseSubspaceWrapper(torch.nn.Module):
    def __init__(self, layer, theta, _lambda=None, layer_index=-1):
        super(BaseSubspaceWrapper, self).__init__()
        self.layer = layer
        self.theta = theta
        self._lambda = _lambda
        self.layer_index = layer_index

        # intrinsic dimension
        dint = theta.size(0)

        # P (projection matrix) here is basically the flat size of the weight vector (or bias vector) x dint
        self.P_weight = torch.nn.Parameter(torch.empty(np.prod(self.layer.weight.size()), dint))
        if hasattr(self.layer, "bias") and self.layer.bias is not None:
            self.P_bias = torch.nn.Parameter(torch.empty(np.prod(self.layer.bias.size()), dint))
        self.reset_parameters()

    def reset_parameters(self):
        self.P_weight.requires_grad_(False)
        if hasattr(self.layer, "bias") and self.layer.bias is not None:
            self.P_bias.requires_grad_(False)

        for parameter in self.layer.parameters():
            parameter.requires_grad_(False)

        torch.nn.init.normal_(self.P_weight)
        if hasattr(self.layer, "bias") and self.layer.bias is not None:
            torch.nn.init.normal_(self.P_bias)

    def forward(self, x):
        delta_theta_weight = self.P_weight @ self.theta

        if self._lambda:
            # structure-aware intrinsic dimension
            delta_theta_weight = delta_theta_weight * self._lambda[self.layer_index]

        weight = self.layer.weight + delta_theta_weight.view(self.layer.weight.size())

        bias = None
        if hasattr(self.layer, "bias") and self.layer.bias is not None:
            delta_theta_bias = self.P_bias @ self.theta
            bias = self.layer.bias + delta_theta_bias.view(self.layer.bias.size())

        return weight, bias
