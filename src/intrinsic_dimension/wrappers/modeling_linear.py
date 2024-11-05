import torch

from .modeling_base import BaseSubspaceWrapper


class LinearSubspaceWrapper(BaseSubspaceWrapper):
    """
    wrapper for linear pytorch layer for using low dimensional weight search
    here we make the original weights non-trainable and introduce non-trainable
    P (projection vector), and theta (low dim parameter vector) which is trainable
    for each weight entity (in this case, weight and bias).

    size of theta is (dint,)
    size of P is (flat size of weight, dint)

    so P x theta will give a vector of weight size which can be reshaped for addition
    """

    def __init__(self, layer, theta, _lambda=None, layer_index=-1, fastfood=True):
        """
        theta will be shared across layers, but the projection matrix P will be unique to layer
        """
        super(LinearSubspaceWrapper, self).__init__(layer, theta, _lambda, layer_index, fastfood=fastfood)

    def forward(self, x):
        weight, bias = super(LinearSubspaceWrapper, self).forward(x)
        return torch.nn.functional.linear(x, weight, bias)
