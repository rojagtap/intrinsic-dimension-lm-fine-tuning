import torch

from .modeling_base import BaseSubspaceWrapper


class LayerNormSubspaceWrapper(BaseSubspaceWrapper):
    def __init__(self, layer, theta, _lambda=None, layer_index=-1):
        """
        theta will be shared across layers, but the projection matrix P will be unique to layer
        """
        super(LayerNormSubspaceWrapper, self).__init__(layer, theta, _lambda, layer_index)

    def forward(self, x):
        weight, bias = super(LayerNormSubspaceWrapper, self).forward(x)
        return torch.nn.functional.layer_norm(x, self.layer.normalized_shape, weight, bias, self.layer.eps)
