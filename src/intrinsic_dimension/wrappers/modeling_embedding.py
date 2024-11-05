import torch

from .modeling_base import BaseSubspaceWrapper


class EmbeddingSubspaceWrapper(BaseSubspaceWrapper):
    """
    wrapper for embedding pytorch layer for using low dimensional weight search
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
        super(EmbeddingSubspaceWrapper, self).__init__(layer, theta, _lambda, layer_index, fastfood=fastfood)

    def forward(self, x):
        weight, _ = super(EmbeddingSubspaceWrapper, self).forward(x)
        return torch.nn.functional.embedding(
            x,
            weight,
            self.layer.padding_idx,
            self.layer.max_norm,
            self.layer.norm_type,
            self.layer.scale_grad_by_freq,
            self.layer.sparse,
        )
