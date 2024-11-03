import torch

from ..util.constants import LAYER_MAP


class TransformerSubspaceWrapper(torch.nn.Module):
    """
    wrapper class for transformer model
    """

    def __init__(self, base_model, dint, said=False):
        if not isinstance(base_model, torch.nn.Module):
            raise ValueError(f"base_model expected to be of the type torch.nn.Module, got {type(base_model)}")

        super(TransformerSubspaceWrapper, self).__init__()

        self.base_model = base_model
        self.num_trainable_layers = sum(self.__is_wrappable_layer(module) for module in self.base_model.modules())

        self._lambda = None
        if said:
            assert dint > self.num_trainable_layers, f"intrinsic dimension {dint} must be greater than number of trainable layers {self.num_trainable_layers}"

            # using size d - m theta and size m lambda
            dint -= self.num_trainable_layers
            self._lambda = torch.nn.Parameter(torch.ones(self.num_trainable_layers))

        # theta is shared across all layers
        self.theta = torch.nn.Parameter(torch.zeros(dint))
        self.reset_parameters()

        # map layer to the corresponding wrapper layer if present
        layer_index = -1
        layers_to_wrap = []
        for name, module in self.base_model.named_modules():
            if self.__is_wrappable_layer(module):
                layer_index += 1
                layers_to_wrap.append((name, module, layer_index))

        for name, module, layer_index in layers_to_wrap:
            wrapper = LAYER_MAP[type(module)](layer=module, theta=self.theta, _lambda=self._lambda, layer_index=layer_index)
            setattr(self.base_model, name, wrapper)

    @staticmethod
    def __is_wrappable_layer(layer: torch.nn.Module):
        # layers with wrappers available and having any parameters trainable are eligible for wrapping
        return isinstance(layer, tuple(LAYER_MAP.keys())) and any(param.requires_grad for param in layer.parameters())

    def reset_parameters(self):
        torch.nn.init.zeros_(self.theta)
        if self._lambda:
            torch.nn.init.ones_(self._lambda)

    def forward(self, x):
        return self.base_model(x)
