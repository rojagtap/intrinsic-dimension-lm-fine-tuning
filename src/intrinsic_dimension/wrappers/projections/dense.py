import torch


class DenseRandomProjection(torch.nn.Module):
    def __init__(self, dint, flat_weight_dim):
        super(DenseRandomProjection, self).__init__()

        # P (projection matrix) here is basically the flat size of the weight vector (or bias vector) x dint
        self.P = torch.nn.Parameter(torch.empty(flat_weight_dim, dint))

        self.reset_parameters()

    def reset_parameters(self):
        self.P.requires_grad_(False)
        torch.nn.init.normal_(self.P)

    def forward(self, theta):
        return self.P @ theta
