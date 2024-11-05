# The codes are from Armen Aghajanyan from facebook, from paper
# Intrinsic Dimensionality Explains the Effectiveness of Language Model Fine-Tuning
# https://arxiv.org/abs/2012.13255


import numpy as np
import torch


class FastWalshHadamard(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x):
        ctx.save_for_backward(torch.tensor([1 / np.sqrt(float(x.size(0)))]).to(x))

        if x.is_cuda:
            from .fwh_cuda import fast_walsh_hadamard_transform as fast_walsh_hadamard_transform_cuda
            return fast_walsh_hadamard_transform_cuda(x.float(), False)
        else:
            return FastWalshHadamard.fast_walsh_hadamard_transform_cpu(x.float(), normalize=False)

    @staticmethod
    def backward(ctx, grad_output):
        x, = ctx.saved_tensors

        if grad_output.is_cuda:
            from .fwh_cuda import fast_walsh_hadamard_transform as fast_walsh_hadamard_transform_cuda
            return x * fast_walsh_hadamard_transform_cuda(grad_output.clone().float(), False).to(grad_output)
        else:
            return x * FastWalshHadamard.fast_walsh_hadamard_transform_cpu(grad_output.clone().float(), normalize=False).to(grad_output)

    @staticmethod
    def fast_walsh_hadamard_transform_cpu(x, axis=0, normalize=False):
        orig_shape = x.size()
        assert 0 <= axis < len(orig_shape), (
                "For a vector of shape %s, axis must be in [0, %d] but it is %d"
                % (orig_shape, len(orig_shape) - 1, axis)
        )
        H_dim = orig_shape[axis]
        H_dim_exp = int(round(np.log(H_dim) / np.log(2)))
        assert H_dim == 2 ** H_dim_exp, (
                "hadamard can only be computed over axis with size that is a power of two, but"
                " chosen axis %d has size %d" % (axis, H_dim)
        )

        working_shape_pre = [int(torch.prod(torch.tensor(orig_shape[:axis])))]
        working_shape_post = [
            int(torch.prod(torch.tensor(orig_shape[axis + 1:])))
        ]
        working_shape_mid = [2] * H_dim_exp
        working_shape = working_shape_pre + working_shape_mid + working_shape_post

        result = x.view(working_shape)
        for i in range(H_dim_exp):
            dim = i + 1
            chunked = torch.chunk(result, 2, dim=dim)
            assert len(chunked) == 2
            result = torch.cat((chunked[0] + chunked[1], chunked[0] - chunked[1]), axis=dim)

        if normalize:
            result = result / np.sqrt(float(H_dim))

        return result.view(orig_shape)


class FastFoodProjection(torch.nn.Module):
    def __init__(self, flat_weight_dim):
        super(FastFoodProjection, self).__init__()

        self.flat_weight_dim = flat_weight_dim

        self.size = 2 ** int(np.ceil(np.log(self.flat_weight_dim) / np.log(2)))

        # Gaussian scaling matrix, whose elements $G_{i,i} \sim \mathcal{N}(0, 1)$
        self.G = torch.nn.Parameter(torch.FloatTensor(self.size, ).normal_())

        # Random permutation matrix
        self.Pi = torch.nn.Parameter(torch.LongTensor(np.random.permutation(self.size)))

        # Binary scaling matrix where $B_{i,i} \in \{\pm 1 \}$ drawn iid
        self.B = 2 * torch.nn.Parameter(torch.FloatTensor(self.size).uniform_(0, 2).type(torch.LongTensor) - 1)

        self.divisor = torch.sqrt(self.size * torch.sum(torch.pow(self.G, 2)))

        self.reset_parameters()

    def reset_parameters(self):
        self.B.requires_grad_(False)
        self.G.requires_grad_(False)
        self.Pi.requires_grad_(False)
        self.divisor.requires_grad_(False)

    def forward(self, theta):
        # Fastfood transform

        # Pad x if needed
        theta_padded = torch.nn.functional.pad(theta, pad=(0, self.size - theta.size(0)), value=0.0, mode="constant")

        # From left to right HGPiH(BX), where H is Walsh-Hadamard matrix
        theta_padded = theta_padded * self.B

        # HGPi(HBX)
        HBX = FastWalshHadamard.apply(theta_padded)

        # HG(PiHBX)
        PiHBX = HBX[self.Pi]

        # H(GPiHBX)
        GPiHBX = PiHBX * self.G

        # (HGPiHBX)
        HGPiHBX = FastWalshHadamard.apply(GPiHBX)

        result = HGPiHBX[:int(self.flat_weight_dim)]
        result = result / (self.divisor * np.sqrt(float(self.flat_weight_dim) / self.size))

        return result
