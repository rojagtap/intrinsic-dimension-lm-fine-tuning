import torch
from setuptools import setup
from torch.utils.cpp_extension import BuildExtension, CUDAExtension, CppExtension

# Check if CUDA is available
use_cuda = torch.cuda.is_available()

extension = CUDAExtension if use_cuda else CppExtension

sources = ['src/intrinsic_dimension/wrappers/projections/fwh_cuda/fast_walsh_hadamard.cpp']
if use_cuda:
    sources.append('src/intrinsic_dimension/wrappers/projections/fwh_cuda/fast_walsh_hadamard.cu')

setup(
    name='fast_walsh_hadamard',
    ext_modules=[
        extension('fast_walsh_hadamard', sources)
    ],
    cmdclass={'build_ext': BuildExtension}
)
