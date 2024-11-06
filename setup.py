import setuptools
from torch.utils.cpp_extension import BuildExtension, CUDAExtension


def setup_package():
    setuptools.setup(
        name='intrinsic-dimension-lm-fine-tuning',
        version='0.0.1',
        long_description="intrinsic-dimension-lm-fine-tuning",
        long_description_content_type='text/markdown',
        author='Rohan Jagtap',
        license='MIT License',
        packages=setuptools.find_packages(exclude=['docs', 'tests', 'scripts', 'examples']),
        dependency_links=['https://download.pytorch.org/whl/torch_stable.html'],
        classifiers=[
            'Intended Audience :: Developers',
            'Intended Audience :: Science/Research',
            'License :: OSI Approved :: MIT License',
            'Topic :: Scientific/Engineering :: Artificial Intelligence',
            'Programming Language :: Python :: 3',
            'Programming Language :: Python :: 3.10.12',
        ],
        keywords='text nlp machinelearning',
        ext_modules=[
            CUDAExtension(
                name='src.intrinsic_dimension.wrappers.projections.fwh_cuda',
                sources=[
                    'src/intrinsic_dimension/wrappers/projections/fwh_cuda/fast_walsh_hadamard.cpp',
                    'src/intrinsic_dimension/wrappers/projections/fwh_cuda/fast_walsh_hadamard.cu',
                ]
            )
        ],
        cmdclass={"build_ext": BuildExtension},
        install_requires=[
            'datasets==3.1.0',
            'torchvision==0.20.1',
            'numpy==2.1.2',
            'matplotlib==3.9.2',
            'torchviz==0.0.2',
            'evaluate==0.4.3',
            'torch==2.5.1',
            'transformers==4.46.1',
            'setuptools==75.1.0'
        ],
    )


if __name__ == '__main__':
    setup_package()
