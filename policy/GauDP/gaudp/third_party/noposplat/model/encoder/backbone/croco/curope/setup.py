# Copyright (C) 2022-present Naver Corporation. All rights reserved.
# Licensed under CC BY-NC-SA 4.0 (non-commercial use only).

import os

from setuptools import setup
from torch import cuda
from torch.utils.cpp_extension import BuildExtension, CUDAExtension

# Respect the architecture list selected by the installation environment.
arch_list = os.environ.get('TORCH_CUDA_ARCH_LIST')
if arch_list:
    all_cuda_archs = []
    for arch in arch_list.replace(';', ' ').split():
        with_ptx = arch.endswith('+PTX')
        arch = arch.removesuffix('+PTX').replace('.', '')
        all_cuda_archs += ['-gencode', f'arch=compute_{arch},code=sm_{arch}']
        if with_ptx:
            all_cuda_archs += ['-gencode', f'arch=compute_{arch},code=compute_{arch}']
else:
    all_cuda_archs = cuda.get_gencode_flags().replace('compute=', 'arch=').split()
# alternatively, you can list cuda archs that you want, eg:
# all_cuda_archs = [
    # '-gencode', 'arch=compute_70,code=sm_70',
    # '-gencode', 'arch=compute_75,code=sm_75',
    # '-gencode', 'arch=compute_80,code=sm_80',
    # '-gencode', 'arch=compute_86,code=sm_86'
# ]

setup(
    name = 'curope',
    ext_modules = [
        CUDAExtension(
                name='curope',
                sources=[
                    "curope.cpp",
                    "kernels.cu",
                ],
                extra_compile_args = dict(
                    nvcc=['-O3','--ptxas-options=-v',"--use_fast_math"]+all_cuda_archs, 
                    cxx=['-O3'])
                )
    ],
    cmdclass = {
        'build_ext': BuildExtension
    })
