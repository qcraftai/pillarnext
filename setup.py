#!/usr/bin/env python
import os
import os.path as osp
import platform
import shutil
import sys
import warnings
from setuptools import find_packages, setup, Extension, dist
from torch.utils.cpp_extension import BuildExtension, CUDAExtension


if __name__ == '__main__':
    setup(
        name='det3d',
        packages=find_packages(exclude=('configs', 'tools', 'demo')),
        include_package_data=True,
        license='Apache License 2.0',
        ext_modules=[
            CUDAExtension(
                name="det3d.core.iou3d_nms.iou3d_nms_cuda",
                sources=[
                        'det3d/core/iou3d_nms/src/iou3d_cpu.cpp',
                        'det3d/core/iou3d_nms/src/iou3d_nms_api.cpp',
                        'det3d/core/iou3d_nms/src/iou3d_nms.cpp',
                        'det3d/core/iou3d_nms/src/iou3d_nms_kernel.cu',
                    ],
                extra_compile_args={'cxx': ['-g', '-I /usr/local/cuda/include'],
                            'nvcc': ['-O2']},
            ),
        ],
        cmdclass={'build_ext': BuildExtension.with_options(no_python_abi_suffix=True)},
        zip_safe=False)

