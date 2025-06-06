# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

from setuptools import setup
from torch.utils.cpp_extension import BuildExtension, CUDAExtension

setup(
    name="neural_pbir_cuda_utils",
    ext_modules=[
        CUDAExtension(
            "neural_pbir_cuda_utils",
            [
                "adam_upd.cpp",
                "adam_upd_kernel.cu",
                "render_utils.cpp",
                "render_utils_kernel.cu",
                "total_variation.cpp",
                "total_variation_kernel.cu",
                "grid_sample.cpp",
                "grid_sample_kernel.cu",
                "sdf_rt.cpp",
                "sdf_rt_kernel.cu",
                "binding.cpp",
            ],
            extra_compile_args={"cxx": ["-O2"], "nvcc": ["-O2"]},
        ),
    ],
    cmdclass={"build_ext": BuildExtension},
)
