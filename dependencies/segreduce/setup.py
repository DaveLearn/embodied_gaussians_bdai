#
# Copyright (C) 2023, Inria
# GRAPHDECO research group, https://team.inria.fr/graphdeco
# All rights reserved.
#
# This software is free for non-commercial, research and evaluation use
# under the terms of the LICENSE.md file.
#
# For inquiries contact  george.drettakis@inria.fr
#

from setuptools import setup
import os


def get_ext():
    from torch.utils.cpp_extension import BuildExtension
    return BuildExtension


# allow lazy load/compilation of the extension
def get_extensions():
    from torch.utils.cpp_extension import CUDAExtension
    from pathlib import Path

    cxx_compiler_flags = []

    if os.name == "nt":
        cxx_compiler_flags.append("/wd4624")
    
    SRC_PATH = str(Path(os.path.abspath(__file__)).parent)

    print(f"Compiling {SRC_PATH}/ext.cu")

    return [CUDAExtension(
            name="pysegreduce.cuda",
            sources=[f"pysegreduce/cuda/ext.cu"],
            include_dirs=["pysegreduce/cuda"],
            extra_compile_args={"nvcc": ["-O3", "--expt-extended-lambda", "--expt-relaxed-constexpr", "--use_fast_math", "-lineinfo"], 
                                "cxx": cxx_compiler_flags},
        )]

setup(
    name="pysegreduce",
    packages=["pysegreduce"],
    ext_modules=get_extensions(),
    cmdclass={"build_ext": get_ext()},
    include_package_data=True,
)
