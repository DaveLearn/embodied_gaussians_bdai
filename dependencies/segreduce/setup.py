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

BUILD_CUDA = os.getenv("BUILD_CUDA", "0") == "1"

def get_ext():
    from torch.utils.cpp_extension import BuildExtension
    return BuildExtension


# allow lazy load/compilation of the extension
def get_extensions():
    from torch.utils.cpp_extension import CUDAExtension, BuildExtension
    from pathlib import Path

    cxx_compiler_flags = []

    if os.name == "nt":
        cxx_compiler_flags.append("/wd4624")
    
    SRC_PATH = str(Path(os.path.abspath(__file__)).parent)
    MGPU_PATH = str(Path(SRC_PATH).parent / "moderngpu" / "src")


    return [CUDAExtension(
            name="pysegreduce_cuda.csrc",
            sources=[f"{SRC_PATH}/ext.cu"],
            include_dirs=[SRC_PATH, MGPU_PATH],
            extra_compile_args={"nvcc": ["-O3", "--expt-extended-lambda", "--expt-relaxed-constexpr", "--use_fast_math", "-lineinfo"], 
                                "cxx": cxx_compiler_flags},
        )]

setup(
    name="pysegreduce",
    packages=["pysegreduce"],
    ext_modules=get_extensions() if BUILD_CUDA else [],
    cmdclass={"build_ext": get_ext()} if BUILD_CUDA else {},
)
