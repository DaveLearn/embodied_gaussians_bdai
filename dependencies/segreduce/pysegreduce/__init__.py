# with many thanks to: https://github.com/nerfstudio-project/gsplat/blob/main/gsplat/cuda/_backend.py

import json
import os
import shutil
from subprocess import DEVNULL, call
from pathlib import Path
from rich.console import Console
import torch
from torch.utils.cpp_extension import (
    _get_build_directory,
    _import_module_from_library,
    load,
)


MAX_JOBS = os.getenv("MAX_JOBS")
need_to_unset_max_jobs = False
if not MAX_JOBS:
    need_to_unset_max_jobs = True
    # set MAX_JOBS to num of processors/2 with max of 10 and min of 1
    os.environ["MAX_JOBS"] = str(max(1, min(os.cpu_count() // 2, 10)))



def update_arch_list():
    """Get the list of compute capabilities to compile for."""
    compute_cap = os.getenv("TORCH_CUDA_ARCH_LIST")
    if compute_cap is not None:
        return
    # Get the compute capability of the current GPU
    (maj, min) = torch.cuda.get_device_capability()
    os.environ["TORCH_CUDA_ARCH_LIST"] = f"{maj}.{min}"
     

def load_extension(
    name,
    sources,
    extra_cflags=None,
    extra_cuda_cflags=None,
    extra_ldflags=None,
    extra_include_paths=None,
    build_directory=None,
):
    update_arch_list()
    
    """Load a JIT compiled extension."""
    # Make sure the build directory exists.
    if build_directory:
        os.makedirs(build_directory, exist_ok=True)

    # If the JIT build happens concurrently in multiple processes,
    # race conditions can occur when removing the lock file at:
    # https://github.com/pytorch/pytorch/blob/e3513fb2af7951ddf725d8c5b6f6d962a053c9da/torch/utils/cpp_extension.py#L1736
    # But it's ok so we catch this exception and ignore it.
    try:
        return load(
            name,
            sources,
            extra_cflags=extra_cflags,
            extra_cuda_cflags=extra_cuda_cflags,
            extra_ldflags=extra_ldflags,
            extra_include_paths=extra_include_paths,
            build_directory=build_directory,
        )
    except OSError:
        # The module should be already compiled
        return _import_module_from_library(name, build_directory, True)


def cuda_toolkit_available():
    """Check if the nvcc is avaiable on the machine."""
    try:
        call(["nvcc"], stdout=DEVNULL, stderr=DEVNULL)
        return True
    except FileNotFoundError:
        return False


def cuda_toolkit_version():
    """Get the cuda toolkit version."""
    cuda_home = os.path.join(os.path.dirname(shutil.which("nvcc")), "..")
    if os.path.exists(os.path.join(cuda_home, "version.txt")):
        with open(os.path.join(cuda_home, "version.txt")) as f:
            cuda_version = f.read().strip().split()[-1]
    elif os.path.exists(os.path.join(cuda_home, "version.json")):
        with open(os.path.join(cuda_home, "version.json")) as f:
            cuda_version = json.load(f)["cuda"]["version"]
    else:
        raise RuntimeError("Cannot find the cuda version.")
    return cuda_version


def load_jit_cuda_extension(name, sources, extra_include_paths, extra_cflags, extra_cuda_cflags):
    """Load a JIT compiled CUDA extension."""
    # If the JIT build happens concurrently in multiple processes
    if cuda_toolkit_available():
        build_dir = _get_build_directory(name, verbose=False)

        # If JIT is interrupted it might leave a lock in the build directory.
        # We dont want it to exist in any case.
        try:
            os.remove(os.path.join(build_dir, "lock"))
        except OSError:
            pass

        if os.path.exists(os.path.join(build_dir, f"{name}.so")) or os.path.exists(
            os.path.join(build_dir, "{name}.lib")
        ):
            # If the build exists, we assume the extension has been built
            # and we can load it.
            return load_extension(
                name=name,
                sources=sources,
                extra_cflags=extra_cflags,
                extra_cuda_cflags=extra_cuda_cflags,
                extra_include_paths=extra_include_paths,
                build_directory=build_dir,
            )
        else:
            # Build from scratch. Remove the build directory just to be safe: pytorch jit might stuck
            # if the build directory exists with a lock file in it.
            shutil.rmtree(build_dir)
            with Console().status(
                f"[bold yellow]{name}: Building CUDA extension to {build_dir} with MAX_JOBS={os.environ['MAX_JOBS']} (This may take a few minutes the first time)",
                spinner="bouncingBall",
            ):
                return load_extension(
                    name=name,
                    sources=sources,
                    extra_cflags=extra_cflags,
                    extra_cuda_cflags=extra_cuda_cflags,
                    extra_include_paths=extra_include_paths,
                    build_directory=build_dir,
                )

    else:
        Console().print(
            "[yellow]{name}: No CUDA toolkit found. {name} will be disabled.[/yellow]"
        )
        return None

_C = None

try:
    # try to import the compiled module (via setup.py)
        from pysegreduce_cuda import csrc as _C
except ImportError:
    SRC_PATH = str(Path(os.path.abspath(__file__)).parent.parent)
    MGPU_PATH = str(Path(SRC_PATH).parent / "moderngpu" / "src")
    _C = load_jit_cuda_extension(
        name="pysegreduce",
        sources=[f"{SRC_PATH}/ext.cu"],
        extra_include_paths=[SRC_PATH, MGPU_PATH],
        extra_cflags=["-O3"],
        extra_cuda_cflags=["-O3", "--expt-extended-lambda", "--expt-relaxed-constexpr", "--use_fast_math", "-lineinfo"],
    )

if need_to_unset_max_jobs:
    os.environ.pop("MAX_JOBS")


__all__ = ["_C"]