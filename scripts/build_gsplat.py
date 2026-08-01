"""Build the pinned gsplat revision for the GPU visible to this environment."""

import os
import subprocess
import sys

import torch


GSPLAT_URL = "git+https://github.com/nerfstudio-project/gsplat.git@d23d7ca5dd26c3756967304b621ae88521672ed5"


def main() -> None:
    if not torch.cuda.is_available():
        raise RuntimeError("A visible CUDA GPU is required to build gsplat")

    major, minor = torch.cuda.get_device_capability()
    build_environment = os.environ.copy()
    build_environment["TORCH_CUDA_ARCH_LIST"] = f"{major}.{minor}"
    subprocess.run(
        [sys.executable, "-m", "pip", "install", GSPLAT_URL, "--no-build-isolation"],
        check=True,
        env=build_environment,
    )


if __name__ == "__main__":
    main()
