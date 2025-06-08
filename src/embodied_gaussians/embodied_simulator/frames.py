# Copyright (c) 2025 Boston Dynamics AI Institute LLC. All rights reserved.

from dataclasses import dataclass
from typing import List, Optional
import numpy as np
import torch


@dataclass
class Frames:
    width: int
    height: int
    names: list[str]
    timestamps: list[float]
    Ks_cpu: torch.Tensor
    Ks_gpu: torch.Tensor
    X_WCs_cpu: torch.Tensor
    X_CWs_opencv_gpu: torch.Tensor
    colors_gpu: torch.Tensor  # float32, [h, w, 3] [0, 1] RGB
    depths_gpu: torch.Tensor  # float32, [h, w] 
    device: str = "cuda"

    """
    colors_gpu is expected to be in rgb format
    """

    def update_colors(self, name: str, timestamp: float, color: torch.Tensor, depth: Optional[torch.Tensor] = None) -> None:
        index = self.names.index(name)
        assert color.shape == (self.height, self.width, 3)
        self.timestamps[index] = timestamp
        self.colors_gpu[index].copy_(color)
        if depth is not None:
            assert depth.shape == (self.height, self.width)
            self.depths_gpu[index].copy_(depth)


class FramesBuilder:
    def __init__(self, width: int, height: int) -> None:
        self.width = width
        self.height = height
        self.names: List[str] = []
        self.Ks: List[torch.Tensor] = []
        self.X_WCs: List[torch.Tensor] = []  # blender standard
        self.X_CWs_opencv: List[torch.Tensor] = []  # opencv standard

    def add_camera(self, name: str, K: np.ndarray, X_WC: np.ndarray) -> None:
        self.names.append(name)
        self.Ks.append(torch.from_numpy(K))
        X_WC_opencv = X_WC @ np.array([[1, 0, 0, 0], [0, -1, 0, 0], [0, 0, -1, 0], [0, 0, 0, 1]])
        X_CW_opencv = np.linalg.inv(X_WC_opencv)
        self.X_WCs.append(torch.from_numpy(X_WC))
        self.X_CWs_opencv.append(torch.from_numpy(X_CW_opencv))

    def finalize(self, device: str = "cuda") -> Frames:
        num_frames = len(self.names)
        return Frames(
            width=self.width,
            height=self.height,
            names=self.names,
            device=device,
            timestamps=[-1.0] * num_frames,
            Ks_cpu=torch.stack(self.Ks).float(),
            Ks_gpu=torch.stack(self.Ks).float().to(device),
            X_WCs_cpu=torch.stack(self.X_WCs).float(),
            X_CWs_opencv_gpu=torch.stack(self.X_CWs_opencv).float().to(device),
            colors_gpu=torch.zeros(
                (num_frames, self.height, self.width, 3),
                dtype=torch.float32,
                device=device,
            ),
            depths_gpu=torch.zeros(
                (num_frames, self.height, self.width),
                dtype=torch.float32,
                device=device,
            ),
        )
