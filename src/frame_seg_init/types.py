from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch


GroundPlane = tuple[float, float, float, float]


@dataclass(frozen=True)
class ObservationFrame:
    """One posed RGB-D observation accepted by the public library API."""

    id: int
    name: str
    color: np.ndarray
    X_WV: np.ndarray
    K: np.ndarray
    depth: np.ndarray | None = None

    def __post_init__(self) -> None:
        if self.color.ndim != 3 or self.color.shape[2] != 3:
            raise ValueError(f"color must have shape HxWx3, got {self.color.shape}")
        if self.X_WV.shape != (4, 4):
            raise ValueError(f"X_WV must have shape 4x4, got {self.X_WV.shape}")
        if self.K.shape != (3, 3):
            raise ValueError(f"K must have shape 3x3, got {self.K.shape}")
        if self.depth is not None and self.depth.shape != self.color.shape[:2]:
            raise ValueError(f"depth shape {self.depth.shape} does not match color shape {self.color.shape[:2]}")


@dataclass(frozen=True)
class Observations:
    frames: list[ObservationFrame]
    cache_key: str | None = None

    def __post_init__(self) -> None:
        frame_ids = [frame.id for frame in self.frames]
        if len(frame_ids) != len(set(frame_ids)):
            raise ValueError("Observation frame IDs must be unique")


@dataclass(frozen=True)
class Workspace:
    """Geometry used to reject proposals outside the physical workspace."""

    ground_points: np.ndarray
    ground_plane: GroundPlane

    def __post_init__(self) -> None:
        if self.ground_points.ndim != 2 or self.ground_points.shape[1:] != (3,):
            raise ValueError(f"ground_points must have shape Nx3, got {self.ground_points.shape}")
        if not np.isfinite(self.ground_points).all():
            raise ValueError("ground_points must be finite")
        if len(self.ground_plane) != 4:
            raise ValueError("ground_plane must contain four coefficients")
        normal = np.asarray(self.ground_plane[:3], dtype=np.float64)
        if not np.isfinite(normal).all() or np.linalg.norm(normal) == 0:
            raise ValueError("ground_plane must have a finite, non-zero normal")


@dataclass(frozen=True)
class SegmentationResult:
    """Globally consistent object IDs for each input frame."""

    frame_ids: list[int]
    pixel_object_ids: list[np.ndarray]

    def __post_init__(self) -> None:
        if len(self.frame_ids) != len(self.pixel_object_ids):
            raise ValueError(f"Frame/mask count mismatch: {len(self.frame_ids)} != {len(self.pixel_object_ids)}")


@dataclass(frozen=True)
class RuntimeConfig:
    """Runtime choices kept separate from the segmentation algorithm."""

    device: str = "cuda"
    cache_dir: Path | None = None
    checkpoint_path: Path | None = None

    def resolved_cache_dir(self) -> Path:
        if self.cache_dir is None:
            raise ValueError("cache_dir must be supplied explicitly")
        return self.cache_dir.expanduser().resolve()


@dataclass
class Frame:
    """Torch representation used internally by the graph and mesh algorithms."""

    id: int
    name: str
    color: torch.Tensor
    X_WV: torch.Tensor
    K: torch.Tensor
    depth: torch.Tensor | None = None

    @classmethod
    def from_observation(cls, observation: ObservationFrame, device: str | torch.device) -> Frame:
        target = torch.device(device)
        return cls(
            id=observation.id,
            name=observation.name,
            color=torch.as_tensor(observation.color, dtype=torch.float32, device=target),
            X_WV=torch.as_tensor(observation.X_WV, dtype=torch.float32),
            K=torch.as_tensor(observation.K, dtype=torch.float32),
            depth=(torch.as_tensor(observation.depth, dtype=torch.float32, device=target) if observation.depth is not None else None),
        )

    @property
    def w(self) -> int:
        return self.color.shape[1]

    @property
    def h(self) -> int:
        return self.color.shape[0]

    @property
    def fl_x(self) -> float:
        return float(self.K[0, 0])

    @property
    def fl_y(self) -> float:
        return float(self.K[1, 1])

    @property
    def cx(self) -> float:
        return float(self.K[0, 2])

    @property
    def cy(self) -> float:
        return float(self.K[1, 2])

    @property
    def X_VW(self) -> torch.Tensor:
        return torch.linalg.inv(self.X_WV)

    @property
    def X_VW_opencv(self) -> torch.Tensor:
        transform = self.X_VW.clone()
        transform[1:3, :] *= -1
        return transform

    @property
    def X_WV_opencv(self) -> torch.Tensor:
        return torch.linalg.inv(self.X_VW_opencv)

    def project(self, points: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        device = points.device
        X_VW = self.X_VW_opencv.to(device)
        xyz = (X_VW[:3, :3] @ points.T + X_VW[:3, 3:4]).T
        uv_h = (self.K.to(device) @ xyz.T).T
        uv = (uv_h / uv_h[:, 2:3])[..., :2].long()
        valid = (uv[:, 0] >= 0) & (uv[:, 0] < self.w) & (uv[:, 1] >= 0) & (uv[:, 1] < self.h) & (xyz[:, 2] > 0)
        return uv, valid
