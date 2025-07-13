# Copyright (c) 2025 Boston Dynamics AI Institute LLC. All rights reserved.

from typing import Dict, Iterator, Tuple

from pathlib import Path
from dataclasses import dataclass

import json

import numpy as np
import torch


@dataclass
class ExtrinsicsData:
    X_WC: np.ndarray


def read_extrinsics(path: Path) -> Dict[str, ExtrinsicsData]:
    with open(path, "r") as f:
        extrinsics = json.load(f)
    res = {}
    for serial, data in extrinsics.items():
        res[serial] = ExtrinsicsData(X_WC=np.array(data["X_WT"]))
    return res


def read_ground(path: Path) -> np.ndarray:
    with open(path, "r") as f:
        ground = json.load(f)
    return np.array(ground["plane"])


class GridBuilder:
    def __init__(self, max_cols: int = 10, spacing: float = 1.0, z: float = 0.0) -> None:
        self.max_cols = max_cols
        self.spacing = spacing
        self.z = z
        self.num: int = 0

    def __iter__(self) -> Iterator[Tuple[float, float, float]]:
        self.num = 0
        return self

    def __next__(self) -> Tuple[float, float, float]:
        col = self.num % self.max_cols
        row = self.num // self.max_cols
        x = col * self.spacing
        y = row * self.spacing
        self.num += 1
        return (x, y, self.z)


def depth_to_points_3d_batch(depth_image: torch.Tensor, K: torch.Tensor, X_WC: torch.Tensor, depth_scale: float = 1.0) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Convert batched depth images to 3D points in world coordinates.
    
    Args:
        depth_image: (B, H, W) tensor with depth values 
        K: (B, 3, 3) camera intrinsics matrices
        X_WC: (B, 4, 4) transformation matrices from camera to world coordinates
        depth_scale: float, scale factor for depth values (default 1.0 for meters)
    
    Returns:
        points_3d: (B, H*W, 3) tensor of 3D points in world coordinates
        uv_coords: (B, H*W, 2) tensor of [u, v] pixel coordinates  
        valid: (B, H*W) tensor of boolean flags indicating valid points (depth > 0)
        
    Formula:
        z = d / depth_scale
        x = (u - cx) * z / fx  
        y = (v - cy) * z / fy
    """
    device = depth_image.device
    B, H, W = depth_image.shape
    N = H * W
    
    # Create pixel coordinates
    v, u = torch.meshgrid(
        torch.arange(H, device=device, dtype=torch.float32),
        torch.arange(W, device=device, dtype=torch.float32),
        indexing='ij'
    )
    
    # Stack to homogeneous coordinates [u, v, 1] and flatten
    pixel_coords = torch.stack([u, v, torch.ones_like(u)], dim=-1)  # (H, W, 3)
    pixel_coords = pixel_coords.view(N, 3)  # (H*W, 3)
    pixel_coords = pixel_coords.unsqueeze(0).expand(B, -1, -1)  # (B, H*W, 3)
    
    # UV coordinates for all pixels
    uv_coords = pixel_coords[..., :2]  # (B, H*W, 2) - [u, v]
    
    # Find valid pixels (depth > 0)
    valid = (depth_image > 0).view(B, N)  # (B, H*W)
    
    # Apply depth scaling: z = d / depth_scale
    depth_scaled = depth_image / depth_scale
    depth_flat = depth_scaled.view(B, N)  # (B, H*W)
    
    # Convert to normalized camera coordinates: [(u-cx)/fx, (v-cy)/fy, 1]
    K_inv = torch.inverse(K)  # (B, 3, 3)
    cam_coords = torch.bmm(pixel_coords, K_inv.transpose(1, 2))  # (B, H*W, 3)
    
    # Scale by depth to get 3D points in camera frame:
    # x = (u-cx)/fx * z,  y = (v-cy)/fy * z,  z = z
    cam_coords_3d = cam_coords * depth_flat.unsqueeze(-1)  # (B, H*W, 3)
    
    # Convert to homogeneous coordinates for world transformation
    cam_coords_homo = torch.cat([
        cam_coords_3d, 
        torch.ones(B, N, 1, device=device)
    ], dim=-1)  # (B, H*W, 4)
    
    # Transform to world coordinates (vectorized across batch)
    points_3d = torch.bmm(cam_coords_homo, X_WC.transpose(1, 2))[..., :3]  # (B, H*W, 3)
    
    return points_3d, uv_coords, valid