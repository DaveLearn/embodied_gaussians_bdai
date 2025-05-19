# Copyright (c) 2025 Boston Dynamics AI Institute LLC. All rights reserved.

import typing
from typing import Dict, Iterator, Tuple

from pathlib import Path
from dataclasses import dataclass

import json

import numpy as np



@dataclass
class ExtrinsicsData:
    X_WC: np.ndarray

def read_extrinsics(path: Path) -> Dict[str, ExtrinsicsData]:
    with open(path, "r") as f:
        extrinsics = json.load(f)
    res = {}
    for serial, data in extrinsics.items():
        res[serial] = ExtrinsicsData(
            X_WC=np.array(data["X_WT"])
        )
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