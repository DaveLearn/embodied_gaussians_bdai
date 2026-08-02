# Copyright (c) 2025 Boston Dynamics AI Institute LLC. All rights reserved.

import warnings
from pathlib import Path

import numpy as np
import warp.sim
import zarr
from embodied_gaussians.utils.physics_utils import load_builder
from embodied_gaussians.utils.timestamps import timestamp_to_index


def as_np_array(val: zarr.Array | zarr.Group) -> np.ndarray:
    if isinstance(val, zarr.Group):
        raise ValueError(f"Expected array, got group: {val}")
    return val[:]  # type: ignore


class Loader:
    def __init__(self, device: str = "cuda"):
        self.device = device
        self.reset()

    def reset(self):
        self.current_index = 0
        self.num_steps = 0
        self._root = None
        self.simulator = None
        self.builder = None
        self._loaded = False

    def timestamp_at_index(self, index: int):
        return self.timestamps[index]

    def load(self, path: Path):
        assert path.exists()
        self._root = root = zarr.open_group(path)
        self.builder = load_builder(f"{path}/builder.pckl")

        p = self.builder.particle_count
        b = self.builder.body_count
        j = len(self.builder.joint_act)

        if p == 0 and b == 0:
            warnings.warn("No particles or bodies found. Not loading.")
            return None

        self.timestamps = as_np_array(root["timestamps"])
        self.num_steps = len(self.timestamps)
        if p > 0:
            self.state_particle_q = as_np_array(root["state_particle_q"])
            self.state_particle_qd = as_np_array(root["state_particle_qd"])
            self.state_particle_f = as_np_array(root["state_particle_f"])
        if b > 0:
            self.state_body_q = as_np_array(root["state_body_q"])
            self.state_body_qd = as_np_array(root["state_body_qd"])
            self.state_body_f = as_np_array(root["state_body_f"])
            self.num_timestamps = self.state_body_q.shape[0]  # type: ignore
        if j > 0:
            self.state_joint_q = as_np_array(root["state_joint_q"])
            self.state_joint_qd = as_np_array(root["state_joint_qd"])
            self.control_joint_act = as_np_array(root["control_joint_act"])

        self._loaded = True
        return root

    def get_state_at_timestampe(self, timestamp: float, device: str = "cuda"):
        index = timestamp_to_index(self.timestamps, timestamp)
        return self.get_state_at_index(index, device)

    def get_state_at_index(self, index: int, device: str = "cuda"):
        assert self.builder is not None
        p = self.builder.particle_count
        b = self.builder.body_count
        j = len(self.builder.joint_act)
        state = warp.sim.State()
        control = warp.sim.Control()
        if p > 0:
            state.particle_q = warp.from_numpy(self.state_particle_q[index], device=device)
            state.particle_qd = warp.from_numpy(self.state_particle_qd[index], device=device)
            state.particle_f = warp.from_numpy(self.state_particle_f[index], device=device)

        if b > 0:
            state.body_q = warp.from_numpy(self.state_body_q[index], device=device)
            state.body_qd = warp.from_numpy(self.state_body_qd[index], device=device)
            state.body_f = warp.from_numpy(self.state_body_f[index], device=device)
        if j > 0:
            state.joint_q = warp.from_numpy(self.state_joint_q[index], device=device)
            state.joint_qd = warp.from_numpy(self.state_joint_qd[index], device=device)
            control.joint_act = warp.from_numpy(self.control_joint_act[index], device=device)
        return state, control
