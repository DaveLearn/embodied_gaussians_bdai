# Copyright (c) 2025 Boston Dynamics AI Institute LLC. All rights reserved.

from dataclasses import dataclass
from typing import Generic, Optional, TypeVar

import numpy as np
import torch
import warp as wp
import warp.sim

from embodied_gaussians.physics_simulator.builder import ModelBuilder
from embodied_gaussians.physics_simulator.integrator import XPBDIntegrator
from embodied_gaussians.utils.physics_utils import (
    clone_control,
    clone_state,
    copy_control,
    copy_state,
    cuda_graph_capture,
    synchronize_control,
    synchronize_state,
    transform_from_matrix,
    transform_to_matrix,
)


@dataclass
class PhysicsSettings:
    substeps: int = 30
    xpbd_iterations: int = 8
    dt: float = 1.0 / 60.0


TBuilderType = TypeVar("TBuilderType", bound=ModelBuilder)

class Simulator(Generic[TBuilderType]):
    def __init__(
        self, builder: TBuilderType, device: str = "cuda", requires_grad: bool = False
    ) -> None:
        self.builder = builder
        self.device = device
        self.model = builder.finalize(device, requires_grad)
        self.num_envs = self.model.num_envs
        self.state_0 = self.model.state()
        self.state_1 = self.model.state()
        self.control: warp.sim.Control = self.model.control()
        self.sim_time = 0.0
        self.eval_fk()

    def clone_control(self) -> warp.sim.Control:
        return clone_control(self.control)

    def clone_state(self) -> warp.sim.State:
        return clone_state(self.state_0)

    def set_state(self, state: warp.sim.State) -> None:
        copy_state(self.state_0, state)

    def set_control(self, control: warp.sim.Control) -> None:
        copy_control(self.control, control)

    def synchronize_state(self, state: warp.sim.State) -> None:
        synchronize_state(
            dst_state=self.state_0,
            src_state=state,
        )

    def synchronize_control(self, control: warp.sim.Control) -> None:
        synchronize_control(dst_control=self.control, src_control=control)

    def reset(self) -> None:
        self.sim_time = 0.0

    def get_time(self) -> float:
        return self.sim_time

    def eval_fk(self, mask: Optional[torch.Tensor] = None) -> None:
        if self.model.joint_count > 0:
            warp.sim.eval_fk(
                self.model,
                self.state_0.joint_q,
                self.state_0.joint_qd,
                mask,
                self.state_0,
            )

    def eval_ik(self) -> None:
        if self.model.joint_count > 0:
            warp.sim.eval_ik(
                self.model, self.state_0, self.state_0.joint_q, self.state_0.joint_qd
            )

    def set_body_q(self, body_id: int, X_WO: np.ndarray) -> None:
        s = wp.to_torch(self.state_0.body_q)
        T = transform_from_matrix(X_WO)
        s[body_id] = torch.from_numpy(T).float().to(self.device)

    def get_body_q(self, body_id: int) -> np.ndarray:
        s = wp.to_torch(self.state_0.body_q)[body_id].cpu().numpy()
        return transform_to_matrix(s)

    def get_articulation_q(self, index: int, num_joints: int) -> torch.Tensor:
        assert index < self.builder.articulation_count
        joint_start = self.builder.articulation_start[index]
        joint_q = wp.to_torch(self.state_0.joint_q)
        return joint_q[joint_start : joint_start + num_joints]
    
    def get_articulation_qd(self, index: int, num_joints: int) -> torch.Tensor:
        assert index < self.builder.articulation_count
        joint_start = self.builder.articulation_start[index]
        joint_qd = wp.to_torch(self.state_0.joint_qd)
        return joint_qd[joint_start : joint_start + num_joints]
    
    def check_articulation_healthy(self, index: int) -> bool:
        assert index < self.builder.articulation_count
        joint_qd = wp.to_torch(self.state_0.joint_qd)
        return bool(torch.isfinite(joint_qd).all())

    def set_articulation_q(self, index: int, q: torch.Tensor) -> None:
        assert index < self.builder.articulation_count
        if q.ndim == 1:
            q = q.unsqueeze(0) # replicate q for all envs
        q = q.to(self.device)
        joint_start = self.builder.articulation_start[index]
        given_joints = q.shape[1]

        joint_q = wp.to_torch(self.state_0.joint_q).reshape((self.num_envs, -1))
        joint_q[:, joint_start : joint_start + given_joints] = q
        joint_act = wp.to_torch(self.control.joint_act).reshape((self.num_envs, -1))
        joint_act[:, joint_start : joint_start + given_joints] = q

        warp.sim.eval_fk(
            self.model, self.state_0.joint_q, self.state_0.joint_qd, None, self.state_0
        )

    def set_articulation_control_q(self, index: int, q: torch.Tensor) -> None:
        assert index < self.builder.articulation_count
        if q.ndim == 1:
            q = q.unsqueeze(0) # replicate q for all envs
        q = q.to(self.device)
        joint_start = self.builder.articulation_start[index]
        given_joints = q.shape[1]
        joint_act = wp.to_torch(self.control.joint_act).reshape((self.num_envs, -1))
        joint_act[:, joint_start : joint_start + given_joints] = q

    def get_joint_act(self) -> torch.Tensor:
        return wp.to_torch(self.control.joint_act).reshape((self.num_envs, -1))

    def set_joint_act(self, joint_act: torch.Tensor) -> None:
        assert joint_act.shape[0] == self.num_envs
        ja = self.get_joint_act()
        ja.copy_(joint_act)

    def physics_step(self, settings: PhysicsSettings) -> None:
        self._physics_step(settings)
        self.sim_time += settings.dt

    @cuda_graph_capture
    def _physics_step(self, settings: PhysicsSettings) -> None:
        self.integrator = XPBDIntegrator(iterations=settings.xpbd_iterations)
        warp.sim.collide(self.model, self.state_0)
        for _ in range(settings.substeps):
            self.integrator.simulate(
                self.model,
                self.state_0,
                self.state_1,
                settings.dt / settings.substeps,
                self.control,
            )
            self.state_0, self.state_1 = self.state_1, self.state_0
        self.state_0.clear_forces()
        self.eval_ik()  # xpbd does not update the state of the joints since it operates on body_q. Do so manually.
