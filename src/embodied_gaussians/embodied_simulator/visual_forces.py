# Copyright (c) 2025 Boston Dynamics AI Institute LLC. All rights reserved.

from dataclasses import dataclass
import torch
import warp as wp

from embodied_gaussians.embodied_simulator.adam import Adam
from embodied_gaussians.embodied_simulator.gaussians import GaussianModel, GaussianState


@dataclass
class VisualForcesSettings:
    iterations: int = 3
    lr_means: float = 0.0015
    lr_quats: float = 0.001
    lr_color: float = 0.000
    lr_opacity: float = 0.0000
    lr_scale: float = 0.0000
    kp: float = 4.0


class VisualForces:
    means: torch.Tensor  # (n_gaussians, 3)
    quats: torch.Tensor  # (n_gaussians, 4) (w, x, y, z)
    forces: torch.Tensor  # (n_gaussians, 3)
    moments: torch.Tensor  # (n_gaussians, 3)

    def __init__(
        self,
        gaussian_model: GaussianModel,
        gaussian_state: GaussianState,
        bodies_affected_by_visual_forces: list[int],
    ):
        num_gaussians = gaussian_model.means.shape[0]
        device = gaussian_model.means.device
        self.device = device
        self.means = torch.zeros((num_gaussians, 3), dtype=torch.float32, device=device)
        self.quats = torch.zeros((num_gaussians, 4), dtype=torch.float32, device=device)
        self.forces = torch.zeros((num_gaussians, 3), dtype=torch.float32, device=device)
        self.moments = torch.zeros((num_gaussians, 3), dtype=torch.float32, device=device)
        self.means.requires_grad = True
        self.quats.requires_grad = True

        bodies_affected_by_visual_forces_tensor = torch.tensor(bodies_affected_by_visual_forces).int().cuda()
        body_ids = gaussian_model.body_ids
        # find body ids that are affected by visual forces
        mask = torch.zeros_like(body_ids, dtype=torch.bool)
        for b in bodies_affected_by_visual_forces_tensor:
            mask = mask | (body_ids == b)
        self._gaussians_not_involved_in_visual_forces = ~mask

        self._initialize(gaussian_model.body_ids)

        self.optimizer = Adam(
            [
                wp.from_torch(self.means, dtype=wp.vec3),
                wp.from_torch(self.quats, dtype=wp.vec4),
            ],
            lrs=[0.01, 0.01],  # type: ignore
        )
        self.gaussian_state = gaussian_state

    def set_learnings_rates(self, lrs):
        self.optimizer.lrs = lrs

    def zero_grad(self):
        if self.means.grad is not None:
            self.means.grad.zero_()
        if self.quats.grad is not None:
            self.quats.grad.zero_()

    def step(self):
        # zero out non-visual forces
        assert self.means.grad is not None
        assert self.quats.grad is not None
        self.means.grad[self._gaussians_not_involved_in_visual_forces, :] = 0
        self.quats.grad[self._gaussians_not_involved_in_visual_forces, :] = 0
        self.optimizer.step(
            grad=[
                wp.from_torch(self.means.grad, dtype=wp.vec3),
                wp.from_torch(self.quats.grad, dtype=wp.vec4),
            ]
        )

    def _initialize(self, body_ids: torch.Tensor):
        segment_body_ids, segment_ids = torch.unique_consecutive(body_ids, return_inverse=True)
        valid_segments = segment_body_ids != -1
        self._body_ids = segment_body_ids[valid_segments]
        self._num_bodies = len(self._body_ids)

        # index_add_ cannot use -1 as an index. Map non-body Gaussians to a
        # trailing sink row, which is ignored by apply_forces_kernel.
        segment_to_output = torch.full(
            (len(segment_body_ids),),
            self._num_bodies,
            device=self.device,
            dtype=torch.int64,
        )
        segment_to_output[valid_segments] = torch.arange(self._num_bodies, device=self.device, dtype=torch.int64)
        self._reduction_ids = segment_to_output[segment_ids]
        self._total_forces = torch.zeros((self._num_bodies + 1, 3), device=self.device, dtype=torch.float32)
        self._total_moments = torch.zeros((self._num_bodies + 1, 3), device=self.device, dtype=torch.float32)

    def sum_forces_by_body(self):
        self._total_forces.zero_()
        self._total_moments.zero_()
        self._total_forces.index_add_(0, self._reduction_ids, self.forces)
        self._total_moments.index_add_(0, self._reduction_ids, self.moments)
