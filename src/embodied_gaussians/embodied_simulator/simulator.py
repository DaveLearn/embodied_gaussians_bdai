# Copyright (c) 2025 Boston Dynamics AI Institute LLC. All rights reserved.

from typing import Literal, Optional, Tuple, Dict, Any
from embodied_gaussians.utils.utils import shrink_masks
import pysegreduce  # type: ignore
from dataclasses import dataclass
import torch
import warp as wp
import warp.sim
from gsplat.rendering import rasterization
from embodied_gaussians.physics_simulator.simulator import Simulator, copy_control, copy_state

from embodied_gaussians.embodied_simulator import EmbodiedGaussiansBuilder
from embodied_gaussians.embodied_simulator.gaussians import GaussianModel, GaussianState

from embodied_gaussians.embodied_simulator.appearance_optimizer import AppearanceOptimizer
from embodied_gaussians.embodied_simulator.frames import Frames
from embodied_gaussians.embodied_simulator.visual_forces import VisualForces, VisualForcesSettings
from embodied_gaussians.embodied_simulator.warp import (
    apply_forces_kernel,
    update_gaussians_transforms_kernel,
    update_visual_forces_kernel,
)
import open3d as o3d


@dataclass
class EmbodiedGaussianState:
    physics_state: warp.sim.State
    physics_control: warp.sim.Control
    gaussian_state: GaussianState


class EmbodiedGaussiansSimulator(Simulator[EmbodiedGaussiansBuilder]):
    def __init__(
        self,
        builder: EmbodiedGaussiansBuilder,
        device: str = "cuda",
        require_grad: bool = False,
    ) -> None:
        super().__init__(builder, device=device, requires_grad=require_grad)
        self.gaussian_model = builder.gaussian_model
        self.gaussian_state = builder.gaussian_state
        self.bodies_affected_by_visual_forces = builder.bodies_affected_by_visual_forces
        self.visual_forces = VisualForces(
            self.gaussian_model,
            self.gaussian_state,
            bodies_affected_by_visual_forces=self.bodies_affected_by_visual_forces,
        )
        self.appearance_optimizer = AppearanceOptimizer(self.gaussian_state)
        self.sync_confidence = 1.0
        self.max_outlier_fraction = 0.0
        self.mean_outlier_fraction = 0.0
        self.frame_outlier_fractions = []
        self.frame_noise_thresholds = []
        self.frame_outlier_thresholds = []
        self.frame_detection_scores = []
        self.workspace_bounding_box: Optional[o3d.t.geometry.AxisAlignedBoundingBox] = None

    def get_specific_environment_state(self, env_ind: int) -> EmbodiedGaussianState:
        with torch.no_grad():
            sim = self
            s = wp.to_torch(sim.state_0).reshape(self.num_envs, -1, 7)[env_ind]
            c = wp.to_torch(sim.control).reshape(self.num_envs, -1)[env_ind]
            g = sim.gaussian_state.reshape(self.num_envs, -1, 7).slice(env_ind).clone()
            s = wp.from_torch(s)
            c = wp.from_torch(c)
            return EmbodiedGaussianState(
                physics_state=s,  # type: ignore
                physics_control=c,  # type: ignore
                gaussian_state=g,  # type: ignore
            )

    def set_specific_environment_state(self, env_ind: int, state: EmbodiedGaussianState) -> None:
        sim = self
        with torch.no_grad():
            wp.to_torch(sim.state_0).reshape(self.num_envs, -1, 7)[env_ind] = wp.to_torch(state.physics_state)
            wp.to_torch(sim.control).reshape(self.num_envs, -1)[env_ind] = wp.to_torch(state.physics_control)
            g = sim.gaussian_state.reshape(self.num_envs, -1, 7).slice(env_ind)
            g.copy(state.gaussian_state)

    def embodied_gaussian_state(self) -> EmbodiedGaussianState:
        s = self.state_0
        c = self.control
        g = self.gaussian_state.clone()
        return EmbodiedGaussianState(physics_state=s, physics_control=c, gaussian_state=g)

    def clone_embodied_gaussian_state(self) -> EmbodiedGaussianState:
        s = self.clone_state()
        c = self.clone_control()
        g = self.gaussian_state.clone()
        return EmbodiedGaussianState(physics_state=s, physics_control=c, gaussian_state=g)

    def copy_embodied_gaussian_state(self, state: EmbodiedGaussianState) -> None:
        self.set_state(state.physics_state)
        self.set_control(state.physics_control)
        self.gaussian_state.copy(state.gaussian_state)

    def render_visual_forces(
        self,
        X_CWs: torch.Tensor,
        Ks: torch.Tensor,
        width: float,
        height: float,
        background: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor, Dict[str, Any]]:
        num_images = X_CWs.shape[0]
        with torch.no_grad():
            render_colors, render_alphas, info = rasterization(
                means=self.visual_forces.means,
                quats=self.visual_forces.quats,
                scales=self.gaussian_state.scales,
                colors=self.gaussian_state.colors,
                opacities=self.gaussian_state.opacities,
                viewmats=X_CWs,
                Ks=Ks,
                width=int(width),
                height=int(height),
                camera_model="pinhole",
                render_mode="RGB",
                # backgrounds=background.reshape(1, 3).repeat(num_images, 1),
                packed=True,
            )

            # todo: replace with backgrounds above when gsplat fixes assertion bug on packed rasterization
            backgrounds = background.reshape(1, 3).repeat(num_images, 1)
            render_colors = render_colors + backgrounds * (1 - render_alphas)
        return render_colors, render_alphas, info

    def render_gaussians(
        self,
        gaussian_state: GaussianState,
        X_CWs: torch.Tensor,
        Ks: torch.Tensor,
        width: float,
        height: float,
        background: torch.Tensor,
        near_plane: float = 0.01,
        far_plane: float = 3.0,
        render_mode: Literal["RGB", "D", "ED", "RGB+D", "RGB+ED"] = "RGB",
        **kwargs: Any,
    ) -> Tuple[torch.Tensor, torch.Tensor, Dict[str, Any]]:
        return render_gaussians(
            gaussian_state,
            X_CWs,
            Ks,
            width,
            height,
            background,
            near_plane,
            far_plane,
            render_mode,
            **kwargs,
        )

    def compute_visual_forces(self, settings: VisualForcesSettings, frames: Frames, dt: float) -> None:
        self._compute_visual_forces(settings, frames, dt)

    def update_gaussian_transforms(self) -> None:
        update_gaussian_transforms(self.gaussian_model, self.state_0.body_q, self.gaussian_state)

    def _compute_visual_forces(self, settings: VisualForcesSettings, frames: Frames, dt: float) -> None:
        with torch.no_grad():
            self.visual_forces.means.copy_(self.gaussian_state.means)
            self.visual_forces.quats.copy_(self.gaussian_state.quats)

        # self.visual_forces.optimizer.reset_internal_state()
        self.visual_forces.set_learnings_rates([settings.lr_means, settings.lr_quats])
        self.appearance_optimizer.set_learnings_rates([settings.lr_color, settings.lr_opacity, settings.lr_scale])

        for i in range(settings.iterations):
            render_colors, render_alphas, info = rasterization(
                means=self.visual_forces.means,
                quats=self.visual_forces.quats,
                scales=self.gaussian_state.scales,
                colors=self.gaussian_state.colors,
                opacities=self.gaussian_state.opacities,
                viewmats=frames.X_CWs_opencv_gpu,
                Ks=frames.Ks_gpu,
                width=int(frames.width),
                height=int(frames.height),
                camera_model="pinhole",
                render_mode="RGB",
                packed=False,  # TODO remove this once gsplat fixes assertion bug on backgrounds
            )

            loss = torch.nn.functional.mse_loss(render_colors, frames.colors_gpu)
            # ideas: add a loss that pushes the colors back to their orignal values or to some sort of ema colors
            # ideas: allow the gaussians to jitter a bit while anchoring them to the original positions

            self.visual_forces.zero_grad()
            self.appearance_optimizer.zero_grad()
            loss.backward()
            self.visual_forces.step()
            self.appearance_optimizer.step()

        wp.launch(
            kernel=update_visual_forces_kernel,
            dim=self.gaussian_model.num_gaussians,  # type: ignore
            inputs=[
                settings.kp,
                self.gaussian_state.means.detach(),
                self.gaussian_state.quats.detach(),
                self.gaussian_state.opacities.detach(),
                self.visual_forces.means.detach(),
                self.visual_forces.quats.detach(),
                self.gaussian_model.body_ids.detach(),
                self.state_0.body_q,
                self.visual_forces.forces.detach(),
                self.visual_forces.moments.detach(),
            ],
        )

        pysegreduce.reduce_vec3f(
            self.visual_forces.forces.data_ptr(),
            self.visual_forces._start_inds.data_ptr(),
            self.visual_forces._end_inds.data_ptr(),
            len(self.visual_forces._start_inds),
            self.visual_forces._total_forces.data_ptr(),
            0,
        )  # Replace this with segmented reduce when it is implemented in warp

        pysegreduce.reduce_vec3f(
            self.visual_forces.moments.data_ptr(),
            self.visual_forces._start_inds.data_ptr(),
            self.visual_forces._end_inds.data_ptr(),
            len(self.visual_forces._start_inds),
            self.visual_forces._total_moments.data_ptr(),
            0,
        )

        wp.launch(
            kernel=apply_forces_kernel,
            dim=self.visual_forces._num_bodies,  # type: ignore
            inputs=[
                dt,
                self.visual_forces._total_forces,
                self.visual_forces._total_moments,
                self.visual_forces._body_ids,
                self.state_0.body_f,
            ],
        )

    # returns the gaussian depth for each frame
    def compute_frames_expected_depths(self, frames: Frames) -> torch.Tensor:
        # render the expected depth from the gaussians we get [N, H, W, 1]
        render_depths, render_alphas, info = rasterization(
                means=self.gaussian_model.means,
                quats=self.gaussian_model.quats,
                scales=self.gaussian_model.scales,
                colors=self.gaussian_model.colors,
                opacities=self.gaussian_model.opacities,
                viewmats=frames.X_CWs_opencv_gpu,
                Ks=frames.Ks_gpu,
                width=int(frames.width),
                height=int(frames.height),
                camera_model="pinhole",
                render_mode="D"
            )
        return render_depths.squeeze(-1)
       

    # returns the depth error between the gaussian depth and the frame depth per pixel/frame
    # only considers depth pixels which are present in both, returns depth error and valid mask
    def compute_frame_depth_error(self, frames: Frames, max_depth: float = 2.0) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        max_depth_error = 0.4 # bleed off workspace
        # get the frame depth
        frame_depths = frames.depths_gpu
        # get the gaussian depth
        gaussian_depths = self.compute_frames_expected_depths(frames)
        valid_gaussian_mask = (gaussian_depths > 0.01)
        
        # shrink the gaussian mask to account for gsplat tendancy to misalign
        # we go for 5% of the image size
        shrink_amount = int(0.05 * frames.width)
        valid_gaussian_mask = shrink_masks(valid_gaussian_mask, shrink_amount=shrink_amount)


        # align the gaussians with our depth sensor using median to account for noise
        valid_median_mask = (valid_gaussian_mask) & (frame_depths > 0.01)
        diffs = frame_depths - gaussian_depths

        # Initialize offsets
        offsets = torch.zeros(frame_depths.shape[0], device=frame_depths.device)
        
        # Compute median for each batch item
        for i in range(frame_depths.shape[0]):
            valid_pixels = diffs[i][valid_median_mask[i]]
            if len(valid_pixels) > 0:
                offsets[i] = torch.median(valid_pixels)

        aligned_frame_depths = frame_depths - offsets.view(frame_depths.shape[0], 1, 1)

        # compute the depth disparity
        depth_error = torch.abs(aligned_frame_depths - gaussian_depths)

        
        valid_mask = valid_gaussian_mask & (frame_depths > 0.01) & (frame_depths < max_depth) & (depth_error < max_depth_error)

        return depth_error, valid_mask, valid_gaussian_mask


    def compute_frame_depth_outliers(self, frames: Frames, max_depth: float = 2.0) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        noise_percentile = 0.75 
        outlier_threshold_multiplier = 3.5
        
        depth_errors, valid_masks, valid_gaussian_masks = self.compute_frame_depth_error(frames, max_depth)
        
        # for each frame calc the threshold for outliers based on noise percentile
        outliers = torch.zeros_like(depth_errors, dtype=torch.int)
        frame_noise_thresholds = []
        frame_outlier_thresholds = []
        frame_detection_scores = []
        for i, (residuals,valid_mask) in enumerate(zip(depth_errors,valid_masks)):
            residuals = residuals[valid_mask].flatten() # ignore 0s as this are invalid pixels
            sorted_residuals, _ = torch.sort(residuals)
            noise_threshold = sorted_residuals[int(noise_percentile * sorted_residuals.numel())]

            # noise estimation using MAD on stable regions
            stable_mask = (residuals < noise_threshold)
            stable_diffs = residuals[stable_mask]
            mad = torch.median(torch.abs(stable_diffs - torch.median(stable_diffs)))
            noise_robust = 1.4826 * mad # make equivalent to std

            outlier_threshold = torch.maximum(noise_robust * outlier_threshold_multiplier, torch.tensor(0.025))
            frame_noise_thresholds.append(noise_robust)
            frame_outlier_thresholds.append(outlier_threshold)

            # weighting
            excess = torch.maximum(residuals - outlier_threshold, torch.zeros_like(residuals))
            weights = excess / (excess + noise_robust)

            weighted_excess = weights * excess
            # normalize by expected number of valid pixels
            detection_score = weighted_excess.sum() / stable_mask.sum()
            frame_detection_scores.append(detection_score.item())

            outliers[i,valid_mask] = (residuals > outlier_threshold).int()
        
        self.frame_noise_thresholds = frame_noise_thresholds
        self.frame_outlier_thresholds = frame_outlier_thresholds
        self.frame_detection_scores = frame_detection_scores
        return outliers, depth_errors, valid_masks, valid_gaussian_masks   

    def compute_depth_outlier_fraction(self, frames: Frames) -> float:
       

        outliers, depth_errors, valid_masks, valid_gaussian_masks = self.compute_frame_depth_outliers(frames) # [N, H, W]
       
        outlier_fractions = []
        for outlier, valid_mask, valid_gaussian_mask in zip(outliers,valid_masks, valid_gaussian_masks):
            outlier_fractions.append(outlier[valid_mask].float().sum() / valid_gaussian_mask.float().sum())
        
        self.frame_outlier_fractions = outlier_fractions
        # since some frames might not be observing the object, we instead just take the max outlier fraction across all frames
        max_outlier_fraction = max(outlier_fractions)
        return max_outlier_fraction

    def compute_sync_confidence(self, frames: Frames) -> float:
        max_outlier_fraction = 0.01  # TODO: this should be based on physical area represented by cluster of outliers
        # get the depth outlier fraction
        depth_outlier_fraction = self.compute_depth_outlier_fraction(frames)
        self.max_outlier_fraction = depth_outlier_fraction
        self.mean_outlier_fraction = sum(self.frame_outlier_fractions) / len(self.frame_outlier_fractions)
        # compute the sync confidence
        sync_confidence = 1.0 - min(depth_outlier_fraction/max_outlier_fraction, 1.0)
        self.sync_confidence = sync_confidence
        return sync_confidence    
 
    @torch.no_grad()
    def set_workspace_bounding_box_from_gaussians(self) -> None:
        import torch.utils.dlpack
        
        # get the ground plane normal
        ground_plane = wp.to_torch(self.model.ground_plane)
        ground_plane_normal = ground_plane[:3]

        # get the gaussian means
        table = self.gaussian_model.means[self.gaussian_model.body_ids < 0] # the table is the only non physical body
        roof = table.clone() + ground_plane_normal * 1.0 # one meter up

        points = torch.cat([table, roof], dim=0)

        points_o3d = o3d.core.Tensor.from_dlpack(points)

        # compute the bounding box
        bbox = o3d.t.geometry.AxisAlignedBoundingBox.create_from_points(points_o3d)

        self.workspace_bounding_box = bbox



def render_gaussians(
    gaussian_state: GaussianState,
    X_CWs: torch.Tensor,
    Ks: torch.Tensor,
    width: float,
    height: float,
    background: torch.Tensor,
    near_plane: float = 0.01,
    far_plane: float = 3.0,
    render_mode: Literal["RGB", "D", "ED", "RGB+D", "RGB+ED"] = "RGB",
    **kwargs: Any,
) -> Tuple[torch.Tensor, torch.Tensor, Dict[str, Any]]:
    num_images = X_CWs.shape[0]
    with torch.no_grad():
        render_colors, render_alphas, info = rasterization(
            means=gaussian_state.means,
            quats=gaussian_state.quats,
            scales=gaussian_state.scales,
            colors=gaussian_state.colors,
            opacities=gaussian_state.opacities,
            viewmats=X_CWs,
            Ks=Ks,
            width=int(width),
            height=int(height),
            camera_model="pinhole",
            render_mode=render_mode,
            # backgrounds=background.reshape(1, 3).repeat(num_images, 1),
            near_plane=near_plane,
            far_plane=far_plane,
            # packed=False,  # TODO remove this once gsplat fixes assertion bug on backgrounds
            **kwargs,
        )

        # todo: replace with backgrounds above when gsplat fixes assertion bug on packed rasterization
        if "RGB" in render_mode:
            backgrounds = background.reshape(1, 3).repeat(num_images, 1)
            if "D" in render_mode:
                # add a 0 to the 4th channel
                backgrounds = torch.cat([backgrounds, torch.zeros_like(backgrounds[:, :1])], dim=1)
        
            render_colors = render_colors + backgrounds * (1 - render_alphas)

    return render_colors, render_alphas, info


def update_gaussian_transforms(model: GaussianModel, body_q, out_state: GaussianState):
    if model.num_gaussians == 0:
        return
    wp.launch(
        kernel=update_gaussians_transforms_kernel,
        dim=model.num_gaussians,  # type: ignore
        inputs=[
            model.means,
            model.quats,
            model.body_ids,
            body_q,
        ],
        outputs=[
            out_state.means,
            out_state.quats,
        ],
    )


def copy_embodied_gaussian_state(dest: EmbodiedGaussianState, src: EmbodiedGaussianState):
    copy_state(dest.physics_state, src.physics_state)
    copy_control(dest.physics_control, src.physics_control)
    dest.gaussian_state.copy(src.gaussian_state)
