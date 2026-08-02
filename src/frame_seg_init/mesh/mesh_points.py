from typing import List, Literal
from frame_seg_init.types import Frame
import numpy as np
import torch
import open3d as o3d
import logging


QueryPointMode = Literal[
    "uniform",
    "vertices",
    "mesh_depth_backproject",
    "frame_depth_backproject",
]


logger = logging.getLogger(__name__)

_DEPTH_ERROR_THRESHOLD = 0.05


def _cap_points(points: torch.Tensor, max_points: int, seed: int = 0) -> torch.Tensor:
    if points.shape[0] <= max_points:
        return points

    generator = torch.Generator(device="cpu")
    generator.manual_seed(seed)
    idx = torch.randperm(points.shape[0], generator=generator, device="cpu")[:max_points]
    return points[idx.to(points.device)]


def _filter_finite_points(points: torch.Tensor) -> torch.Tensor:
    if points.shape[0] == 0:
        return points
    return points[torch.isfinite(points).all(dim=1)]


def _get_mesh_depth_backproject_points(
    frames: List[Frame],
    ray_scene: o3d.t.geometry.RaycastingScene,
    max_points: int,
) -> torch.Tensor:
    if len(frames) == 0:
        return torch.empty((0, 3), dtype=torch.float32)

    frame_cap = max(1, max_points // len(frames))
    all_points = []
    for frame in frames:
        rays = ray_scene.create_rays_pinhole(
            o3d.core.Tensor(frame.K.cpu().numpy()),
            o3d.core.Tensor(frame.X_VW_opencv.cpu().numpy()),
            frame.w,
            frame.h,
        )
        ans = ray_scene.cast_rays(rays)

        rays_np = rays.numpy().reshape(-1, 6)
        t_hit = ans["t_hit"].numpy().reshape(-1)
        valid = np.isfinite(t_hit)
        if not np.any(valid):
            continue

        points_np = rays_np[valid, :3] + rays_np[valid, 3:] * t_hit[valid, None]
        frame_points = torch.from_numpy(points_np.astype(np.float32))
        frame_points = _filter_finite_points(frame_points)
        frame_points = _cap_points(frame_points, frame_cap)
        all_points.append(frame_points)

    if len(all_points) == 0:
        return torch.empty((0, 3), dtype=torch.float32)

    return _cap_points(torch.cat(all_points, dim=0), max_points)


def _get_frame_depth_backproject_points(frames: List[Frame], max_points: int) -> torch.Tensor:
    if len(frames) == 0:
        return torch.empty((0, 3), dtype=torch.float32)

    frame_cap = max(1, max_points // len(frames))
    all_points = []
    for frame in frames:
        if frame.depth is None:
            continue

        depth = frame.depth
        h, w = depth.shape
        yy, xx = torch.meshgrid(
            torch.arange(h, device=depth.device),
            torch.arange(w, device=depth.device),
            indexing="ij",
        )

        valid = depth > 0
        if not valid.any():
            continue

        z = depth[valid]
        x = xx[valid].to(dtype=depth.dtype)
        y = yy[valid].to(dtype=depth.dtype)

        cam_points = torch.stack(
            [
                (x - frame.K[0, 2].to(depth.device)) * z / frame.K[0, 0].to(depth.device),
                (y - frame.K[1, 2].to(depth.device)) * z / frame.K[1, 1].to(depth.device),
                z,
            ],
            dim=1,
        )

        X_WV_opencv = frame.X_WV_opencv.to(depth.device)
        world_points = (X_WV_opencv[:3, :3] @ cam_points.T + X_WV_opencv[:3, 3:4]).T
        world_points = _filter_finite_points(world_points)
        world_points = _cap_points(world_points, frame_cap)
        all_points.append(world_points.detach().cpu())

    if len(all_points) == 0:
        return torch.empty((0, 3), dtype=torch.float32)

    return _cap_points(torch.cat(all_points, dim=0), max_points)


def get_query_points_from_mesh(
    mesh: o3d.geometry.TriangleMesh,
    num_points: int = 500000,
    mode: QueryPointMode = "uniform",
    frames: List[Frame] | None = None,
    ray_scene: o3d.t.geometry.RaycastingScene | None = None,
) -> torch.Tensor:
    if mode == "uniform":
        o3d.utility.random.seed(0)
        points = mesh.sample_points_uniformly(number_of_points=num_points)
        return torch.from_numpy(np.asarray(points.points)).to(dtype=torch.float32)

    if mode == "vertices":
        return _cap_points(torch.from_numpy(np.asarray(mesh.vertices)).to(dtype=torch.float32), num_points)

    frames = frames or []
    if mode == "mesh_depth_backproject":
        if ray_scene is None:
            raise ValueError("ray_scene is required for mesh_depth_backproject query points")
        return _get_mesh_depth_backproject_points(frames=frames, ray_scene=ray_scene, max_points=num_points)

    if mode == "frame_depth_backproject":
        return _get_frame_depth_backproject_points(frames=frames, max_points=num_points)

    raise ValueError(f"Unknown query point mode: {mode}")


def get_points_in_frustrum(query_points: torch.Tensor, frame: Frame) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    all_points = query_points

    # get the points in the frame
    uv, valid_mask = frame.project(all_points)
    visible_point_idx = torch.arange(len(valid_mask), device=valid_mask.device)[valid_mask]
    visible_uv = uv[valid_mask]
    points_in_frame = all_points[valid_mask]

    return visible_uv, visible_point_idx, points_in_frame


# takes tensor of points [N,3] and returns a tensor of points that are inside the frames view frustrum and their uv coordinates
# query_points: [N, 3]
# returns: [M, 2] (u, v), [M] idx of visible points
def get_visible_points(
    query_points: torch.Tensor,
    ray_scene: o3d.t.geometry.RaycastingScene,
    frame: Frame,
    filter_occluded: bool = True,
    debug_context: str = "",
):
    all_points = query_points

    total_points = all_points.shape[0]
    visible_uv, visible_point_idx, points_in_frame = get_points_in_frustrum(all_points, frame)
    frustum_kept = points_in_frame.shape[0]
    frustum_dropped = total_points - frustum_kept

    bad_depth_dropped = 0
    depth_error_dropped = 0

    # filter out points where frame.depth = 0 at the UV coordinates
    if frame.depth is not None:
        # compute point depths
        X_VW = frame.X_VW_opencv.to(points_in_frame.device)
        xyz = (X_VW[:3, :3] @ points_in_frame.T + X_VW[:3, 3:4]).T
        point_depths = xyz[:, 2]  # these depths correspond to the visible uv

        # get depth values at the UV coordinates for valid points
        frame_depths = frame.depth[visible_uv[:, 1], visible_uv[:, 0]]  # note: UV is (x, y) but tensor indexing is [y, x]

        bad_depth_mask = frame_depths <= 0
        bad_depth_dropped = int(bad_depth_mask.sum().item())

        depth_error_mask = (~bad_depth_mask) & (torch.abs((point_depths - frame_depths) / point_depths) >= _DEPTH_ERROR_THRESHOLD)
        depth_error_dropped = int(depth_error_mask.sum().item())
        valid_depth_mask = (~bad_depth_mask) & (~depth_error_mask)

        # update masks and arrays to exclude points with zero depth
        visible_uv = visible_uv[valid_depth_mask]
        visible_point_idx = visible_point_idx[valid_depth_mask]
        points_in_frame = points_in_frame[valid_depth_mask]

    # now filter out points that are occluded
    occlusion_dropped = 0
    if filter_occluded:
        visible_mask = filter_occluded_points(points_in_frame, ray_scene, frame)
        occlusion_dropped = int((~visible_mask).sum().item())
    else:
        visible_mask = torch.ones(len(points_in_frame), dtype=torch.bool, device=points_in_frame.device)

    if logger.isEnabledFor(logging.DEBUG):
        logger.debug(
            ("Visible point filtering frame=%s context=%s total=%d frustum_drop=%d bad_depth_drop=%d depth_error_drop=%d occlusion_drop=%d kept=%d"),
            frame.id,
            debug_context,
            total_points,
            frustum_dropped,
            bad_depth_dropped,
            depth_error_dropped,
            occlusion_dropped,
            int(visible_mask.sum().item()),
        )

    return visible_uv[visible_mask], visible_point_idx[visible_mask]


# takes points [N, 3] and frame and returns [N] bool of whether point is visible
def filter_occluded_points(
    points: torch.Tensor,
    ray_scene: o3d.t.geometry.RaycastingScene,
    frame: Frame,
) -> torch.Tensor:
    # create some rays from the points to the camera.
    # get vector from point to camera, normalize it and use it as the direction of the ray.
    cam_pos = frame.X_WV_opencv[:3, 3].to(points.device)
    point_to_camera = cam_pos - points

    # normalize ray direction
    ray_direction = point_to_camera / torch.norm(point_to_camera, dim=1, keepdim=True)

    # offset ray origin slightly toward camera to avoid immediate self-hits
    RAY_ORIGIN_OFFSET = 0.0005  # half a mm
    ray_origin = points + ray_direction * RAY_ORIGIN_OFFSET

    # distance to camera
    cam_lense_offset = 0.05  # the camera housing may be part of the mesh and we don't want to count it as occluded if the ray hits the camera itself
    dist_to_camera = torch.abs(torch.norm(point_to_camera, dim=1)) - cam_lense_offset - RAY_ORIGIN_OFFSET

    # turn into ray 3d
    rays = o3d.core.Tensor(torch.cat([ray_origin, ray_direction], dim=1).cpu().numpy())
    ans = ray_scene.cast_rays(rays)

    dists = torch.tensor(ans["t_hit"].numpy(), device=points.device)

    visible = torch.abs(dists) >= dist_to_camera  # add 5cm to the distance to camera to account for the mesh being slightly offset from the camera

    return visible

    # A point is visible if:
    # 1. The ray doesn't intersect anything (intersection_count == 0), or
    # 2. The first intersection is farther than the camera (intersection_distance > distance_to_camera)


# takes query points [N, 3] and frame,  returns a tensor of [M, 4] (point_idx, frame_id, u, v)
def get_point_frame_pixels(
    query_points: torch.Tensor,
    ray_scene: o3d.t.geometry.RaycastingScene,
    frame: Frame,
    filter_occluded: bool = True,
    debug_context: str = "",
):
    # get the visible points
    uv, point_idx = get_visible_points(
        query_points,
        ray_scene,
        frame,
        filter_occluded=filter_occluded,
        debug_context=debug_context,
    )

    return torch.cat(
        [
            point_idx.view(-1, 1),
            frame.id * torch.ones(len(point_idx), dtype=torch.int64, device=point_idx.device).view(-1, 1),
            uv,
        ],
        dim=1,
    )


def get_frame_points(
    query_points: torch.Tensor,
    ray_scene: o3d.t.geometry.RaycastingScene,
    frames: List[Frame],
    filter_occluded: bool = True,
    debug_context: str = "",
):
    frame_points = []
    for frame in frames:
        frame_points.append(
            get_point_frame_pixels(
                query_points,
                ray_scene,
                frame,
                filter_occluded=filter_occluded,
                debug_context=debug_context,
            )
        )
    return torch.cat(frame_points, dim=0)


def get_point_idx_for_mask(points, mask, frame_id):
    """
    Extract point IDs from points tensor that match the given frame_id and fall within the mask.

    Args:
        points: Tensor of shape [N, 4] with each row as (point_id, frame_id, u, v)
        mask: Boolean tensor of shape [H, W] representing a mask for a particular frame
        frame_id: The frame ID to filter by

    Returns:
        Tensor containing the point_ids that match the criteria
    """
    # Step 1: Filter points by frame_id
    frame_mask = points[:, 1] == frame_id
    frame_points = points[frame_mask]

    if frame_points.size(0) == 0:
        return torch.tensor([], dtype=torch.long)

    # Step 2: Get u, v coordinates of the filtered points
    u_coords = frame_points[:, 2].long()
    v_coords = frame_points[:, 3].long()

    # Step 3: Filter by mask - check if coordinates are within bounds
    valid_indices = (u_coords >= 0) & (u_coords < mask.size(1)) & (v_coords >= 0) & (v_coords < mask.size(0))

    if not valid_indices.any():
        return torch.tensor([], dtype=torch.long)

    valid_points = frame_points[valid_indices]
    u_coords = valid_points[:, 2].long()
    v_coords = valid_points[:, 3].long()

    # Step 4: Index into the mask to check which points are within the masked region
    mask_values = mask[v_coords, u_coords]

    # Step 5: Return point_ids that fall within the mask
    return valid_points[mask_values, 0]
