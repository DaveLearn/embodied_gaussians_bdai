from frame_seg_init.types import Frame, Workspace
import open3d as o3d
import numpy as np
from typing import List

from frame_seg_init.proposals import FrameSegmenter, ImageSegment, SegmenterModel


def _erode_voxel_grid_xy(voxel_grid: o3d.geometry.VoxelGrid, layers: int) -> o3d.geometry.VoxelGrid:
    if layers <= 0 or not voxel_grid.has_voxels():
        return voxel_grid

    voxel_indices = [tuple(int(idx) for idx in voxel.grid_index) for voxel in voxel_grid.get_voxels()]
    xy_occupied = {(x, y) for x, y, _ in voxel_indices}

    for _ in range(layers):
        if not xy_occupied:
            break
        prev_xy = xy_occupied
        xy_occupied = {
            (x, y) for (x, y) in prev_xy if ((x - 1, y) in prev_xy and (x + 1, y) in prev_xy and (x, y - 1) in prev_xy and (x, y + 1) in prev_xy)
        }

    for voxel_index in voxel_indices:
        if (voxel_index[0], voxel_index[1]) not in xy_occupied:
            voxel_grid.remove_voxel(voxel_index)

    return voxel_grid


def get_workspace_voxels(workspace: Workspace, shrink_xy_m: float = 0.04) -> o3d.geometry.VoxelGrid:
    table_xyz = workspace.ground_points

    table_plane = workspace.ground_plane
    table_normal = np.array([table_plane[0], table_plane[1], table_plane[2]])
    DESIRED_HEIGHT = 1.0
    BELOW_TABLE_HEIGHT = 0.10
    VOXEL_SIZE = 0.02
    above_offsets = np.arange(int(np.ceil(DESIRED_HEIGHT / VOXEL_SIZE)), dtype=np.float32)
    below_offsets = -np.arange(1, int(np.ceil(BELOW_TABLE_HEIGHT / VOXEL_SIZE)) + 1, dtype=np.float32)
    offsets = np.concatenate([below_offsets, above_offsets]) * VOXEL_SIZE
    table_pcd_extruded = (table_xyz[:, None, :] + offsets[None, :, None] * table_normal).reshape(-1, 3)

    table_pcd_extruded = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(table_pcd_extruded))

    voxel_grid = o3d.geometry.VoxelGrid.create_from_point_cloud(table_pcd_extruded, VOXEL_SIZE)

    layers = max(0, int(np.round(shrink_xy_m / voxel_grid.voxel_size)))
    return _erode_voxel_grid_xy(voxel_grid, layers)


# extracts segments but only those that are in the workspace defined by the scene setup
class WorkspaceSegmenter(FrameSegmenter):
    def __init__(self, workspace: Workspace, raw_segmenter: SegmenterModel):
        self.raw_segmenter = raw_segmenter
        self.workspace_voxels = get_workspace_voxels(workspace)
        # Cache rotation matrix
        from scipy.spatial.transform import Rotation as R

        rot = np.eye(4, dtype=np.float32)
        rot[:3, :3] = R.from_euler("x", 180, degrees=True).as_matrix()
        self.rot = rot

    def segment(self, frame: Frame) -> List[ImageSegment]:
        segments = self.raw_segmenter.segment_everything(frame.color)

        if not segments:
            return []

        camera_intrinsic = o3d.camera.PinholeCameraIntrinsic(frame.w, frame.h, frame.fl_x, frame.fl_y, frame.cx, frame.cy)
        transform = frame.X_WV.cpu().numpy() @ self.rot

        workspace_segments = []

        assert frame.depth is not None
        depth = frame.depth.cpu().contiguous().numpy()

        for segment in segments:
            mask = segment.get_bool_mask()
            masked_depth = np.where(mask, depth, 0)
            di = o3d.geometry.Image(masked_depth)
            pcd = o3d.geometry.PointCloud.create_from_depth_image(
                di,
                camera_intrinsic,
                depth_scale=1.0,
            )

            original_points = len(pcd.points)
            if original_points == 0:
                continue

            pcd.transform(transform)

            valid_mask = self.workspace_voxels.check_if_included(pcd.points)
            valid_points = np.count_nonzero(valid_mask)

            if valid_points / original_points > 0.50:
                workspace_segments.append(segment)

        if not workspace_segments:
            return []
        return workspace_segments
