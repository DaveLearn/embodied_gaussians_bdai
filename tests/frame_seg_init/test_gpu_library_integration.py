from pathlib import Path

import numpy as np
import open3d as o3d
import pytest
import torch

from frame_seg_init import GraphParams, ObservationFrame, Observations, RuntimeConfig, Workspace, segment_scene
from frame_seg_init.proposals import ImageSegment


class _FullFrameProposalModel:
    def segment_everything(self, image: torch.Tensor) -> list[ImageSegment]:
        mask = torch.ones(image.shape[:2], dtype=torch.bool, device=image.device)
        rle, area = ImageSegment.mask_data_from_pytorch_mask(mask)
        return [ImageSegment(mask_rle=rle, area=area, predicted_iou=1.0)]


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA device required")
def test_public_library_path_associates_instances_on_cuda(tmp_path: Path) -> None:
    height = width = 4
    intrinsics = np.array([[100.0, 0.0, 1.5], [0.0, 100.0, 1.5], [0.0, 0.0, 1.0]], dtype=np.float32)
    frames = [
        ObservationFrame(
            id=frame_id,
            name=f"frame_{frame_id}",
            color=np.zeros((height, width, 3), dtype=np.float32),
            depth=np.ones((height, width), dtype=np.float32),
            X_WV=np.eye(4, dtype=np.float32),
            K=intrinsics,
        )
        for frame_id in range(3)
    ]

    xy = np.linspace(-0.1, 0.1, 11, dtype=np.float32)
    xx, yy = np.meshgrid(xy, xy)
    ground_points = np.column_stack([xx.reshape(-1), yy.reshape(-1), np.full(xx.size, -1.1, dtype=np.float32)])

    vertices = np.array(
        [
            [-0.02, -0.02, -1.0],
            [0.02, -0.02, -1.0],
            [0.02, 0.02, -1.0],
            [-0.02, 0.02, -1.0],
        ]
    )
    triangles = np.tile(np.array([[0, 1, 2], [0, 2, 3]], dtype=np.int32), (12, 1))
    mesh = o3d.geometry.TriangleMesh(
        o3d.utility.Vector3dVector(vertices),
        o3d.utility.Vector3iVector(triangles),
    )
    mesh_path = tmp_path / "mesh.ply"
    assert o3d.io.write_triangle_mesh(str(mesh_path), mesh)

    result = segment_scene(
        Observations(frames=frames),
        Workspace(ground_points=ground_points, ground_plane=(0.0, 0.0, 1.0, 1.1)),
        params=GraphParams(intersection_count_thresh=1, query_point_mode="frame_depth_backproject"),
        runtime=RuntimeConfig(device="cuda"),
        mesh_path=mesh_path,
        proposal_model=_FullFrameProposalModel(),
    )

    assert result.frame_ids == [0, 1, 2]
    assert all(mask.shape == (height, width) for mask in result.pixel_object_ids)
    assert all(np.any(mask > 0) for mask in result.pixel_object_ids)
