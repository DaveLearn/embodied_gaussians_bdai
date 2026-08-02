from types import SimpleNamespace
from typing import Any, cast

import numpy as np
import open3d as o3d
import torch

from frame_seg_init.proposals import ImageSegment
from frame_seg_init.types import Frame
from frame_seg_init.workspace_segmenter import WorkspaceSegmenter


def test_workspace_segmenter_forwards_metric_depth_scale(monkeypatch: Any) -> None:
    mask = torch.ones((2, 2), dtype=torch.bool)
    rle, area = ImageSegment.mask_data_from_pytorch_mask(mask)
    raw_segmenter = SimpleNamespace(segment_everything=lambda _image: [ImageSegment(mask_rle=rle, area=area, predicted_iou=1.0)])
    segmenter = cast(WorkspaceSegmenter, object.__new__(WorkspaceSegmenter))
    segmenter.raw_segmenter = raw_segmenter
    segmenter.rot = np.eye(4, dtype=np.float32)
    segmenter.workspace_voxels = SimpleNamespace(check_if_included=lambda points: [True] * len(points))

    captured: dict[str, float] = {}
    pointcloud = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(np.ones((4, 3))))

    def create_from_depth_image(*args: Any, **kwargs: Any) -> o3d.geometry.PointCloud:
        captured["depth_scale"] = kwargs["depth_scale"]
        return pointcloud

    monkeypatch.setattr(o3d.geometry.PointCloud, "create_from_depth_image", create_from_depth_image)
    frame = Frame(
        id=0,
        name="camera",
        color=torch.zeros((2, 2, 3)),
        depth=torch.ones((2, 2)),
        X_WV=torch.eye(4),
        K=torch.eye(3),
    )

    result = segmenter.segment(frame)

    assert len(result) == 1
    assert captured["depth_scale"] == 1.0
