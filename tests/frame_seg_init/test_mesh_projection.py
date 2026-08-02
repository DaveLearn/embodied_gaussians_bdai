from types import SimpleNamespace
from typing import Any, cast

import open3d as o3d
import torch
from frame_seg_init.types import Frame

from frame_seg_init.instance_gen.frame_seg_scene_graph import FrameSegSceneGraph
from frame_seg_init.instance_gen.mesh_points import segment_projection
from frame_seg_init.mesh.mesh_points import get_visible_points
from frame_seg_init.mesh.mesh_with_segments import MeshWithSegments


class _ProjectionFrame:
    def __init__(self, depth: torch.Tensor | None):
        self.id = 5
        self.depth = depth
        self.X_VW_opencv = torch.eye(4)

    def project(self, points: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        x = torch.arange(len(points), device=points.device)
        uv = torch.stack([x, torch.zeros_like(x)], dim=1)
        return uv, torch.ones(len(points), dtype=torch.bool, device=points.device)


def _unused_ray_scene() -> o3d.t.geometry.RaycastingScene:
    return cast(o3d.t.geometry.RaycastingScene, object())


def test_visible_points_always_filter_invalid_and_inaccurate_depth():
    frame = cast(Frame, _ProjectionFrame(torch.tensor([[1.0, 0.0, 0.95]])))
    query_points = torch.tensor([[0.0, 0.0, 1.0], [0.0, 0.0, 1.0], [0.0, 0.0, 1.0]])

    uv, point_ids = get_visible_points(query_points, _unused_ray_scene(), frame, filter_occluded=False)

    assert torch.equal(uv, torch.tensor([[0, 0]]))
    assert torch.equal(point_ids, torch.tensor([0]))


def test_visible_points_without_depth_keep_frustum_points():
    frame = cast(Frame, _ProjectionFrame(None))
    query_points = torch.tensor([[0.0, 0.0, 1.0], [0.0, 0.0, 1.0], [0.0, 0.0, 1.0]])

    uv, point_ids = get_visible_points(query_points, _unused_ray_scene(), frame, filter_occluded=False)

    assert torch.equal(uv, torch.tensor([[0, 0], [1, 0], [2, 0]]))
    assert torch.equal(point_ids, torch.tensor([0, 1, 2]))


def test_segment_projection_uses_global_frame_segment_id(monkeypatch: Any):
    frame = SimpleNamespace(id=5, depth=torch.ones((3, 3)))

    class FrameMasks:
        def mask_for_id(self, segment_idx: int) -> torch.Tensor:
            assert segment_idx == 7
            return torch.ones((3, 3), dtype=torch.bool)

    scene_graph = cast(
        FrameSegSceneGraph,
        SimpleNamespace(
            frames=[frame],
            frame_masks=[FrameMasks()],
            frame_segs=SimpleNamespace(
                frame_ids=torch.tensor([5]),
                ids=torch.tensor([42]),
                seg_idx=torch.tensor([7]),
            ),
        ),
    )

    assigned: list[tuple[torch.Tensor, torch.Tensor]] = []
    mesh = cast(
        MeshWithSegments,
        SimpleNamespace(
            points=torch.empty((0, 3)),
            intersection_volume=object(),
            assign_segment_to_points=lambda point_ids, segment_id: assigned.append((point_ids, segment_id)),
        ),
    )

    monkeypatch.setattr(segment_projection, "get_frame_points", lambda *args, **kwargs: torch.empty((0, 4)))
    monkeypatch.setattr(segment_projection, "get_point_idx_for_mask", lambda *args, **kwargs: torch.tensor([11]))

    segment_projection.assign_segments_to_mesh_points(scene_graph, mesh, filter_occluded=False)

    assert len(assigned) == 1
    assert torch.equal(assigned[0][0], torch.tensor([11]))
    assert int(assigned[0][1]) == 42
