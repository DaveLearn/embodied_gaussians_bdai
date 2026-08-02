from typing import List, Optional
import open3d as o3d
import torch
from frame_seg_init.types import Frame
import logging

from frame_seg_init.mesh.point_segments import PointSegments
from frame_seg_init.mesh.mesh_points import QueryPointMode, get_query_points_from_mesh


logger = logging.getLogger(__name__)


class MeshWithSegments:
    def __init__(
        self,
        mesh: o3d.geometry.TriangleMesh,
        query_point_mode: QueryPointMode = "uniform",
        frames: Optional[List[Frame]] = None,
    ):
        self.mesh = mesh
        self.intersection_volume = o3d.t.geometry.RaycastingScene()
        self.intersection_volume.add_triangles(o3d.t.geometry.TriangleMesh.from_legacy(mesh))
        target_point_count = max(1, len(mesh.triangles) * 2)
        query_points = get_query_points_from_mesh(
            mesh,
            num_points=target_point_count,
            mode=query_point_mode,
            frames=frames,
            ray_scene=self.intersection_volume,
        )
        device = frames[0].color.device if frames else "cpu"
        self.point_segments = PointSegments(device=device)
        self.points = query_points.to(device=self.point_segments.device, dtype=torch.float32)
        logger.info(
            "Query point mode=%s target=%d actual=%d",
            query_point_mode,
            target_point_count,
            self.points.shape[0],
        )

    def assign_segment_to_points(self, point_ids: torch.Tensor, segment_id: torch.Tensor):
        segments_for_ids = segment_id.expand((point_ids.shape[0]))
        self.point_segments.add(point_ids, segments_for_ids)
