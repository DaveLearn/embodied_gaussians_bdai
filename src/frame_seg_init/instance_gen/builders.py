import math
import time
import open3d as o3d
from typing import List, Optional, Tuple
from frame_seg_init.types import Frame
import torch
from frame_seg_init.instance_gen.frame_seg_scene_graph import FrameSegSceneGraph
from frame_seg_init.instance_gen.edge_matcher import EdgeScoreMethod
from frame_seg_init.instance_gen.instance_generator_new import compute_segment_instance_ids_new
from frame_seg_init.mesh.mesh_with_segments import MeshWithSegments

from frame_seg_init.proposals import FrameSegmenter
from frame_seg_init.instance_gen.mesh_points.edge_builder import (
    build_edges_from_mesh_points,
)
from frame_seg_init.instance_gen.mesh_points.segment_projection import (
    assign_segments_to_mesh_points,
)
from frame_seg_init.mesh import extract_mesh_bounded_with_res
from frame_seg_init.mesh.mesh_points import QueryPointMode

import logging

logger = logging.getLogger(__name__)


def build_scene_graph_with_mesh(
    frames: List[Frame],
    segmenter: FrameSegmenter,
    *,
    intersection_thresh: int,
    edge_score_method: EdgeScoreMethod,
    assign_orphaned_parents: bool,
    query_point_mode: QueryPointMode,
    existing_mesh: Optional[o3d.geometry.TriangleMesh] = None,
) -> Tuple[FrameSegSceneGraph, MeshWithSegments]:
    device = frames[0].color.device if frames else torch.device("cpu")
    sg = FrameSegSceneGraph(device=device)

    filter_occluded = query_point_mode != "frame_depth_backproject"
    if not filter_occluded:
        logger.info(
            "Disabling occlusion filtering for query_point_mode=%s",
            query_point_mode,
        )

    start_time = time.time()
    # populate nodes
    for frame in frames:
        segments = segmenter.segment(frame)
        sg.add_frame_segs(frame, segments)

    frame_time = time.time()
    logger.info(f"Computing frame Segments took {frame_time - start_time:.2f} seconds")
    logger.info(f"frame segs ids: {sg.frame_segs.ids.shape}")

    # spin up the mesh
    mesh = existing_mesh if existing_mesh is not None else extract_mesh_bounded_with_res(frames, mesh_res=1024)
    mesh_with_segments = MeshWithSegments(
        mesh,
        query_point_mode=query_point_mode,
        frames=frames,
    )

    mesh_time = time.time()
    logger.info(f"Updating mesh took {mesh_time - frame_time:.2f} seconds")
    logger.info(f"mesh points: {mesh_with_segments.points.shape}")
    logger.info(f"mesh segment ids: {mesh_with_segments.point_segments.point_ids_to_segment_ids.shape}")

    assign_segments_to_mesh_points(
        sg,
        mesh_with_segments,
        filter_occluded=filter_occluded,
    )
    build_edges_from_mesh_points(
        sg,
        mesh_with_segments,
        filter_occluded=filter_occluded,
    )
    element_segments = mesh_with_segments.point_segments.point_ids_to_segment_ids[:, :2]

    project_time = time.time()
    logger.info(f"projecting segments to mesh took {project_time - mesh_time:.2f} seconds")
    logger.info(f"edges: {sg.edges.frame_seg_ids.shape}")

    # merge segment instances based on overlap
    compute_segment_instance_ids_new(
        sg,
        intersection_thresh=intersection_thresh,
        edge_score_method=edge_score_method,
        assign_orphaned_parents=assign_orphaned_parents,
        element_segments=element_segments,
    )
    instance_time = time.time()
    logger.info(f"assigning instances took {instance_time - project_time:.2f} seconds")

    logger.info("--------------------")
    logger.info(f"graph create (after segmentation) {project_time - frame_time: .2f} seconds")
    logger.info(f"total time {instance_time - start_time}")
    return sg, mesh_with_segments


def _determine_table_instance_id(
    sg: FrameSegSceneGraph,
    table_plane: Tuple[float, float, float, float],
    object_ids: torch.Tensor,
) -> int:
    # for each instance, calculate the distance to the table plane for all points
    # the instance with the lowest distance is the table

    instance_ids = object_ids

    if len(instance_ids) == 0:
        return -1

    table_instance_candidates = []
    table_instance_counts = []

    for frame in sg.frames:
        assert frame.depth is not None
        # compute a mask of pixels in frame that are withing 0.01m of the table plane
        # Project depth image points to 3D world coordinates
        h, w = frame.depth.shape
        y, x = torch.meshgrid(
            torch.arange(h, device=frame.depth.device),
            torch.arange(w, device=frame.depth.device),
            indexing="ij",
        )

        valid_mask = frame.depth > 0

        # Get 3D points in camera space
        Z = frame.depth
        X = (x - frame.cx) * Z / frame.fl_x
        Y = (y - frame.cy) * Z / frame.fl_y

        # Stack into homogeneous coordinates
        points = torch.stack([X, Y, Z, torch.ones_like(Z)], dim=0)

        # Transform to world space
        points = frame.X_WV_opencv.to(points.device) @ points.reshape(4, -1)
        points = points.reshape(4, h, w)

        # Calculate signed distance to plane
        # plane equation: ax + by + cz + d = 0
        a, b, c, d = table_plane
        plane_dist = (a * points[0] + b * points[1] + c * points[2] + d) / math.sqrt(a * a + b * b + c * c)

        # Create mask for points within threshold of plane
        table_mask = torch.abs(plane_dist) < 0.02
        table_mask = table_mask & valid_mask

        for instance_id in instance_ids:
            instance_mask = sg.get_instance_id_mask_for_frame(int(instance_id.item()), frame.id)
            instance_mask_valid = instance_mask & valid_mask
            instance_mask_near_table = instance_mask & table_mask

            if instance_mask_near_table.sum() / instance_mask_valid.sum() > 0.7:
                table_instance_candidates.append(instance_id)
                table_instance_counts.append(instance_mask_near_table.sum())

    if len(table_instance_candidates) == 0:
        logger.warning("no table candidates found")
        return -1

    # Convert to tensors for easier indexing
    table_instance_candidates = torch.tensor(table_instance_candidates, device=sg.device)
    table_instance_counts = torch.tensor(table_instance_counts, device=sg.device)

    # Get the instance id with highest count near table
    best_candidate_idx = torch.argmax(table_instance_counts)
    best_candidate_id = table_instance_candidates[best_candidate_idx]

    return int(best_candidate_id.item())


def determine_object_instance_ids(sg: FrameSegSceneGraph, table_plane: Tuple[float, float, float, float]):
    # get instance_ids by size
    instance_ids, instance_counts = sg.get_instance_ids_by_size()

    # remove instances not seen in enough frames (see required_frame_count below)
    instance_frames = torch.cat(
        [sg.frame_segs.instance_ids.unsqueeze(1), sg.frame_segs.frame_ids.unsqueeze(1)],
        dim=1,
    ).unique(dim=0)
    instance_frame_ids, instance_frame_count = instance_frames[:, 0].unique(return_counts=True)
    required_frame_count = 3
    logger.debug(f"required_frame_count {required_frame_count}, instance ids before cull {instance_frame_ids} with counts {instance_frame_count}")
    object_instance_ids = instance_frame_ids[instance_frame_count >= (required_frame_count)]

    # exclude -1
    object_instance_ids = object_instance_ids[object_instance_ids != -1]
    logger.debug(f"instance ids after cull {object_instance_ids}")
    # our table is the one with highest count
    is_object_instance = torch.isin(instance_ids, object_instance_ids)

    instance_ids = instance_ids[is_object_instance]
    instance_counts = instance_counts[is_object_instance]

    # assign table instance id
    if instance_ids.shape[0] == 0:
        sg.object_instance_ids = torch.empty(0, dtype=torch.int32, device=sg.device)
    else:
        table_instance_id = _determine_table_instance_id(sg, table_plane, object_instance_ids)

        # assign object instance ids
        sg.object_instance_ids = instance_ids[instance_ids != table_instance_id].contiguous()
