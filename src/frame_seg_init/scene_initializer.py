from __future__ import annotations

from dataclasses import dataclass
import logging
from pathlib import Path

import numpy as np
import open3d as o3d
import torch

from frame_seg_init.cache import CachingSegmenter, PrecomputedSegmenter
from frame_seg_init.instance_gen.builders import build_scene_graph_with_mesh, determine_object_instance_ids
from frame_seg_init.instance_gen.edge_matcher import EdgeScoreMethod
from frame_seg_init.instance_gen.vis import generate_instance_mask_stacks
from frame_seg_init.mesh.mesh_points import QueryPointMode
from frame_seg_init.proposals import FrameSegmenter, SegmenterModel
from frame_seg_init.sam import SAMSegmenter
from frame_seg_init.types import Frame, Observations, RuntimeConfig, SegmentationResult, Workspace
from frame_seg_init.workspace_segmenter import WorkspaceSegmenter


logger = logging.getLogger(__name__)


@dataclass
class GraphParams:
    intersection_count_thresh: int = 300
    flatten_masks: bool = False
    rebuild_cache: bool = False
    edge_score_method: EdgeScoreMethod = "iou"
    assign_orphaned_parents: bool = True
    query_point_mode: QueryPointMode = "uniform"


def build_segmenter(
    params: GraphParams,
    observations: Observations,
    workspace: Workspace,
    runtime: RuntimeConfig,
    proposal_model: SegmenterModel | None = None,
) -> FrameSegmenter:
    if proposal_model is None:
        logger.info("Building SAM segmenter")
        base_segmenter: SegmenterModel = SAMSegmenter(
            flatten_masks=params.flatten_masks,
            device=runtime.device,
            checkpoint_path=runtime.checkpoint_path,
        )
    else:
        base_segmenter = proposal_model
    workspace_segmenter = WorkspaceSegmenter(workspace, base_segmenter)

    if observations.cache_key is None:
        logger.info("No cache key supplied; frame proposals will not be cached")
        return workspace_segmenter

    cache_name = "sam_hq_flattened" if params.flatten_masks else "sam_hq"
    cache_path = runtime.resolved_cache_dir() / observations.cache_key / "segmentations" / cache_name
    if params.rebuild_cache and cache_path.exists():
        logger.info("Rebuilding cache at %s", cache_path)
        for item in cache_path.iterdir():
            if item.is_file():
                item.unlink()

    id_to_name: dict[int, str] = {}
    for frame in observations.frames:
        if frame.id in id_to_name:
            raise ValueError(f"Duplicate frame id {frame.id} in observations")
        id_to_name[frame.id] = frame.name

    return CachingSegmenter(
        precomputed_segmenter=PrecomputedSegmenter(cache_path, id_to_name, device=runtime.device),
        segmenter=workspace_segmenter,
    )


def segment_scene(
    observations: Observations,
    workspace: Workspace,
    params: GraphParams,
    runtime: RuntimeConfig,
    intermediate_outputs_path: Path | None = None,
    mesh_path: Path | None = None,
    proposal_model: SegmenterModel | None = None,
) -> SegmentationResult:
    """Associate per-frame SAM proposals into persistent object IDs."""

    if not observations.frames:
        return SegmentationResult(frame_ids=[], pixel_object_ids=[])
    if len(workspace.ground_points) == 0:
        raise ValueError("workspace ground_points must not be empty")

    segmenter = build_segmenter(params, observations, workspace, runtime, proposal_model)
    frames = [Frame.from_observation(frame, runtime.device) for frame in observations.frames]
    mesh_cache_path: Path | None = None

    logger.info("Building scene graph")
    if mesh_path is not None:
        if not mesh_path.exists():
            raise FileNotFoundError(f"Mesh not found at {mesh_path}")
        logger.info("Loading shared mesh from %s", mesh_path)
        mesh = o3d.io.read_triangle_mesh(str(mesh_path))
    elif observations.cache_key is not None:
        mesh_cache_path = runtime.resolved_cache_dir() / observations.cache_key / "mesh" / "mesh.ply"
        if mesh_cache_path.exists() and not params.rebuild_cache:
            logger.info("Loading cached mesh from %s", mesh_cache_path)
            mesh = o3d.io.read_triangle_mesh(str(mesh_cache_path))
        else:
            mesh = None
    else:
        mesh = None

    scene_graph, mesh_with_segments = build_scene_graph_with_mesh(
        frames,
        segmenter,
        intersection_thresh=params.intersection_count_thresh,
        existing_mesh=mesh,
        edge_score_method=params.edge_score_method,
        assign_orphaned_parents=params.assign_orphaned_parents,
        query_point_mode=params.query_point_mode,
    )

    if mesh is None and mesh_cache_path is not None:
        mesh_cache_path.parent.mkdir(parents=True, exist_ok=True)
        o3d.io.write_triangle_mesh(str(mesh_cache_path), mesh_with_segments.mesh)

    determine_object_instance_ids(scene_graph, workspace.ground_plane)
    object_ids = scene_graph.object_instance_ids

    if intermediate_outputs_path is not None and len(object_ids) > 0:
        output_path = intermediate_outputs_path / "instance_masks"
        output_path.mkdir(parents=True, exist_ok=True)
        for frame, stack in zip(
            observations.frames,
            generate_instance_mask_stacks(scene_graph, object_ids, pack_ids=True),
            strict=True,
        ):
            stack.save_to_file(str(output_path / f"{frame.name}.ms"))

    stacks = (
        generate_instance_mask_stacks(
            scene_graph,
            object_ids,
            pack_ids=False,
            allowed_overlap_fraction=100.0,
        )
        if len(object_ids) > 0
        else []
    )

    masks: list[np.ndarray] = []
    for index, frame in enumerate(frames):
        if stacks:
            stack = stacks[index]
            if stack.depth() != 1:
                raise RuntimeError(f"MaskStack for frame {frame.id} should contain one layer, got {stack.depth()}")
            mask = stack.masks[0]
        else:
            mask = torch.zeros(frame.color.shape[:2], dtype=torch.int32, device=frame.color.device)
        masks.append(mask.to(dtype=torch.int32).cpu().numpy())

    return SegmentationResult(
        frame_ids=[frame.id for frame in frames],
        pixel_object_ids=masks,
    )
