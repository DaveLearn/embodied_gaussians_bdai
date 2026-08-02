import torch

from frame_seg_init.instance_gen.frame_seg_scene_graph import FrameSegSceneGraph
from frame_seg_init.mesh.mesh_points import get_frame_points, get_point_idx_for_mask
from frame_seg_init.mesh.mesh_with_segments import MeshWithSegments


def shrink_mask(mask: torch.Tensor) -> torch.Tensor:
    kernel = torch.ones(1, 1, 3, 3, device=mask.device)
    neighbors = torch.nn.functional.conv2d(mask.float().unsqueeze(0).unsqueeze(0), kernel, padding=1)
    return (neighbors == 9).squeeze(0).squeeze(0)


def assign_segments_to_mesh_points(
    sg: FrameSegSceneGraph,
    mesh: MeshWithSegments,
    filter_occluded: bool,
):
    # build a lookup of points to frame uv pixels
    frame_points = get_frame_points(
        mesh.points,
        mesh.intersection_volume,
        sg.frames,
        filter_occluded=filter_occluded,
        debug_context="assign_segments_to_mesh_points",
    )

    # step 1: assign segments to points
    for frame, frame_masks in zip(sg.frames, sg.frame_masks):
        # this helps exclude some bleed where the the depth doesn't match the mesh.
        assert frame.depth is not None, "Requires depth"

        # determine which segments and segment_ids belong to this frame
        current_frame_idx = sg.frame_segs.frame_ids == frame.id
        frame_segment_ids = sg.frame_segs.ids[current_frame_idx]
        frame_segment_idxs = sg.frame_segs.seg_idx[current_frame_idx]

        # loop over each segment in this frame and assign segments to points
        for frame_segment_id, segment_idx in zip(
            frame_segment_ids,
            frame_segment_idxs,
        ):
            # for each mask determine unoccluded points visible to the camera within the segment mask
            seg_mask = frame_masks.mask_for_id(int(segment_idx.item()))

            # shrink the mask to prevent bleed
            seg_mask = shrink_mask(seg_mask)

            # check mask intersection with points.
            point_ids = get_point_idx_for_mask(frame_points, seg_mask, frame.id)
            if len(point_ids) != 0:
                mesh.assign_segment_to_points(point_ids, frame_segment_id)
