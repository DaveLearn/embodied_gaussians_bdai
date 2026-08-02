import torch

from frame_seg_init.instance_gen.frame_seg_scene_graph import FrameSegSceneGraph
from frame_seg_init.mesh.mesh_points import get_point_frame_pixels
from frame_seg_init.mesh.mesh_with_segments import MeshWithSegments
from frame_seg_init.mesh.point_segments import SEGMENT_ID_IDX, POINT_ID_IDX


def build_edges_from_mesh_points(
    sg: FrameSegSceneGraph,
    mesh: MeshWithSegments,
    filter_occluded: bool,
):
    point_segments = mesh.point_segments.point_ids_to_segment_ids

    # for each unique frame_id in the scene graph
    for frame in sg.frames:
        # first the visible points in the frame [N, (point_idx, frame_id, u, v)]
        frame_points = get_point_frame_pixels(
            mesh.points,
            mesh.intersection_volume,
            frame,
            filter_occluded=filter_occluded,
            debug_context="build_edges_from_mesh_points",
        )

        # [N]
        frame_point_ids = frame_points[:, 0].unique()

        # then all the segments for these points
        frame_point_segments = mesh.point_segments.get_by_point_ids(frame_point_ids)

        # [M, (point_idx, segment_idx)]

        frame_visible_segment_ids, frame_visible_segment_counts = frame_point_segments[:, SEGMENT_ID_IDX].unique(return_counts=True, sorted=True)

        # loop over all segments in this frame and determine the edges

        current_frame_idx = sg.frame_segs.frame_ids == frame.id
        frame_segment_ids = sg.frame_segs.ids[current_frame_idx]

        for segment_id in frame_segment_ids:
            # get the point ids covered by this segment
            segment_point_segments = point_segments[point_segments[:, SEGMENT_ID_IDX] == segment_id]
            segment_point_ids = segment_point_segments[:, POINT_ID_IDX]

            # now query the mesh for all the other segments that intersect these point_ids
            intersecting_point_segments = point_segments[torch.isin(point_segments[:, POINT_ID_IDX], segment_point_ids)]

            if intersecting_point_segments.shape[0] == 0:
                # it may not have ended up in the mesh.
                continue

            # get unique segments
            intersecting_segment_ids, intersecting_segment_counts = intersecting_point_segments[:, SEGMENT_ID_IDX].unique(return_counts=True)

            # we also want how much of the current frame is covered by the intersecting segments.
            frame_area_idx = torch.searchsorted(frame_visible_segment_ids, intersecting_segment_ids)
            # assert that none of the idx are negative
            assert torch.all(frame_area_idx >= 0), "expected all frame_area_idx to be non-negative"
            frame_area_sums = frame_visible_segment_counts[frame_area_idx]

            this_segment_mask = intersecting_segment_ids == segment_id
            this_segment_intersection = intersecting_segment_ids[this_segment_mask]
            this_segment_intersection_area_sums = intersecting_segment_counts[this_segment_mask]
            this_segment_intersection_frac = this_segment_intersection_area_sums / frame_area_sums[this_segment_mask]

            # some sanity checks
            assert this_segment_intersection.shape[0] == 1, "expected single matching segment, got %d" % this_segment_intersection.shape[0]
            assert this_segment_intersection.shape[0] == this_segment_intersection_area_sums.shape[0] == this_segment_intersection_frac.shape[0], (
                "expected all to be the same shape"
            )

            assert this_segment_intersection_frac >= 0.98, (
                "expected intersection fraction to be around 1.0, but was %f" % this_segment_intersection_frac
            )

            sg.edges.add_for_seg(
                segment_id,
                intersecting_ids=intersecting_segment_ids,
                intersection_count=intersecting_segment_counts,
                intersection_frac=intersecting_segment_counts / frame_area_sums,
            )
