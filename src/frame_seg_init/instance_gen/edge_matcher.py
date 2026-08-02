from typing import List, Literal, Tuple

import numpy as np
import torch

from frame_seg_init.instance_gen.frame_seg_scene_graph import FrameSegEdges

EdgeScoreMethod = Literal["iou", "binary"]
EdgeMatch = Tuple[int, int, float]


def score_edge(
    overlap_a: float,
    overlap_b: float,
    edge_score_method: EdgeScoreMethod = "iou",
) -> float:
    if edge_score_method == "binary":
        return 1.0
    return overlap_a / (1 + overlap_a / overlap_b - overlap_a)


def _build_score_tables(
    edges: FrameSegEdges,
    frame_a_segment_ids: torch.Tensor,
    frame_b_segment_ids: torch.Tensor,
    min_overlap_count: int,
    edge_score_method: EdgeScoreMethod,
) -> Tuple[np.ndarray, np.ndarray]:
    costs = np.zeros((len(frame_a_segment_ids), len(frame_b_segment_ids)))
    scores = np.zeros_like(costs)

    for a_idx, seg_a in enumerate(frame_a_segment_ids):
        for b_idx, seg_b in enumerate(frame_b_segment_ids):
            a_mask = (edges.frame_seg_ids[:, 0] == seg_a) & (edges.frame_seg_ids[:, 1] == seg_b)
            b_mask = (edges.frame_seg_ids[:, 0] == seg_b) & (edges.frame_seg_ids[:, 1] == seg_a)
            if not a_mask.any() or not b_mask.any():
                continue

            overlap_a = float(edges.intersection_frac[a_mask].min())
            overlap_b = float(edges.intersection_frac[b_mask].min())
            count = float(edges.intersection_count[a_mask | b_mask].min())
            score = score_edge(overlap_a, overlap_b, edge_score_method)

            cost = -score if count >= min_overlap_count else 0.0

            costs[a_idx, b_idx] = cost
            scores[a_idx, b_idx] = score

    return costs, scores


def _edge_tuple(
    frame_a_segment_ids: torch.Tensor,
    frame_b_segment_ids: torch.Tensor,
    a_idx: int,
    b_idx: int,
    scores: np.ndarray,
) -> EdgeMatch:
    return (
        int(frame_a_segment_ids[a_idx]),
        int(frame_b_segment_ids[b_idx]),
        float(scores[a_idx, b_idx]),
    )


def global_best_matching(
    edges: FrameSegEdges,
    frame_a_segment_ids: torch.Tensor,
    frame_b_segment_ids: torch.Tensor,
    min_overlap_count: int,
    edge_score_method: EdgeScoreMethod,
) -> List[EdgeMatch]:
    if frame_a_segment_ids.numel() == 0 or frame_b_segment_ids.numel() == 0:
        return []

    costs, scores = _build_score_tables(
        edges,
        frame_a_segment_ids,
        frame_b_segment_ids,
        min_overlap_count,
        edge_score_method,
    )
    best_b_for_a = np.argmin(costs, axis=1)
    best_a_for_b = np.argmin(costs, axis=0)
    unmatched_b = set(range(costs.shape[1])) - set(best_b_for_a)

    matching = [
        _edge_tuple(frame_a_segment_ids, frame_b_segment_ids, a_idx, int(b_idx), scores)
        for a_idx, b_idx in enumerate(best_b_for_a)
        if costs[a_idx, b_idx] < 0
    ]
    matching.extend(
        _edge_tuple(frame_a_segment_ids, frame_b_segment_ids, int(best_a_for_b[b_idx]), b_idx, scores)
        for b_idx in unmatched_b
        if costs[best_a_for_b[b_idx], b_idx] < 0
    )
    return matching
