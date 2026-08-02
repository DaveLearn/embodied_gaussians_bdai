import torch

from frame_seg_init.instance_gen.edge_matcher import global_best_matching, score_edge
from frame_seg_init.instance_gen.frame_seg_scene_graph import FrameSegEdges
from frame_seg_init.instance_gen.instance_generator_new import EdgeCompatibility


def test_score_edge_methods():
    assert abs(score_edge(0.5, 0.25) - 0.2) < 1e-9
    assert score_edge(0.5, 0.25, edge_score_method="binary") == 1.0


def test_global_best_matching_supports_iou_and_binary():
    frame_a_ids = torch.tensor([1])
    frame_b_ids = torch.tensor([3])
    edges = FrameSegEdges(device="cpu")
    edges.add_for_seg(
        seg_id=torch.tensor(1),
        intersecting_ids=torch.tensor([3]),
        intersection_count=torch.tensor([10]),
        intersection_frac=torch.tensor([0.9]),
    )
    edges.add_for_seg(
        seg_id=torch.tensor(3),
        intersecting_ids=torch.tensor([1]),
        intersection_count=torch.tensor([10]),
        intersection_frac=torch.tensor([0.5]),
    )

    iou_matching = global_best_matching(edges, frame_a_ids, frame_b_ids, min_overlap_count=3, edge_score_method="iou")
    binary_matching = global_best_matching(
        edges,
        frame_a_ids,
        frame_b_ids,
        min_overlap_count=3,
        edge_score_method="binary",
    )

    assert abs(iou_matching[0][2] - 0.9 / 1.9) < 1e-6
    assert binary_matching[0][2] == 1.0


def test_candidate_scores_weight_edge_scores_by_shared_point_count():
    edges = EdgeCompatibility(device="cpu")
    edges.all_scores = torch.tensor(
        [
            [1, 3, 0, 1, 0.5, 10],
            [3, 5, 1, 2, 0.25, 20],
        ],
        dtype=torch.float32,
    )

    scores = edges.compute_candidate_scores(torch.tensor([[1, 3, 5], [1, 3, 0]]))

    assert torch.equal(scores, torch.tensor([10.0, 5.0]))
