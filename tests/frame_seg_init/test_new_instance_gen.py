import torch

from frame_seg_init.instance_gen.instance_generator_new import EdgeCompatibility, InstanceCandidates


def _edges(rows: list[list[int]]) -> EdgeCompatibility:
    edges = EdgeCompatibility(device="cpu")
    edges.compatible = torch.tensor(rows, dtype=torch.int)
    return edges


def test_unused_edge_finder_finds_unused_edge():
    edges = _edges([[1, 3, 0, 1], [2, 3, 0, 1]])
    candidates = InstanceCandidates(2, device="cpu")
    candidates.candidates = torch.tensor([[1, 3]])

    assert candidates.select_unused_edge(edges) == (2, 3, 0, 1)


def test_unused_edge_finder_returns_none_if_no_unused_edge():
    edges = _edges([[1, 3, 0, 1], [2, 3, 0, 1]])
    candidates = InstanceCandidates(2, device="cpu")
    candidates.candidates = torch.tensor([[1, 3], [2, 3]])

    assert candidates.select_unused_edge(edges) is None


def test_compatible_edge_finder_finds_edge_for_empty_frame():
    edges = _edges([[1, 3, 0, 1], [2, 3, 0, 1], [3, 5, 1, 2]])
    candidates = InstanceCandidates(3, device="cpu")

    assert candidates.select_compatible_edge(torch.tensor([2, 3, 0]), edges) == (3, 5, 1, 2)


def test_complete_instance_follows_compatible_edges():
    edges = _edges([[1, 3, 0, 1], [2, 3, 0, 1], [3, 5, 1, 2], [5, 6, 2, 3], [6, 7, 3, 4]])
    candidates = InstanceCandidates(5, device="cpu")

    completed = candidates.complete_instance(torch.tensor([0, 3, 5, 0, 0]), edges)

    assert torch.equal(completed, torch.tensor([1, 3, 5, 6, 7]))


def test_build_from_edges_builds_all_instances():
    edges = _edges([[1, 3, 0, 1], [2, 3, 0, 1], [3, 5, 1, 2], [5, 6, 2, 3], [6, 7, 3, 4], [2, 6, 0, 3]])
    candidates = InstanceCandidates(5, device="cpu")

    candidates.build_from_edges(edges)

    assert len(candidates.candidates) == 2
    assert torch.any((candidates.candidates == torch.tensor([1, 3, 5, 6, 7])).all(dim=1))
    assert torch.any((candidates.candidates == torch.tensor([2, 3, 5, 6, 7])).all(dim=1))


def test_subset_detection():
    candidates = InstanceCandidates(5, device="cpu")
    candidates.candidates = torch.tensor([[1, 3, 5, 6, 7], [2, 3, 0, 6, 7]])

    assert candidates.is_subset_of_candidate(torch.tensor([1, 3, 0, 0, 0]))
    assert candidates.is_subset_of_candidate(torch.tensor([0, 0, 0, 6, 7]))
    assert not candidates.is_subset_of_candidate(torch.tensor([1, 3, 5, 6, 8]))
