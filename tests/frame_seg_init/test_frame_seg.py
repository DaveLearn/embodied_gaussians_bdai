import torch

from frame_seg_init.frame_segments import FrameSegments


def test_frame_segments_adds_segment_identity_and_area():
    target_device = torch.device("cuda", index=0)
    frame_segments = FrameSegments(device=target_device)
    segment_indices = torch.tensor([1, 2, 3])
    frame_ids = torch.tensor([1, 1, 1])
    areas = torch.tensor([10, 20, 30])

    new_ids = frame_segments.add(segment_indices, frame_ids, areas)

    assert torch.equal(frame_segments.ids.cpu(), new_ids.cpu())
    assert torch.equal(frame_segments.frame_ids.cpu(), frame_ids)
    assert torch.equal(frame_segments.seg_idx.cpu(), segment_indices)
    assert torch.equal(frame_segments.area.cpu(), areas)
    assert torch.equal(frame_segments.instance_ids.cpu(), new_ids.cpu())
    assert frame_segments.ids.device == target_device
    assert frame_segments.frame_ids.device == target_device
    assert frame_segments.seg_idx.device == target_device
    assert frame_segments.area.device == target_device
    assert frame_segments.instance_ids.device == target_device


def test_frame_segments_returns_areas_in_requested_id_order():
    frame_segments = FrameSegments(device="cpu")
    ids = frame_segments.add(
        torch.tensor([1, 2, 3]),
        torch.tensor([1, 1, 2]),
        torch.tensor([10, 20, 30]),
    )

    assert torch.equal(frame_segments.get_areas_by_ids(ids[[2, 0, 1]]), torch.tensor([30, 10, 20]))
