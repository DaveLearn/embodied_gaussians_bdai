import torch

from frame_seg_init.proposals import ImageSegment
from frame_seg_init.sam import SAMSegmenter, flatten_overlapping_segments


class _FakeSAMModel:
    def __init__(self):
        self.device = None
        self.is_eval = False

    def to(self, device: str):
        self.device = device
        return self

    def eval(self):
        self.is_eval = True
        return self


def test_sam_segmenter_moves_model_to_requested_device(monkeypatch):
    model = _FakeSAMModel()
    monkeypatch.setattr(SAMSegmenter, "build_model", staticmethod(lambda _checkpoint: model))

    segmenter = SAMSegmenter(device="cpu")

    assert segmenter.model is model
    assert model.device == torch.device("cpu")
    assert model.is_eval


def _segment_from_mask(mask: torch.Tensor, predicted_iou: float) -> ImageSegment:
    rle, area = ImageSegment.mask_data_from_pytorch_mask(mask)
    return ImageSegment(
        mask_rle=rle,
        area=area,
        predicted_iou=predicted_iou,
    )


def test_flatten_overlapping_segments_keeps_one_mask_per_pixel():
    low_iou_mask = torch.tensor(
        [
            [True, True, False],
            [True, True, False],
            [False, False, False],
        ]
    )
    high_iou_mask = torch.tensor(
        [
            [False, False, False],
            [False, True, True],
            [False, True, True],
        ]
    )

    flattened = flatten_overlapping_segments(
        [
            _segment_from_mask(high_iou_mask, predicted_iou=0.9),
            _segment_from_mask(low_iou_mask, predicted_iou=0.5),
        ]
    )

    assert len(flattened) == 2
    flattened_masks = [torch.tensor(segment.get_bool_mask()) for segment in flattened]
    assert not (flattened_masks[0] & flattened_masks[1]).any()
    assert flattened_masks[0].sum().item() == 3
    assert flattened_masks[1].sum().item() == 4
    assert flattened_masks[1][1, 1]
