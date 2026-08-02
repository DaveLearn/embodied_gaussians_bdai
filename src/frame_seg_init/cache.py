# persist ImageSegments to disk
import json
from pathlib import Path
from typing import Dict, List

import torch
from frame_seg_init.types import Frame

from frame_seg_init.mask_stack import MaskStack
from frame_seg_init.proposals import FrameSegmenter, ImageSegment


def persist_image_segments(segments: List[ImageSegment], output_file: Path, device: str = "cuda"):
    output_file.parent.mkdir(parents=True, exist_ok=True)
    # save our segments as mask stack for easier viewing (and possible resizing)
    prediced_iou = []
    mask_stack = MaskStack(device=device)
    for id, seg in enumerate(segments):
        mask_stack.add(id, torch.tensor(seg.get_bool_mask()))
        prediced_iou.append(seg.predicted_iou)

    mask_stack.save_to_file(str(output_file.with_suffix(".masks.ms")))
    with open(str(output_file.with_suffix(".values.json")), "w", encoding="utf-8") as f:
        json.dump({"iou": prediced_iou}, f)


def load_image_segments(segment_file: Path, device: str = "cuda") -> List[ImageSegment]:
    mask_stack = MaskStack.from_file(str(segment_file.with_suffix(".masks.ms")), device=device)
    segments = []
    ids, areas = mask_stack.get_ids_and_areas()

    with open(str(segment_file.with_suffix(".values.json")), "r", encoding="utf-8") as f:
        values = json.load(f)

    for id, area, iou in zip(ids, areas, values["iou"]):
        mask = mask_stack.mask_for_id(id)
        mask_rle, area = ImageSegment.mask_data_from_pytorch_mask(mask)

        segments.append(
            ImageSegment(
                mask_rle=mask_rle,
                area=area,
                predicted_iou=iou,
            )
        )
    return segments


class PrecomputedSegmenter(FrameSegmenter):
    def __init__(self, segment_path: Path, id_to_name: Dict[int, str], device: str = "cuda"):
        self.segment_path = segment_path
        self.id_to_name = id_to_name
        self.device = device

    def segment(self, frame: Frame) -> List[ImageSegment]:
        name = self.id_to_name[frame.id]
        return load_image_segments(self.segment_path / name, device=self.device)

    def store(self, frame_id: int, segments: List[ImageSegment]):
        name = self.id_to_name[frame_id]
        persist_image_segments(segments, self.segment_path / name, device=self.device)

    def exists(self, frame_id: int) -> bool:
        name = self.id_to_name.get(frame_id)
        if name is None:
            return False
        segment_base = self.segment_path / name
        return segment_base.with_suffix(".masks.ms").exists() and segment_base.with_suffix(".values.json").exists()


class CachingSegmenter(FrameSegmenter):
    def __init__(self, precomputed_segmenter: PrecomputedSegmenter, segmenter: FrameSegmenter):
        self.precomputed_segmenter = precomputed_segmenter
        self.segmenter = segmenter

    def segment(self, frame: Frame) -> List[ImageSegment]:
        if self.precomputed_segmenter.exists(frame.id):
            return self.precomputed_segmenter.segment(frame)

        segments = self.segmenter.segment(frame)
        self.precomputed_segmenter.store(frame.id, segments)
        return segments
