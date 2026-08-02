from typing import List
import torch
from frame_seg_init.proposals import ImageSegment, SegmenterModel
from pathlib import Path
import logging


def check_sam_dependency():
    global sam_model_registry, SamAutomaticMaskGenerator
    try:
        from segment_anything import sam_model_registry, SamAutomaticMaskGenerator
    except ImportError as error:
        raise ImportError("Failed to import segment-anything and its runtime dependencies") from error


check_sam_dependency()


logger = logging.getLogger(__name__)


class SAMSegmenter(SegmenterModel):
    def __init__(
        self,
        flatten_masks: bool = False,
        device: str = "cuda",
        checkpoint_path: Path | None = None,
    ):
        self.flatten_masks = flatten_masks
        self.device = torch.device(device)
        self.model = self.build_model(checkpoint_path)
        self.model.to(self.device)
        self.model.eval()

    @staticmethod
    def build_model(checkpoint_path: Path | None = None):
        if checkpoint_path is None:
            raise ValueError("checkpoint_path must be supplied explicitly")
        checkpoint_path = checkpoint_path.expanduser().resolve()
        checkpoint_url = "https://dl.fbaipublicfiles.com/segment_anything/sam_vit_h_4b8939.pth"
        model_type = "vit_h"

        if not checkpoint_path.is_file():
            raise FileNotFoundError(f"SAM checkpoint not found at {checkpoint_path}. Download it explicitly from {checkpoint_url}.")

        logger.info("Loading model %s from %s", model_type, checkpoint_path)

        return sam_model_registry[model_type](checkpoint=str(checkpoint_path))

    @torch.no_grad()
    def segment_everything(self, image: torch.Tensor) -> List[ImageSegment]:
        segmenter = SamAutomaticMaskGenerator(
            self.model,
            output_mode="uncompressed_rle",
            points_per_side=64,
            pred_iou_thresh=0.7,
            crop_n_layers=0,
        )
        masks = segmenter.generate((image * 255).to(dtype=torch.uint8).cpu().numpy())

        segs = [
            ImageSegment(
                mask_rle=mask["segmentation"],
                area=mask["area"],
                predicted_iou=mask["predicted_iou"],
            )
            for mask in masks
        ]

        if self.flatten_masks:
            segs = flatten_overlapping_segments(segs)

        return segs


def flatten_overlapping_segments(segments: List[ImageSegment]) -> List[ImageSegment]:
    # Match the 2D mask interface used by SegmentAnything3D, SAI3D, and
    # MaskClustering: paint overlapping proposal masks into a single per-pixel
    # id image, using mask quality order so higher-IoU masks win conflicts.
    if len(segments) <= 1:
        return segments

    segment_masks = [torch.tensor(segment.get_bool_mask(), dtype=torch.bool) for segment in segments]
    label_image = torch.zeros_like(segment_masks[0], dtype=torch.int32)
    sorted_indices = sorted(range(len(segments)), key=lambda idx: segments[idx].predicted_iou)

    for label_id, segment_idx in enumerate(sorted_indices, start=1):
        mask = segment_masks[segment_idx]
        if mask.shape != label_image.shape:
            raise ValueError(f"Segment mask shape {mask.shape} does not match {label_image.shape}")
        label_image[mask] = label_id

    flattened_segments = []
    for label_id, segment_idx in enumerate(sorted_indices, start=1):
        mask = label_image == label_id
        if not mask.any():
            continue

        source_segment = segments[segment_idx]
        rle, area = ImageSegment.mask_data_from_pytorch_mask(mask)
        flattened_segments.append(
            ImageSegment(
                mask_rle=rle,
                area=area,
                predicted_iou=source_segment.predicted_iou,
            )
        )

    return flattened_segments
