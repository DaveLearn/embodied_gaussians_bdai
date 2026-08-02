from typing import List, Tuple
import torch
import logging
from frame_seg_init.types import Frame
from frame_seg_init.frame_segments import FrameSegments
from frame_seg_init.mask_stack import MaskStack
from frame_seg_init.proposals import ImageSegment

logger = logging.getLogger(__name__)


class FrameSegEdges:
    def __init__(self, device: str | torch.device = "cuda"):
        self.frame_seg_ids: torch.Tensor = torch.empty((0, 2), dtype=torch.int32, device=device)
        self.intersection_count: torch.Tensor = torch.empty((0,), dtype=torch.int32, device=device)
        self.intersection_frac: torch.Tensor = torch.empty((0,), dtype=torch.float, device=device)

    def add_for_seg(
        self,
        seg_id: torch.Tensor,
        intersecting_ids: torch.Tensor,
        intersection_count: torch.Tensor,
        intersection_frac: torch.Tensor,
    ):
        assert len(seg_id.shape) == 0, "expected a single seg_id"
        assert intersecting_ids.shape[0] == intersection_count.shape[0] == intersection_frac.shape[0], (
            f"intersecting ids, counts, and fractions must have equal lengths: {intersection_count.shape[0]}, {intersection_frac.shape[0]}"
        )

        # turn seg_id and intersecting_ids into a tensor of pairs with the lowest id as the first element
        # Note: projecting segA onto segB may have different intersections than projecting segB onto segA, but here we just take the first mapping
        pairs = torch.stack([seg_id.expand((intersecting_ids.shape[0],)), intersecting_ids], dim=1)  # [N, 2]
        # pairs = torch.sort(pairs, dim=1).values

        # mask out the pairs that already exist in self.frame_seg_ids
        matches = pairs.unsqueeze(1) == self.frame_seg_ids.unsqueeze(0)  # [N, 1, 2] == [1, M, 2] -> [N, M, 2]
        matches = matches.all(dim=2)  # [N, M]
        matches = matches.any(dim=1)  # [N]
        unique_mask = ~matches

        # add the unique pairs
        self.frame_seg_ids = torch.cat([self.frame_seg_ids, pairs[unique_mask]])
        self.intersection_count = torch.cat([self.intersection_count, intersection_count[unique_mask]])
        self.intersection_frac = torch.cat([self.intersection_frac, intersection_frac[unique_mask]])


class FrameSegSceneGraph:
    def __init__(self, device: str | torch.device = "cuda"):
        self.device = device
        self.frame_segs = FrameSegments(device=device)
        self.frames: List[Frame] = []
        self.frame_masks: List[MaskStack] = []
        self.edges: FrameSegEdges = FrameSegEdges(device=device)
        self.object_instance_ids: torch.Tensor = torch.empty(0, dtype=torch.int32, device=device)

    def add_frame_segs(self, frame: Frame, segments: List[ImageSegment]):

        # check if we have done this frame, if so reuse
        existing_frame = next((f for f in self.frames if f.id == frame.id), None)
        if existing_frame is not None:
            raise ValueError("Frame already exists in the scene graph")

        self.frames.append(frame)
        masks = MaskStack(
            device=self.device,
            allowed_overlap_fraction=0.01,
        )
        seg_ids = torch.arange(len(segments), dtype=torch.int32, device=self.device)

        areas = []
        for i, seg in enumerate(segments):
            masks.add(i, torch.tensor(seg.get_bool_mask()))
            areas.append(seg.area)

        self.frame_masks.append(masks)

        self.frame_segs.add(
            seg_ids,
            torch.ones((seg_ids.shape[0],), dtype=torch.int32) * frame.id,
            torch.tensor(areas, dtype=torch.int32, device=self.device),
        )

    def get_instance_id_mask_for_frame(self, instance_id: int, frame_id: int) -> torch.Tensor:
        # get seg ids
        seg_mask = self.frame_segs.instance_ids == instance_id
        frame_mask = self.frame_segs.frame_ids == frame_id
        seg_ids = self.frame_segs.seg_idx[seg_mask & frame_mask]

        # get frame idx for id
        frame_idx = [f.id for f in self.frames].index(frame_id)

        mask = self.frame_masks[frame_idx].mask_for_id(-10000)

        # mask = self.frame_masks[frame_idx].mask_for_id(int(seg_ids[0].item()))
        for seg_id in seg_ids:
            mask |= self.frame_masks[frame_idx].mask_for_id(int(seg_id.item()))

        return mask

    # returns the pixel count of the instance across all frames
    def get_instance_id_counts(self, instance_ids: torch.Tensor) -> torch.Tensor:
        counts = torch.zeros((len(instance_ids),), dtype=torch.int32, device=self.device)
        for i, instance_id in enumerate(instance_ids):
            for frame in self.frames:
                counts[i] += self.get_instance_id_mask_for_frame(int(instance_id.item()), frame.id).int().sum()

        return counts

    # get instance_ids by size
    def get_instance_ids_by_size(self) -> Tuple[torch.Tensor, torch.Tensor]:
        instance_ids = self.frame_segs.instance_ids.unique()
        instance_counts = self.get_instance_id_counts(instance_ids)
        sorted_counts, sorted_idxs = torch.sort(instance_counts, descending=True)
        return instance_ids[sorted_idxs], sorted_counts
