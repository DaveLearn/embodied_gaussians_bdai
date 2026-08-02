from typing import Union
import torch


class FrameSegments:
    def __init__(self, device: Union[str, torch.device, int] = "cuda"):
        self.ids = torch.tensor([], dtype=torch.int32, device=device)
        self.seg_idx = torch.tensor([], dtype=torch.int16, device=device)  # [N] Segment index within the SegmentedFrame.segments
        self.frame_ids = torch.tensor([], dtype=torch.int32, device=device)  # [N] Frame.id to which the segment belongs
        self.area = torch.empty((0,), dtype=torch.int32, device=device)  # [N] Area of the segment in pixels
        self.instance_ids = torch.tensor([], dtype=torch.int32, device=device)  # [N] Instance id of the segment
        self.device = device

    def add(
        self,
        seg_idxs: torch.Tensor,
        frame_ids: torch.Tensor,
        area: torch.Tensor,
    ) -> torch.Tensor:
        assert len(seg_idxs) == len(frame_ids) == len(area), f"Length mismatch: {len(seg_idxs)} != {len(frame_ids)} != {len(area)}"

        new_ids = self._gen_frame_seg_ids(len(seg_idxs))
        self.ids = torch.cat([self.ids, new_ids])
        self.frame_ids = torch.cat([self.frame_ids, frame_ids.to(self.device)])
        self.seg_idx = torch.cat([self.seg_idx, seg_idxs.to(self.device)])
        self.area = torch.cat([self.area, area.to(self.device)])
        self.instance_ids = torch.cat([self.instance_ids, new_ids])

        return new_ids

    def get_areas_by_ids(self, ids: torch.Tensor) -> torch.Tensor:
        id_idx = torch.searchsorted(self.ids, ids.to(self.ids.device), out_int32=True)
        matched_areas = self.area[id_idx]
        return matched_areas

    def _get_largest_id(self) -> int:
        return int(self.ids.max().item()) if len(self.ids) > 0 else 0

    def _gen_frame_seg_ids(self, count: int) -> torch.Tensor:
        offset = self._get_largest_id() + 1
        ids = torch.arange(offset, offset + count, dtype=torch.int32, device=self.device)
        return ids
