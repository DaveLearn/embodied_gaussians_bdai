import torch

POINT_ID_IDX = 0
SEGMENT_ID_IDX = 1


class PointSegments:
    def __init__(self, device: str | torch.device = "cuda"):
        self.point_ids_to_segment_ids = torch.empty((0, 2), dtype=torch.int32, device=device)
        self.device = device

    # takes 2 N tensors
    def add(self, point_ids: torch.Tensor, segment_ids: torch.Tensor):
        assert len(point_ids.shape) == 1
        assert len(segment_ids.shape) == 1
        assert point_ids.shape[0] == segment_ids.shape[0]
        new_links = torch.stack((point_ids, segment_ids), dim=1).to(self.device)
        self.point_ids_to_segment_ids = torch.cat([self.point_ids_to_segment_ids, new_links])

    # takes a [N] tensor of point ids and returns [N,(triangle_id, segment_id, area)]
    # the resulting tuples are not necessarily in the order of triangle_ids queried
    def get_by_point_ids(self, point_ids: torch.Tensor) -> torch.Tensor:
        # Efficient lookup for multiple point ids
        mask = torch.isin(
            self.point_ids_to_segment_ids[:, POINT_ID_IDX],
            point_ids.to(self.device),
        )
        return self.point_ids_to_segment_ids[mask]
