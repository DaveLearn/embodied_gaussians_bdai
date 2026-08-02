from dataclasses import dataclass
from typing import List, Protocol, Tuple

import numpy as np
import torch
from frame_seg_init.types import Frame


@dataclass
class ImageSegment:
    mask_rle: dict  # uncompressed coco rle
    area: int
    predicted_iou: float

    """
      returns the mask as a boolean numpy array
    """

    def get_bool_mask(self):
        rle = self.mask_rle
        """
        Compute a binary mask from an uncompressed RLE.
        "code from segment_anything/utils/amg.py"
        """
        h, w = rle["size"]
        mask = np.empty(h * w, dtype=bool)
        idx = 0
        parity = False
        for count in rle["counts"]:
            mask[idx : idx + count] = parity
            idx += count
            parity ^= True
        mask = mask.reshape(w, h)
        return mask.transpose()  # Put in C order

    @staticmethod
    def mask_data_from_pytorch_mask(
        mask_tensor: torch.Tensor,
    ) -> Tuple[dict, int]:
        """
        Encodes masks to an uncompressed RLE, in the format expected by
        pycoco tools.
         "code from segment_anything/utils/amg.py"
         and returns it along with the mask area.
        """
        # Put in fortran order and flatten h,w
        h, w = mask_tensor.shape
        tensor = mask_tensor.unsqueeze(0).permute(0, 2, 1).flatten(1)

        # Compute change indices
        diff = tensor[:, 1:] ^ tensor[:, :-1]
        change_indices = diff.nonzero()

        # Encode run length
        out = []
        for i in range(1):
            cur_idxs = change_indices[change_indices[:, 0] == i, 1]
            cur_idxs = torch.cat(
                [
                    torch.tensor([0], dtype=cur_idxs.dtype, device=cur_idxs.device),
                    cur_idxs + 1,
                    torch.tensor([h * w], dtype=cur_idxs.dtype, device=cur_idxs.device),
                ]
            )
            btw_idxs = cur_idxs[1:] - cur_idxs[:-1]
            counts = [] if tensor[i, 0] == 0 else [0]
            counts.extend(btw_idxs.detach().cpu().tolist())
            out.append({"size": [h, w], "counts": counts})

        mask_rle = out[0]
        area = int(torch.sum(mask_tensor).item())
        return mask_rle, area


class SegmenterModel(Protocol):
    """
    Takes an image (H, W, 3) [0,1], and returns a list of ImageSegment objects for the image that aims to cover all pixels
    """

    def segment_everything(self, image: torch.Tensor) -> List[ImageSegment]: ...


class FrameSegmenter(Protocol):
    def segment(self, frame: Frame) -> List[ImageSegment]: ...
