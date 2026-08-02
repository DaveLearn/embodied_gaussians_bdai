import os
import PIL.Image
import torch
import numpy as np


def key_for_id(id):
    return id + 1


def id_from_key(key):
    return key - 1


class MaskStack:
    def __init__(self, device: str | torch.device = "cuda", allowed_overlap_fraction=0.0):
        self.masks = torch.empty((0, 720, 1280), dtype=torch.int16, device=device)
        self.device = device
        self.allowed_overlap_fraction = allowed_overlap_fraction

    def depth(self):
        return self.masks.shape[0]

    def get_ids_and_areas(self):
        keys, counts = self.masks.unique(return_counts=True)
        valid_mask = keys != 0
        return id_from_key(keys[valid_mask]), counts[valid_mask]

    # takes id and returns H, W bool mask for the given id
    def mask_for_id(self, id: int) -> torch.Tensor:
        return (self.masks == key_for_id(id)).sum(dim=0).bool()

    def can_fit_in_layer(self, mask: torch.Tensor, layer_idx: int) -> bool:
        if layer_idx >= self.masks.shape[0]:
            return True

        maskd = mask.int().to(self.device)

        # calc  max pixel overlap
        mask_pixels = maskd.sum()

        all_ids, all_counts = self.masks[layer_idx].unique(return_counts=True, sorted=True)
        intersect_ids, intersect_counts = self.masks[layer_idx][maskd.bool()].unique(return_counts=True)

        valid_intersect_mask = intersect_ids != 0
        intersect_ids = intersect_ids[valid_intersect_mask]
        intersect_counts = intersect_counts[valid_intersect_mask]

        # find itersect_ids that are in all_ids
        intersect_ids_idx = torch.searchsorted(all_ids, intersect_ids, out_int32=True)
        intersect_full_counts = all_counts[intersect_ids_idx]

        # get the max intersection fraction of existing
        if intersect_counts.shape[0] > 0:
            max_intersect_fraction = (intersect_counts.float() / intersect_full_counts.float()).max(dim=0).values
        else:
            max_intersect_fraction = 0.0

        if max_intersect_fraction > self.allowed_overlap_fraction:
            return False

        # get fraction of current mask that is covered by existing
        current_mask_fraction = intersect_counts.float().sum() / mask_pixels.float()

        if current_mask_fraction > self.allowed_overlap_fraction:
            return False

        return True

    def insert_into_layer(self, id: int, mask: torch.Tensor, layer_idx: int):
        num_layers = self.masks.shape[0]
        assert layer_idx <= num_layers, f"Invalid layer index: {layer_idx} <= {num_layers}"
        assert id >= 0, f"Invalid id: {id}, must be >=0"

        if self.masks.shape[0] == 0:
            self.masks = (mask.int() * key_for_id(id)).unsqueeze(0).to(self.device)
            return

        assert mask.shape == self.masks.shape[1:], f"Mask shape mismatch: {mask.shape} != {self.masks.shape}"

        if layer_idx == num_layers:
            self.masks = torch.cat(
                [
                    self.masks,
                    (mask.int() * key_for_id(id)).unsqueeze(0).to(self.device),
                ],
                dim=0,
            )
            return

        self.masks[layer_idx, mask] = key_for_id(id)

    # takes a mask (torch.bool) and an id and adds the mask to the stack
    # returns which layer it was added to (0-indexed)
    def add(self, id: int, mask: torch.Tensor) -> int:
        # use our first mask as layer 1
        if self.masks.shape[0] == 0:
            self.insert_into_layer(id, mask, 0)
            return 0

        # find best layer to add to based on intersection counts
        for layer_idx in range(self.masks.shape[0]):
            if self.can_fit_in_layer(mask, layer_idx):
                self.insert_into_layer(id, mask, layer_idx)
                return layer_idx

        best_layer = self.masks.shape[0]
        self.insert_into_layer(id, mask, best_layer)
        return best_layer

    def export_layer_as_image(self, layer_idx: int) -> PIL.Image.Image:
        assert layer_idx < self.masks.shape[0], f"Invalid layer index: {layer_idx} >= {self.masks.shape[0]}"

        # if we can be palletized, do so for easier debugging
        max_val = self.masks.max().item()
        if max_val < 256:
            palette = []
            for i in range(256):  # 256 colors for 8-bit palette
                # Generate a color for each unique value by multiplying primes
                r = (i * 73) % 256
                g = (i * 127) % 256
                b = (i * 179) % 256
                palette.extend([r, g, b])

            img = PIL.Image.fromarray(self.masks[layer_idx].cpu().to(dtype=torch.uint8).numpy())
            img.putpalette(palette)
        else:
            img = PIL.Image.fromarray(self.masks[layer_idx].cpu().to(dtype=torch.int32).numpy())
        return img

    def update_layer_from_image(self, layer_idx: int, layer_img: PIL.Image.Image):
        assert layer_img.mode == "I" or layer_img.mode == "P", f"Invalid image mode: {layer_img.mode}"

        img = layer_img
        mask = torch.tensor(np.array(img), dtype=torch.int16, device=self.device)

        if self.masks.shape[0] == 0:
            self.masks = mask.unsqueeze(0)
        else:
            while (self.masks.shape[0] - 1) < layer_idx:
                self.masks = torch.cat([self.masks, torch.zeros_like(self.masks[0]).unsqueeze(0)], dim=0)
            self.masks[layer_idx] = mask

    def save_to_file(self, output_file: str):
        # open output_file_path for writing
        with open(output_file, "w") as f:
            for i in range(self.masks.shape[0]):
                layer_name = f"{os.path.basename(output_file)}_{i}.png"
                img = self.export_layer_as_image(i)
                img.save(os.path.join(os.path.dirname(output_file), layer_name), format="PNG")
                f.write(f"{layer_name}\n")

    def load_from_file(self, input_file: str):
        # open inputfile and read all lines
        with open(input_file, "r") as f:
            lines = f.readlines()
            for i, line in enumerate(lines):
                layer_name = line.strip()
                full_name = os.path.join(os.path.dirname(input_file), layer_name)
                img = PIL.Image.open(full_name)
                self.update_layer_from_image(i, img)

    @staticmethod
    def from_file(input_file: str, device="cuda", allowed_overlap_fraction=0.0):
        mask_stack = MaskStack(device, allowed_overlap_fraction)
        mask_stack.load_from_file(input_file)
        return mask_stack
