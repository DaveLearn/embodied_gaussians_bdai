from typing import List, Optional

import torch

from frame_seg_init.mask_stack import MaskStack
from frame_seg_init.instance_gen.frame_seg_scene_graph import FrameSegSceneGraph


def generate_instance_mask_stacks(
    sg: FrameSegSceneGraph,
    ids: Optional[torch.Tensor] = None,
    pack_ids: bool = False,
    allowed_overlap_fraction: float = 0.01,
) -> List[MaskStack]:
    instance_ids, _ = sg.get_instance_ids_by_size()
    if ids is not None:
        instance_ids = instance_ids[torch.isin(instance_ids, ids)]

    mask_stacks = [MaskStack(device=sg.device, allowed_overlap_fraction=allowed_overlap_fraction) for _ in sg.frames]
    for unique_id, instance_id in enumerate(instance_ids):
        if instance_id < 0:
            continue
        masks = [sg.get_instance_id_mask_for_frame(int(instance_id.item()), frame.id) for frame in sg.frames]
        layer = 0
        while not all(stack.can_fit_in_layer(mask, layer) for stack, mask in zip(mask_stacks, masks)):
            layer += 1
        output_id = unique_id + 1 if pack_ids else int(instance_id.item())
        for stack, mask in zip(mask_stacks, masks):
            stack.insert_into_layer(output_id, mask, layer)

    return mask_stacks
