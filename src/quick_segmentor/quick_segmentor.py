# Copyright (c) 2025 Boston Dynamics AI Institute LLC. All rights reserved.

import os
import numpy as np
from isegm_gui import run_interactive_segmentor, load_model
import urllib.request


class QuickSegmentor:
    def __init__(self, device="cuda"):
        self.device = device
        
        # download the model if it doesn't exist
        base_link = "https://github.com/hkchengrex/Cutie/releases/download/v1.0/"
        checkpoint = "coco_lvis_h18_itermask.pth"
        checkpoint_path = f"weights/{checkpoint}"
        os.makedirs(os.path.dirname(checkpoint_path), exist_ok=True)
        if not os.path.exists(checkpoint_path):
            print(f"Downloading {checkpoint} to {checkpoint_path}...")
            urllib.request.urlretrieve(f"{base_link}/{checkpoint}", checkpoint_path)
        else:
            print(f"Using existing {checkpoint} at {checkpoint_path}")

        self.model = load_model(checkpoint, device=self.device)

    def segment_with_gui(self, image: np.ndarray) -> np.ndarray | None:
        """Blocks and returns the mask. Mask is None if no points are selected.
        Mask is of type bool where True is foreground and False is background.
        """

        assert image.ndim == 3, "Image must be 3D"
        assert image.shape[-1] == 3, "Image must be RGB"
        assert image.dtype == np.uint8, "Image must be uint8"

        gui = run_interactive_segmentor(self.model, device=self.device)
        gui.update_image(image)
        gui.mainloop()
        mask = gui.get_mask()
        try:
            gui.master.destroy()
        except Exception:
            pass
        return mask
