import json
import os
from pathlib import Path

import numpy as np
from frame_seg_init.cache import load_image_segments, persist_image_segments
from frame_seg_init.proposals import ImageSegment
from tempfile import TemporaryDirectory


def test_persist_image_segments():
    with open(Path(__file__).parent / "test_segs.json", "r") as f:
        test_seg_values = json.load(f)

    segments = [
        ImageSegment(
            mask_rle=value["mask_rle"],
            area=value["area"],
            predicted_iou=value["predicted_iou"],
        )
        for value in test_seg_values
    ]

    # get a temporary directory to save the file using the operating systems
    with TemporaryDirectory() as tmpdir:
        filename = os.path.join(tmpdir, "test_segs")

        persist_image_segments(segments, Path(filename))
        assert Path(filename).with_suffix(".masks.ms").exists()
        assert Path(filename).with_suffix(".values.json").exists()

        # load the file and compare with current segments
        loaded_segments = load_image_segments(Path(filename))
        assert len(loaded_segments) == len(segments)
        for idx, s in enumerate(segments):
            loaded = loaded_segments[idx]
            assert np.array_equal(s.get_bool_mask(), loaded.get_bool_mask())

            assert s.predicted_iou == loaded.predicted_iou
            assert s.area == loaded.area
