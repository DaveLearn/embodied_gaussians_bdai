import numpy as np
import pytest

from embodied_gaussians.segmentation import overlay_instances


def test_overlay_instances_converts_bgr_and_preserves_background() -> None:
    image = np.array([[[10, 20, 30], [40, 50, 60]]], dtype=np.uint8)
    labels = np.array([[0, 1]], dtype=np.int32)

    overlay = overlay_instances(image, "bgr", labels, alpha=1.0)

    assert np.array_equal(overlay[0, 0], [30, 20, 10])
    assert np.array_equal(overlay[0, 1], [73, 127, 179])


def test_overlay_instances_validates_alpha() -> None:
    with pytest.raises(ValueError, match="between zero and one"):
        overlay_instances(np.zeros((1, 1, 3)), "rgb", np.zeros((1, 1)), alpha=1.1)
