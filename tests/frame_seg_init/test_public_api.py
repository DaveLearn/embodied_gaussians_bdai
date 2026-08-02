from pathlib import Path

import numpy as np
import pytest
import torch

from frame_seg_init import GraphParams, ObservationFrame, Observations, RuntimeConfig, Workspace, segment_scene
from frame_seg_init.types import Frame


def test_frame_from_observation_uses_requested_device() -> None:
    observation = ObservationFrame(
        id=4,
        name="camera",
        color=np.zeros((3, 5, 3), dtype=np.float32),
        depth=np.ones((3, 5), dtype=np.float32),
        X_WV=np.eye(4, dtype=np.float32),
        K=np.eye(3, dtype=np.float32),
    )

    frame = Frame.from_observation(observation, "cpu")

    assert frame.color.device == torch.device("cpu")
    assert frame.depth is not None and frame.depth.device == torch.device("cpu")
    assert (frame.h, frame.w) == (3, 5)


def test_runtime_cache_directory_is_explicit(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="cache_dir must be supplied explicitly"):
        RuntimeConfig().resolved_cache_dir()

    assert RuntimeConfig(cache_dir=tmp_path).resolved_cache_dir() == tmp_path.resolve()


def test_observations_reject_duplicate_frame_ids() -> None:
    frame = ObservationFrame(
        id=1,
        name="camera",
        color=np.zeros((2, 2, 3), dtype=np.float32),
        X_WV=np.eye(4, dtype=np.float32),
        K=np.eye(3, dtype=np.float32),
    )

    with pytest.raises(ValueError, match="unique"):
        Observations(frames=[frame, frame])


def test_empty_scene_does_not_require_model_or_cache() -> None:
    result = segment_scene(
        Observations(frames=[]),
        Workspace(
            ground_points=np.empty((0, 3), dtype=np.float32),
            ground_plane=(0.0, 0.0, 1.0, 0.0),
        ),
        params=GraphParams(),
        runtime=RuntimeConfig(device="cpu"),
    )

    assert result.frame_ids == []
    assert result.pixel_object_ids == []
