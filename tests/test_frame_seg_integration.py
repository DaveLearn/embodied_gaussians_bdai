from pathlib import Path
from types import SimpleNamespace
from typing import Any

import numpy as np
import pytest

from embodied_gaussians.scene_builders.domain import Ground, MaskedPosedImageAndDepth, PosedImageAndDepth
from embodied_gaussians.segmentation import frame_seg
from embodied_gaussians.segmentation.frame_seg import (
    FrameSegConfig,
    InstanceSegmentation,
    build_frame_seg_observations,
    datapoints_for_instance,
    segment_datapoints,
)


def _datapoint(*, image_format: str = "bgr") -> MaskedPosedImageAndDepth:
    return MaskedPosedImageAndDepth(
        mask=np.ones((2, 2), dtype=bool),
        X_WC=np.eye(4, dtype=np.float32),
        K=np.array([[10.0, 0.0, 1.0], [0.0, 10.0, 1.0], [0.0, 0.0, 1.0]], dtype=np.float32),
        image=np.array(
            [
                [[10, 20, 30], [40, 50, 60]],
                [[70, 80, 90], [100, 110, 120]],
            ],
            dtype=np.uint8,
        ),
        format=image_format,  # type: ignore[arg-type]
        depth=np.array([[1000, 2000], [0, 500]], dtype=np.float32),
        depth_scale=0.001,
    )


def test_observation_adapter_converts_bgr_and_depth_to_metric() -> None:
    masked = _datapoint()
    raw = PosedImageAndDepth(
        X_WC=masked.X_WC,
        K=masked.K,
        image=masked.image,
        format=masked.format,
        depth=masked.depth,
        depth_scale=masked.depth_scale,
    )
    observations = build_frame_seg_observations([raw], cache_key="scene")

    frame = observations.frames[0]
    assert observations.cache_key == "scene"
    assert np.allclose(frame.color[0, 0], [30 / 255, 20 / 255, 10 / 255])
    assert np.array_equal(frame.depth, np.array([[1.0, 2.0], [0.0, 0.5]], dtype=np.float32))
    assert np.array_equal(frame.X_WV, np.eye(4, dtype=np.float32))


def test_datapoints_for_instance_builds_boolean_masks_without_mutation() -> None:
    datapoint = _datapoint()
    labels = np.array([[0, 2], [1, 2]], dtype=np.int32)
    segmentation = InstanceSegmentation(frame_ids=[0], pixel_object_ids=[labels])

    selected = datapoints_for_instance([datapoint], segmentation, object_id=2)

    assert np.array_equal(selected[0].mask, np.array([[False, True], [False, True]]))
    assert np.array_equal(datapoint.mask, np.ones((2, 2), dtype=bool))
    assert segmentation.object_ids == [1, 2]


def test_datapoints_for_instance_rejects_background() -> None:
    with pytest.raises(ValueError, match="positive"):
        datapoints_for_instance([_datapoint()], InstanceSegmentation([0], [np.zeros((2, 2))]), 0)


def test_instance_segmentation_round_trip(tmp_path: Path) -> None:
    path = tmp_path / "instances.npz"
    expected = InstanceSegmentation(
        frame_ids=[2, 7],
        pixel_object_ids=[
            np.array([[0, 1], [2, 2]], dtype=np.int32),
            np.array([[0, 0], [2, 1]], dtype=np.int32),
        ],
    )

    expected.save(path)
    actual = InstanceSegmentation.load(path)

    assert actual.frame_ids == expected.frame_ids
    assert all(np.array_equal(left, right) for left, right in zip(actual.pixel_object_ids, expected.pixel_object_ids, strict=True))


def test_segment_datapoints_forwards_explicit_runtime(monkeypatch: Any, tmp_path: Path) -> None:
    captured: dict[str, Any] = {}

    def fake_segment_scene(observations: Any, workspace: Any, **kwargs: Any) -> Any:
        captured.update(observations=observations, workspace=workspace, **kwargs)
        return SimpleNamespace(frame_ids=[0], pixel_object_ids=[np.ones((2, 2), dtype=np.int32)])

    monkeypatch.setattr(frame_seg, "segment_scene", fake_segment_scene)
    checkpoint = tmp_path / "sam.pth"
    result = segment_datapoints(
        [_datapoint()],
        Ground(plane=(0.0, 0.0, 1.0, -0.1)),
        np.array([[0.0, 0.0, 0.1]], dtype=np.float32),
        FrameSegConfig(checkpoint_path=checkpoint, cache_dir=tmp_path / "cache"),
        cache_key="test-scene",
    )

    assert result.object_ids == [1]
    assert captured["observations"].cache_key == "test-scene"
    assert captured["runtime"].checkpoint_path == checkpoint
    assert captured["runtime"].cache_dir == tmp_path / "cache"
    assert captured["workspace"].ground_plane == (0.0, 0.0, 1.0, -0.1)
