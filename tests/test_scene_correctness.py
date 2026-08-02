from types import SimpleNamespace

import numpy as np
import pytest

from embodied_gaussians.embodied_simulator import builder as builder_module
from embodied_gaussians.scene_builders import simple_body_builder as body_builder_module
from embodied_gaussians.scene_builders.domain import Body, Particles
from ground_finder import ground_finder as ground_finder_module


class _RGBDCallObserved(RuntimeError):
    pass


def _open3d_probe(depth_scales: list[float]) -> SimpleNamespace:
    class RGBDImage:
        @staticmethod
        def create_from_color_and_depth(_color, _depth, **kwargs):
            depth_scales.append(kwargs["depth_scale"])
            raise _RGBDCallObserved

    return SimpleNamespace(
        camera=SimpleNamespace(PinholeCameraIntrinsic=lambda *_args: object()),
        geometry=SimpleNamespace(Image=lambda value: value, RGBDImage=RGBDImage),
    )


def _datapoint(depth_scale: float = 0.002) -> SimpleNamespace:
    return SimpleNamespace(
        depth=np.ones((2, 2), dtype=np.float32),
        depth_scale=depth_scale,
        image=np.zeros((2, 2, 3), dtype=np.uint8),
        mask=None,
        K=np.eye(3),
        X_WC=np.eye(4),
    )


def test_simple_body_rgbd_uses_reciprocal_depth_scale(monkeypatch: pytest.MonkeyPatch) -> None:
    depth_scales: list[float] = []
    monkeypatch.setattr(body_builder_module, "o3d", _open3d_probe(depth_scales))

    with pytest.raises(_RGBDCallObserved):
        body_builder_module.SimpleBodyBuilder._merge_into_pointcloud([_datapoint()], max_depth=2.0)

    assert depth_scales == [500.0]


def test_ground_finder_rgbd_uses_reciprocal_depth_scale(monkeypatch: pytest.MonkeyPatch) -> None:
    depth_scales: list[float] = []
    monkeypatch.setattr(ground_finder_module, "o3d", _open3d_probe(depth_scales))

    with pytest.raises(_RGBDCallObserved):
        ground_finder_module.GroundFinder.find_ground(
            ground_finder_module.GroundFinderSettings(),
            [_datapoint()],
        )

    assert depth_scales == [500.0]


def test_bounding_box_rejects_undersized_point_cloud() -> None:
    class PointCloud:
        points = list(range(9))

        def remove_radius_outlier(self, **_kwargs):
            return self, list(range(9))

        def select_by_index(self, _indices):
            return self

        def get_minimal_oriented_bounding_box(self):
            raise AssertionError("bounding-box construction must not be attempted")

    result = body_builder_module.SimpleBodyBuilder._filter_and_get_bounding_box(
        PointCloud(),  # type: ignore[arg-type]
        outlier_radius=0.01,
        outlier_nb_points=20,
    )

    assert result is None


def test_add_rigid_body_preserves_supplied_z_translation(monkeypatch: pytest.MonkeyPatch) -> None:
    captured: dict[str, tuple[float, ...]] = {}
    fake_wp = SimpleNamespace(
        quat_from_matrix=lambda _rotation: (1.0, 0.0, 0.0, 0.0),
        transformf=lambda *values: tuple(float(value) for value in values),
    )
    monkeypatch.setattr(builder_module, "wp", fake_wp)
    monkeypatch.setattr(
        builder_module.EmbodiedGaussiansBuilder,
        "add_body",
        lambda _self, *, origin: captured.setdefault("origin", origin) and 0,
    )
    monkeypatch.setattr(builder_module.EmbodiedGaussiansBuilder, "add_shape_sphere", lambda *_args, **_kwargs: None)

    builder = object.__new__(builder_module.EmbodiedGaussiansBuilder)
    builder.bodies_affected_by_visual_forces = []
    body = Body(
        name="test",
        X_WB=[[1.0, 0.0, 0.0, 0.2], [0.0, 1.0, 0.0, -0.3], [0.0, 0.0, 1.0, 0.47], [0.0, 0.0, 0.0, 1.0]],
        particles=Particles(means=[[0.0, 0.0, 0.0]], quats=[[1.0, 0.0, 0.0, 0.0]], radii=[0.01], colors=[[1.0, 1.0, 1.0]]),
    )

    builder.add_rigid_body(body, add_gaussians=False)

    assert captured["origin"][2] == pytest.approx(0.47)
