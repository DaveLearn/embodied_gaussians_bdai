from types import SimpleNamespace

import numpy as np
import pytest

from embodied_gaussians.embodied_visualizer import embodied_viewer
from embodied_gaussians.embodied_visualizer import visualizer
from embodied_gaussians.physics_visualizer import simulation_viewer as viewer_module


def _viewer() -> viewer_module.SimulationViewer:
    viewer = object.__new__(viewer_module.SimulationViewer)
    viewer.enable_manipulate = False
    viewer.num_bodies = 1
    viewer.body_id = 0
    viewer.manipulate_operation = viewer_module.guizmo.OPERATION.translate
    viewer.manipulate_mode = viewer_module.guizmo.MODE.local
    return viewer


def test_refresh_accepts_simulation_without_body_state(monkeypatch: pytest.MonkeyPatch) -> None:
    viewer = _viewer()
    viewer.simulator = SimpleNamespace(state_0=SimpleNamespace(body_q=None))
    viewer.render_state = SimpleNamespace(body_q=None)
    monkeypatch.setattr(viewer_module.wp, "launch", lambda **_kwargs: pytest.fail("Warp kernel should not launch"))

    viewer._refresh_body_q()


def test_manipulation_is_disabled_when_simulation_has_no_bodies(monkeypatch: pytest.MonkeyPatch) -> None:
    viewer = _viewer()
    viewer.num_bodies = 0
    viewer.enable_manipulate = True
    viewer.simulator = SimpleNamespace(state_0=SimpleNamespace(body_q=None))
    viewer.render_state = SimpleNamespace(body_q=None)
    monkeypatch.setattr(viewer, "keyboard", lambda: None)

    viewer.render_manipulation()

    assert viewer.enable_manipulate is False


def test_keyboard_shortcuts_require_hover(monkeypatch: pytest.MonkeyPatch) -> None:
    viewer = _viewer()
    pressed_keys: list[object] = []
    fake_imgui = SimpleNamespace(
        Key=viewer_module.imgui.Key,
        is_window_hovered=lambda: False,
        is_key_pressed=lambda key: pressed_keys.append(key) or True,
    )
    monkeypatch.setattr(viewer_module, "imgui", fake_imgui)

    viewer.keyboard()

    assert pressed_keys == []
    assert viewer.enable_manipulate is False


def test_hovered_keyboard_shortcut_toggles_manipulation(monkeypatch: pytest.MonkeyPatch) -> None:
    viewer = _viewer()
    fake_imgui = SimpleNamespace(
        Key=viewer_module.imgui.Key,
        is_window_hovered=lambda: True,
        is_key_pressed=lambda key: key == viewer_module.imgui.Key.m,
    )
    monkeypatch.setattr(viewer_module, "imgui", fake_imgui)

    viewer.keyboard()

    assert viewer.enable_manipulate is True


def test_numpy_transform_conversion_has_scalar_equality() -> None:
    transform = np.eye(4, dtype=np.float32)

    first = embodied_viewer._mat4_from_numpy(transform)
    second = embodied_viewer._mat4_from_numpy(transform)

    assert first == second
    assert len(first) == 16
    assert all(isinstance(value, float) for value in first)


def test_ui_scale_uses_pixel_ratio_by_default(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("EMBODIED_GAUSSIANS_UI_SCALE", raising=False)

    assert visualizer._resolve_ui_scale(1.5) == 1.5
    assert visualizer._resolve_ui_scale(0.75) == 1.0


def test_ui_scale_can_be_overridden(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("EMBODIED_GAUSSIANS_UI_SCALE", "2.25")

    assert visualizer._resolve_ui_scale(1.5) == 2.25


def test_imgui_uses_measured_framebuffer_scale() -> None:
    assert visualizer._measured_framebuffer_scale((320, 240), (320, 240)) == (1.0, 1.0)
    assert visualizer._measured_framebuffer_scale((320, 240), (640, 480)) == (2.0, 2.0)
    assert visualizer._measured_framebuffer_scale((0, 0), (0, 0)) == (1.0, 1.0)
