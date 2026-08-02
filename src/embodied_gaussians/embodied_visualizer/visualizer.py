# Copyright (c) 2025 Boston Dynamics AI Institute LLC. All rights reserved.

import math
import os

import marsoom.overlay

from trio_util import periodic

from imgui_bundle import imgui

import marsoom

from embodied_gaussians.environments.embodied_environment import EmbodiedGaussiansEnvironment
from embodied_gaussians.embodied_visualizer.embodied_viewer import EmbodiedViewer


def _resolve_ui_scale(pixel_ratio: float) -> float:
    configured_scale = os.environ.get("EMBODIED_GAUSSIANS_UI_SCALE")
    if configured_scale is None:
        return max(1.0, pixel_ratio)
    scale = float(configured_scale)
    if not math.isfinite(scale) or scale <= 0.0:
        raise ValueError("EMBODIED_GAUSSIANS_UI_SCALE must be a positive finite number")
    return scale


def _measured_framebuffer_scale(window_size: tuple[int, int], framebuffer_size: tuple[int, int]) -> tuple[float, float]:
    window_width, window_height = window_size
    framebuffer_width, framebuffer_height = framebuffer_size
    if window_width <= 0 or window_height <= 0:
        return 1.0, 1.0
    return framebuffer_width / window_width, framebuffer_height / window_height


class EmbodiedGUI(marsoom.Window):
    def __init__(
        self,
        width: int = 1280,
        height: int = 720,
        draw_controls: bool = True,
    ):
        super().__init__(width=width, height=height, caption="Visualizer")
        self._sync_imgui_framebuffer_scale()
        self.ui_scale = _resolve_ui_scale(self.get_pixel_ratio())
        if self.ui_scale != 1.0:
            style = imgui.get_style()
            style.scale_all_sizes(self.ui_scale)
            style.font_scale_main = self.ui_scale
        self.viewer_3d = EmbodiedViewer(self)
        self.draw_controls = draw_controls
        self.callbacks_3d = []
        self.callbacks_render = []

    def _sync_imgui_framebuffer_scale(self) -> None:
        scale_x, scale_y = _measured_framebuffer_scale(self.get_size(), self.get_framebuffer_size())
        imgui.get_io().display_framebuffer_scale = imgui.ImVec2(scale_x, scale_y)

    def on_resize(self, width: int, height: int) -> None:
        super().on_resize(width, height)
        if hasattr(self, "imgui_renderer"):
            self._sync_imgui_framebuffer_scale()

    def set_environment(self, environment: EmbodiedGaussiansEnvironment):
        self.viewer_3d.set_environment(environment)

    async def run_async(self, fps: float = 60.0):
        async for _ in periodic(1 / fps):
            if self.should_exit():
                break
            self.step()

    def render(self):
        imgui.begin("3D Viewer")
        with self.viewer_3d.draw(in_imgui_window=True):
            self.viewer_3d.render()

        self.viewer_3d.render_manipulation()
        for callback in self.callbacks_3d:
            callback()

        if self.draw_controls:
            self.viewer_3d.render_controls()
        self.viewer_3d.process_nav()
        imgui.end()

        for callback in self.callbacks_render:
            callback()
