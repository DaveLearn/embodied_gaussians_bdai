# Copyright (c) 2025 Boston Dynamics AI Institute LLC. All rights reserved.

from typing import Any, Literal

import torch
from gsplat.rendering import rasterization as _gsplat_rasterization


RenderMode = Literal["RGB", "D", "ED", "RGB+D", "RGB+ED"]


def rasterization(
    *,
    backgrounds: torch.Tensor | None = None,
    packed: bool = True,
    render_mode: RenderMode = "RGB",
    **kwargs: Any,
) -> tuple[torch.Tensor, torch.Tensor, dict[str, Any]]:
    """Rasterize while working around gsplat's packed-background shape check."""
    composite_background = backgrounds is not None and packed and render_mode in ("RGB", "RGB+D", "RGB+ED")
    render_colors, render_alphas, info = _gsplat_rasterization(
        backgrounds=None if composite_background else backgrounds,
        packed=packed,
        render_mode=render_mode,
        **kwargs,
    )
    if not composite_background:
        return render_colors, render_alphas, info

    assert backgrounds is not None
    image_dims = render_colors.shape[:-3]
    expanded_backgrounds = torch.broadcast_to(backgrounds, image_dims + (3,))
    expanded_backgrounds = expanded_backgrounds.reshape(image_dims + (1, 1, 3))
    rgb = render_colors[..., :3] + expanded_backgrounds * (1.0 - render_alphas)
    if render_colors.shape[-1] == 3:
        render_colors = rgb
    else:
        render_colors = torch.cat((rgb, render_colors[..., 3:]), dim=-1)
    return render_colors, render_alphas, info
