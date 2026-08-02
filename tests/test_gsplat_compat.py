from typing import Any

import torch

from embodied_gaussians.utils import gsplat


def test_packed_background_is_composited_after_rasterization(monkeypatch) -> None:
    backend_arguments: dict[str, Any] = {}

    def fake_rasterization(**kwargs):
        backend_arguments.update(kwargs)
        colors = torch.full((2, 4, 5, 3), 0.1)
        alphas = torch.full((2, 4, 5, 1), 0.25)
        return colors, alphas, {"gaussian_ids": torch.tensor([0])}

    monkeypatch.setattr(gsplat, "_gsplat_rasterization", fake_rasterization)
    backgrounds = torch.tensor([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]])

    colors, alphas, info = gsplat.rasterization(backgrounds=backgrounds, packed=True)

    assert backend_arguments["backgrounds"] is None
    torch.testing.assert_close(colors[0, 0, 0], torch.tensor([0.85, 0.1, 0.1]))
    torch.testing.assert_close(colors[1, 0, 0], torch.tensor([0.1, 0.85, 0.1]))
    assert alphas.shape == (2, 4, 5, 1)
    torch.testing.assert_close(info["gaussian_ids"], torch.tensor([0]))


def test_unpacked_background_is_forwarded(monkeypatch) -> None:
    backend_arguments: dict[str, Any] = {}

    def fake_rasterization(**kwargs):
        backend_arguments.update(kwargs)
        return torch.empty(0), torch.empty(0), {}

    monkeypatch.setattr(gsplat, "_gsplat_rasterization", fake_rasterization)
    backgrounds = torch.ones((1, 3))

    gsplat.rasterization(backgrounds=backgrounds, packed=False)

    assert backend_arguments["backgrounds"] is backgrounds
