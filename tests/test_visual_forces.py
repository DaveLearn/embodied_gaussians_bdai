import torch

from embodied_gaussians.embodied_simulator.visual_forces import VisualForces


def _visual_forces(body_ids: torch.Tensor) -> VisualForces:
    visual_forces = object.__new__(VisualForces)
    visual_forces.device = body_ids.device
    visual_forces.means = torch.empty((len(body_ids), 3), device=body_ids.device)
    visual_forces._initialize(body_ids)
    return visual_forces


def test_sum_forces_by_body_uses_torch_segment_indices() -> None:
    visual_forces = _visual_forces(torch.tensor([0, 0, -1, 2, 2]))
    visual_forces.forces = torch.tensor(
        [
            [1.0, 2.0, 3.0],
            [4.0, 5.0, 6.0],
            [100.0, 100.0, 100.0],
            [7.0, 8.0, 9.0],
            [10.0, 11.0, 12.0],
        ]
    )
    visual_forces.moments = visual_forces.forces * 2.0

    visual_forces.sum_forces_by_body()

    torch.testing.assert_close(visual_forces._body_ids, torch.tensor([0, 2]))
    torch.testing.assert_close(
        visual_forces._total_forces[: visual_forces._num_bodies],
        torch.tensor([[5.0, 7.0, 9.0], [17.0, 19.0, 21.0]]),
    )
    torch.testing.assert_close(
        visual_forces._total_moments[: visual_forces._num_bodies],
        torch.tensor([[10.0, 14.0, 18.0], [34.0, 38.0, 42.0]]),
    )


def test_sum_forces_by_body_handles_no_gaussians() -> None:
    visual_forces = _visual_forces(torch.empty(0, dtype=torch.int64))
    visual_forces.forces = torch.empty((0, 3))
    visual_forces.moments = torch.empty((0, 3))

    visual_forces.sum_forces_by_body()

    assert visual_forces._num_bodies == 0
    assert visual_forces._total_forces.shape == (1, 3)
    assert visual_forces._total_moments.shape == (1, 3)
