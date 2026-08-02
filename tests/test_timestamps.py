import numpy as np
import pytest

from embodied_gaussians.utils.timestamps import timestamp_to_index


@pytest.mark.parametrize(
    ("timestamp", "expected"),
    [
        (-10.0, 0),
        (1.0, 0),
        (1.999, 0),
        (2.0, 1),
        (3.999, 1),
        (4.0, 2),
        (10.0, 2),
    ],
)
def test_timestamp_to_index_uses_clamped_zero_order_hold(timestamp: float, expected: int) -> None:
    timestamps = np.array([1.0, 2.0, 4.0])

    assert timestamp_to_index(timestamps, timestamp) == expected


def test_timestamp_to_index_rejects_empty_timestamps() -> None:
    with pytest.raises(ValueError, match="must not be empty"):
        timestamp_to_index(np.array([]), 0.0)
