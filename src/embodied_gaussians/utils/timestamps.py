import numpy as np


def timestamp_to_index(timestamps: np.ndarray, timestamp: float) -> int:
    """Return the latest sample at or before a timestamp, clamped to the data."""
    if timestamps.ndim != 1:
        raise ValueError("timestamps must be one-dimensional")
    if len(timestamps) == 0:
        raise ValueError("timestamps must not be empty")

    index = np.searchsorted(timestamps, timestamp, side="right") - 1
    return int(np.clip(index, 0, len(timestamps) - 1))
