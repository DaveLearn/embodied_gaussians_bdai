from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import cast

import numpy as np
import tyro

from embodied_gaussians.scene_builders.domain import Ground, PosedImageAndDepth, load_posed_images
from embodied_gaussians.segmentation import FrameSegConfig, segment_datapoints
from frame_seg_init import GraphParams


@dataclass
class Params:
    observations_path: tyro.conf.Positional[Path]
    ground_path: tyro.conf.Positional[Path]
    output_path: tyro.conf.Positional[Path]
    checkpoint_path: Path
    cache_dir: Path
    cache_key: str
    ground_points_path: Path | None = None
    mesh_path: Path | None = None
    intermediate_outputs_path: Path | None = None
    device: str = "cuda"
    graph: GraphParams = field(default_factory=GraphParams)


def main(params: Params) -> None:
    if params.observations_path.suffix != ".npz":
        raise ValueError("observations_path must be a .npz created by save_posed_images")
    if params.ground_path.suffix != ".json":
        raise ValueError("ground_path must be a Ground JSON file")
    if params.output_path.suffix != ".npz":
        raise ValueError("output_path must use the .npz suffix")

    ground = Ground.model_validate_json(params.ground_path.read_text())
    ground_points_path = params.ground_points_path or params.ground_path.with_suffix(".npy")
    ground_points = np.load(ground_points_path, allow_pickle=False)
    stored = load_posed_images(params.observations_path)
    datapoints = cast(list[PosedImageAndDepth], list(stored))

    result = segment_datapoints(
        datapoints,
        ground,
        ground_points,
        FrameSegConfig(
            checkpoint_path=params.checkpoint_path,
            cache_dir=params.cache_dir,
            device=params.device,
            graph=params.graph,
        ),
        cache_key=params.cache_key,
        mesh_path=params.mesh_path,
        intermediate_outputs_path=params.intermediate_outputs_path,
    )
    result.save(params.output_path)
    print(f"Saved {len(result.object_ids)} object instances to {params.output_path}")


if __name__ == "__main__":
    main(tyro.cli(Params))
