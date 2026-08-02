from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import cast

from PIL import Image
import tyro

from embodied_gaussians.scene_builders.domain import PosedImageAndDepth, load_posed_images
from embodied_gaussians.segmentation import InstanceSegmentation, overlay_instances


@dataclass
class Params:
    observations_path: tyro.conf.Positional[Path]
    segmentation_path: tyro.conf.Positional[Path]
    output_dir: tyro.conf.Positional[Path]
    overlay_alpha: float = 0.55


def main(params: Params) -> None:
    observations = cast(list[PosedImageAndDepth], list(load_posed_images(params.observations_path)))
    segmentation = InstanceSegmentation.load(params.segmentation_path)
    if len(observations) != len(segmentation.pixel_object_ids):
        raise ValueError(f"Observation/mask count mismatch: {len(observations)} != {len(segmentation.pixel_object_ids)}")

    params.output_dir.mkdir(parents=True, exist_ok=True)
    for frame_id, observation, labels in zip(
        segmentation.frame_ids,
        observations,
        segmentation.pixel_object_ids,
        strict=True,
    ):
        image = overlay_instances(observation.image, observation.format, labels, params.overlay_alpha)
        Image.fromarray(image).save(params.output_dir / f"frame_{frame_id:06d}.png")
    print(f"Exported {len(observations)} instance overlays to {params.output_dir}")


if __name__ == "__main__":
    main(tyro.cli(Params))
