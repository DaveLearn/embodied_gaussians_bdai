from dataclasses import dataclass
from pathlib import Path

import tyro

from embodied_gaussians.scene_builders.domain import save_posed_images
from embodied_gaussians.utils.utils import read_extrinsics
from utils import get_rgbd_datapoints_from_live_cameras


@dataclass
class Params:
    output_path: tyro.conf.Positional[Path]
    extrinsics: Path


def main(params: Params) -> None:
    if params.output_path.suffix != ".npz":
        raise ValueError("output_path must use the .npz suffix")
    datapoints = get_rgbd_datapoints_from_live_cameras(read_extrinsics(params.extrinsics))
    save_posed_images(params.output_path, datapoints)
    print(f"Saved {len(datapoints)} raw posed RGB-D observations to {params.output_path}")


if __name__ == "__main__":
    main(tyro.cli(Params))
