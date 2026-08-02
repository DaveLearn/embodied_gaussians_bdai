from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

from frame_seg_init import (
    GraphParams,
    ObservationFrame,
    Observations,
    RuntimeConfig,
    Workspace,
    segment_scene,
)
import numpy as np

from embodied_gaussians.scene_builders.domain import Ground, MaskedPosedImageAndDepth, PosedImageAndDepth


@dataclass(frozen=True)
class InstanceSegmentation:
    frame_ids: list[int]
    pixel_object_ids: list[np.ndarray]

    @property
    def object_ids(self) -> list[int]:
        if not self.pixel_object_ids:
            return []
        values = np.unique(np.concatenate([mask.reshape(-1) for mask in self.pixel_object_ids]))
        return [int(value) for value in values if value > 0]

    def save(self, path: Path) -> None:
        if path.suffix != ".npz":
            raise ValueError("Instance segmentation output must use the .npz suffix")
        path.parent.mkdir(parents=True, exist_ok=True)
        masks = np.stack(self.pixel_object_ids) if self.pixel_object_ids else np.empty((0, 0, 0), dtype=np.int32)
        np.savez_compressed(
            path,
            frame_ids=np.asarray(self.frame_ids, dtype=np.int64),
            pixel_object_ids=masks.astype(np.int32, copy=False),
        )

    @classmethod
    def load(cls, path: Path) -> "InstanceSegmentation":
        with np.load(path, allow_pickle=False) as data:
            frame_ids = np.asarray(data["frame_ids"], dtype=np.int64)
            masks = np.asarray(data["pixel_object_ids"], dtype=np.int32)
        if masks.ndim != 3:
            raise ValueError(f"Expected saved instance masks with shape NxHxW, got {masks.shape}")
        if len(frame_ids) != len(masks):
            raise ValueError(f"Saved frame/mask count mismatch: {len(frame_ids)} != {len(masks)}")
        return cls(
            frame_ids=[int(frame_id) for frame_id in frame_ids],
            pixel_object_ids=[mask for mask in masks],
        )


@dataclass(frozen=True)
class FrameSegConfig:
    checkpoint_path: Path
    cache_dir: Path
    device: str = "cuda"
    graph: GraphParams = field(default_factory=GraphParams)


def _instance_color(instance_id: int) -> np.ndarray:
    return np.array(
        [
            (instance_id * 73) % 256,
            (instance_id * 127) % 256,
            (instance_id * 179) % 256,
        ],
        dtype=np.float32,
    )


def overlay_instances(image: np.ndarray, image_format: str, labels: np.ndarray, alpha: float) -> np.ndarray:
    if not 0.0 <= alpha <= 1.0:
        raise ValueError("alpha must be between zero and one")
    if image.shape[:2] != labels.shape:
        raise ValueError(f"Image/label shape mismatch: {image.shape[:2]} != {labels.shape}")

    rgb = np.asarray(image)[..., ::-1] if image_format == "bgr" else np.asarray(image)
    rgb = rgb.astype(np.float32, copy=False)
    if rgb.size and float(rgb.max()) <= 1.0:
        rgb = rgb * 255.0

    overlay = rgb.copy()
    for instance_id in np.unique(labels):
        if instance_id <= 0:
            continue
        mask = labels == instance_id
        overlay[mask] = (1.0 - alpha) * rgb[mask] + alpha * _instance_color(int(instance_id))
    return np.clip(overlay, 0, 255).astype(np.uint8)


def _rgb_float_image(datapoint: PosedImageAndDepth) -> np.ndarray:
    color = np.asarray(datapoint.image)
    if color.ndim != 3 or color.shape[2] != 3:
        raise ValueError(f"Expected an HxWx3 color image, got {color.shape}")
    if datapoint.format == "bgr":
        color = color[..., ::-1]
    color = color.astype(np.float32, copy=False)
    if color.size and float(color.max()) > 1.0:
        color = color / 255.0
    return np.ascontiguousarray(color)


def build_frame_seg_observations(
    datapoints: list[PosedImageAndDepth],
    *,
    cache_key: str | None,
) -> Observations:
    frames: list[ObservationFrame] = []
    for frame_id, datapoint in enumerate(datapoints):
        depth = np.asarray(datapoint.depth, dtype=np.float32) * float(datapoint.depth_scale)
        if depth.shape != datapoint.image.shape[:2]:
            raise ValueError(f"Depth shape {depth.shape} does not match image shape {datapoint.image.shape[:2]}")
        frames.append(
            ObservationFrame(
                id=frame_id,
                name=f"frame_{frame_id:06d}",
                color=_rgb_float_image(datapoint),
                depth=np.ascontiguousarray(depth),
                X_WV=np.asarray(datapoint.X_WC, dtype=np.float32),
                K=np.asarray(datapoint.K, dtype=np.float32),
            )
        )
    return Observations(frames=frames, cache_key=cache_key)


def segment_datapoints(
    datapoints: list[PosedImageAndDepth],
    ground: Ground,
    ground_points: np.ndarray,
    config: FrameSegConfig,
    *,
    cache_key: str | None,
    mesh_path: Path | None = None,
    intermediate_outputs_path: Path | None = None,
) -> InstanceSegmentation:
    observations = build_frame_seg_observations(datapoints, cache_key=cache_key)
    result = segment_scene(
        observations,
        Workspace(
            ground_points=np.asarray(ground_points, dtype=np.float32),
            ground_plane=ground.plane,
        ),
        params=config.graph,
        runtime=RuntimeConfig(
            device=config.device,
            cache_dir=config.cache_dir,
            checkpoint_path=config.checkpoint_path,
        ),
        mesh_path=mesh_path,
        intermediate_outputs_path=intermediate_outputs_path,
    )
    return InstanceSegmentation(
        frame_ids=result.frame_ids,
        pixel_object_ids=result.pixel_object_ids,
    )


def datapoints_for_instance(
    datapoints: list[PosedImageAndDepth],
    segmentation: InstanceSegmentation,
    object_id: int,
) -> list[MaskedPosedImageAndDepth]:
    if object_id <= 0:
        raise ValueError("object_id must be positive; zero is reserved for background")
    if len(datapoints) != len(segmentation.pixel_object_ids):
        raise ValueError(f"Datapoint/mask count mismatch: {len(datapoints)} != {len(segmentation.pixel_object_ids)}")

    return [
        MaskedPosedImageAndDepth(
            X_WC=datapoint.X_WC,
            K=datapoint.K,
            image=datapoint.image,
            format=datapoint.format,
            depth=datapoint.depth,
            depth_scale=datapoint.depth_scale,
            mask=np.asarray(mask) == object_id,
        )
        for datapoint, mask in zip(datapoints, segmentation.pixel_object_ids, strict=True)
    ]
