# Copyright (c) 2025 Boston Dynamics AI Institute LLC. All rights reserved.

from dataclasses import dataclass

import numpy as np

import open3d as o3d
from embodied_gaussians.scene_builders.domain import MaskedPosedImageAndDepth


@dataclass
class GroundFinderSettings:
    points_per_cm: float = 0.8
    xmin: float = -0.9
    xmax: float = 0.9
    ymin: float = -1.0
    ymax: float = 1.0

    plane_segment_distance_threshold: float = 0.001
    plane_segment_ransac_n: int = 3
    plane_segment_num_iterations: int = 1000

    max_depth: float = 10.0
    # we calibrated camers with charcuo on the table so top of table should be offset 0
    fix_offset_to_zero: bool = False
    use_instance_masks_for_plane_points: bool = False


@dataclass
class GroundFinderResult:
    plane: np.ndarray  # (4,) ax + by + cz + d = 0
    points: np.ndarray


class GroundFinder:
    @staticmethod
    def find_ground(
        settings: GroundFinderSettings,
        datapoints: list[MaskedPosedImageAndDepth],
        visualize: bool = False,
    ) -> GroundFinderResult:
        """Find the ground plane from a list of data"""

        # ====================
        # GET ALL POINTCLOUDS
        # ====================
        all_pointclouds = []
        for datapoint in datapoints:
            if datapoint.mask is not None:
                datapoint.depth[datapoint.mask == False] = 0.0  # noqa: E712

            w = datapoint.depth.shape[1]
            h = datapoint.depth.shape[0]
            intrinsics = o3d.camera.PinholeCameraIntrinsic(
                w,
                h,
                datapoint.K[0, 0],
                datapoint.K[1, 1],
                datapoint.K[0, 2],
                datapoint.K[1, 2],
            )
            depth_image = o3d.geometry.Image(datapoint.depth)
            if datapoint.image is not None:
                color_image = o3d.geometry.Image(datapoint.image)
                rgbd_image = o3d.geometry.RGBDImage.create_from_color_and_depth(
                    color_image, depth_image, depth_scale=1.0 / datapoint.depth_scale, convert_rgb_to_intensity=False
                )
                pointcloud = o3d.geometry.PointCloud.create_from_rgbd_image(rgbd_image, intrinsics)
            else:
                pointcloud = o3d.geometry.PointCloud.create_from_depth_image(
                    depth_image,
                    intrinsics,
                    depth_scale=1.0 / datapoint.depth_scale,
                    depth_trunc=settings.max_depth,
                )
            pointcloud.transform(datapoint.get_X_WC("opencv"))
            all_pointclouds.append(pointcloud)

        final_pointcloud = o3d.geometry.PointCloud()
        for p in all_pointclouds:
            final_pointcloud += p

        if visualize:
            # origin = o3d.geometry.TriangleMesh.create_coordinate_frame(size=1.0, origin=[0, 0, 0])
            o3d.visualization.draw_geometries([*all_pointclouds])

        # ====================
        # FIT PLANE
        # ====================
        plane_model, inliers = final_pointcloud.segment_plane(
            distance_threshold=settings.plane_segment_distance_threshold,
            ransac_n=settings.plane_segment_ransac_n,
            num_iterations=settings.plane_segment_num_iterations,
        )
        inlier_cloud = final_pointcloud.select_by_index(inliers)

        # ====================
        # GET PLANE POINTCLOUD
        # ====================
        plane_model = np.array(plane_model)

        if settings.fix_offset_to_zero:
            plane_model[3] = 0.0

        points_per_cm = settings.points_per_cm
        xmin, xmax, ymin, ymax = (
            settings.xmin,
            settings.xmax,
            settings.ymin,
            settings.ymax,
        )
        num_x = int(abs(xmax - xmin) * points_per_cm * 100)
        num_y = int(abs(ymax - ymin) * points_per_cm * 100)
        x = np.linspace(xmin, xmax, num_x)
        y = np.linspace(ymin, ymax, num_y)
        x, y = np.meshgrid(x, y)
        z = (-plane_model[0] * x - plane_model[1] * y - plane_model[3]) / plane_model[2]
        plane_points = np.stack([x, y, z], axis=-1)
        plane_points = plane_points.reshape(-1, 3)
        plane_points = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(plane_points))

        # ====================
        # CROP PLANE POINTCLOUD
        # ====================
        if settings.use_instance_masks_for_plane_points:
            mask_points = []
            for datapoint in datapoints:
                if datapoint.mask is None:
                    continue

                mask = datapoint.mask != 0
                ys, xs = np.nonzero(mask)
                if ys.size == 0:
                    continue

                fx, fy = datapoint.K[0, 0], datapoint.K[1, 1]
                cx, cy = datapoint.K[0, 2], datapoint.K[1, 2]
                x = (xs - cx) / fx
                y = (ys - cy) / fy
                dirs_cam = np.stack([x, y, np.ones_like(x)], axis=1)

                X_WC = datapoint.get_X_WC("opencv")
                R = X_WC[:3, :3]
                C = X_WC[:3, 3]
                dirs_world = (R @ dirs_cam.T).T

                denom = plane_model[:3] @ dirs_world.T
                valid = np.abs(denom) > 1e-8
                t = -(plane_model[:3] @ C + plane_model[3]) / denom
                valid = valid & (t > 0.0)
                if settings.max_depth > 0.0:
                    valid = valid & (t <= settings.max_depth)

                if not np.any(valid):
                    continue

                points = C + dirs_world * t[:, None]
                mask_points.append(points[valid])

            if mask_points:
                mask_points = np.concatenate(mask_points, axis=0)
            else:
                mask_points = np.empty((0, 3), dtype=np.float64)

            inlier_cloud = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(mask_points))
            inlier_cloud = inlier_cloud.voxel_down_sample(voxel_size=0.001)

        final_points = []
        if len(inlier_cloud.points) > 0:
            kdtree = o3d.geometry.KDTreeFlann(inlier_cloud)
            for p in plane_points.points:
                [k, _, _] = kdtree.search_radius_vector_3d(p, 0.01)
                if k >= 1:
                    final_points.append(p)
        final_points = np.array(final_points)

        print(f"Found {len(final_points)} points on the ground")
        plane_points = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(final_points))

        res = GroundFinderResult(plane_model, final_points)

        if visualize:
            origin = o3d.geometry.TriangleMesh.create_coordinate_frame(size=1.0, origin=[0, 0, 0])
            inlier_cloud.paint_uniform_color([0.0, 1.0, 0.0])
            o3d.visualization.draw_geometries([plane_points, origin, *all_pointclouds])

        return res
