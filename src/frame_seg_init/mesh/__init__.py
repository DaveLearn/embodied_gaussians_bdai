from typing import List
import torch
import open3d as o3d
import numpy as np
from tqdm import tqdm
from frame_seg_init.types import Frame
from scipy.spatial.transform import Rotation as R

x_rot = np.eye(4, dtype=np.float32)
x_rot[:3, :3] = R.from_euler("x", 180, degrees=True).as_matrix()


def to_cam_open3d(frame: Frame) -> o3d.camera.PinholeCameraIntrinsic:
    w = frame.w
    h = frame.h
    fl_x = frame.fl_x
    fl_y = frame.fl_y
    cx = frame.cx
    cy = frame.cy
    camera_intrinsic = o3d.camera.PinholeCameraIntrinsic(w, h, fl_x, fl_y, cx, cy)

    camera_extrinsic = frame.X_VW_opencv.cpu().numpy()  # @ x_rot
    camera = o3d.camera.PinholeCameraParameters()
    camera.extrinsic = camera_extrinsic
    camera.intrinsic = camera_intrinsic
    return camera


def post_process_mesh(mesh: o3d.geometry.TriangleMesh, cluster_to_keep=1000):
    """
    Post-process a mesh to filter out floaters and disconnected parts
    """
    import copy

    print("post processing the mesh to have {} clusterscluster_to_kep".format(cluster_to_keep))
    mesh_0 = copy.deepcopy(mesh)
    with o3d.utility.VerbosityContextManager(o3d.utility.VerbosityLevel.Debug):
        triangle_clusters, cluster_n_triangles, cluster_area = mesh_0.cluster_connected_triangles()

    triangle_clusters = np.asarray(triangle_clusters)
    cluster_n_triangles = np.asarray(cluster_n_triangles)
    cluster_area = np.asarray(cluster_area)
    n_cluster = np.sort(cluster_n_triangles.copy())[-cluster_to_keep]
    n_cluster = max(n_cluster, 50)  # filter meshes smaller than 50
    triangles_to_remove = cluster_n_triangles[triangle_clusters] < n_cluster
    mesh_0.remove_triangles_by_mask(triangles_to_remove)
    mesh_0.remove_unreferenced_vertices()
    mesh_0.remove_degenerate_triangles()
    print("num vertices raw {}".format(len(mesh.vertices)))
    print("num vertices post {}".format(len(mesh_0.vertices)))
    return mesh_0


@torch.no_grad()
def extract_mesh_bounded(frames: List[Frame], voxel_size=0.004, sdf_trunc=0.02, depth_trunc=3) -> o3d.geometry.TriangleMesh:
    """
    Perform TSDF fusion given a fixed depth range, used in the paper.

    voxel_size: the voxel size of the volume
    sdf_trunc: truncation value
    depth_trunc: maximum depth range, should depended on the scene's scales
    mask_backgrond: whether to mask backgroud, only works when the dataset have masks

    return o3d.mesh
    """
    print("Running tsdf volume integration ...")
    print(f"voxel_size: {voxel_size}")
    print(f"sdf_trunc: {sdf_trunc}")
    print(f"depth_truc: {depth_trunc}")

    for frame in frames:
        assert frame.depth is not None
        assert frame.color is not None

    volume = o3d.pipelines.integration.ScalableTSDFVolume(
        voxel_length=voxel_size,
        sdf_trunc=sdf_trunc,
        color_type=o3d.pipelines.integration.TSDFVolumeColorType.RGB8,
    )

    for frame in tqdm(frames, desc="TSDF integration progress"):
        rgb = frame.color.cpu().numpy()
        assert frame.depth is not None
        depth = frame.depth.cpu().numpy()

        ci = o3d.geometry.Image((rgb * 255).astype(np.uint8))
        di = o3d.geometry.Image(depth)

        cam_o3d = to_cam_open3d(frame)

        # make open3d rgbd
        rgbd = o3d.geometry.RGBDImage.create_from_color_and_depth(
            ci,
            di,
            depth_trunc=depth_trunc,
            convert_rgb_to_intensity=False,
            depth_scale=1.0,
        )

        volume.integrate(rgbd, intrinsic=cam_o3d.intrinsic, extrinsic=cam_o3d.extrinsic)

    mesh = volume.extract_triangle_mesh()
    return mesh


def extract_mesh_bounded_with_res(frames: List[Frame], depth_trunc=2, mesh_res=1024) -> o3d.geometry.TriangleMesh:
    voxel_size = depth_trunc / mesh_res
    sdf_trunc = 5.0 * voxel_size
    raw_mesh = extract_mesh_bounded(frames, voxel_size, sdf_trunc, depth_trunc)
    mesh = post_process_mesh(raw_mesh, cluster_to_keep=50)
    return mesh
