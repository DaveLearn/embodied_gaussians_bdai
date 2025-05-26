from __future__ import annotations
import typing
from . import rpc
__all__ = ['CONTAINS_LINES', 'CONTAINS_POINTS', 'CONTAINS_TRIANGLES', 'CONTENTS_UNKNOWN', 'FileGeometry', 'read_feature', 'read_file_geometry_type', 'read_image', 'read_line_set', 'read_octree', 'read_pinhole_camera_intrinsic', 'read_pinhole_camera_parameters', 'read_pinhole_camera_trajectory', 'read_point_cloud', 'read_point_cloud_from_bytes', 'read_pose_graph', 'read_triangle_mesh', 'read_triangle_model', 'read_voxel_grid', 'rpc', 'write_feature', 'write_image', 'write_line_set', 'write_octree', 'write_pinhole_camera_intrinsic', 'write_pinhole_camera_parameters', 'write_pinhole_camera_trajectory', 'write_point_cloud', 'write_point_cloud_to_bytes', 'write_pose_graph', 'write_triangle_mesh', 'write_voxel_grid']
class FileGeometry:
    """
    Geometry types
    """
    CONTAINS_LINES: typing.ClassVar[FileGeometry]  # value = <FileGeometry.CONTAINS_LINES: 2>
    CONTAINS_POINTS: typing.ClassVar[FileGeometry]  # value = <FileGeometry.CONTAINS_POINTS: 1>
    CONTAINS_TRIANGLES: typing.ClassVar[FileGeometry]  # value = <FileGeometry.CONTAINS_TRIANGLES: 4>
    CONTENTS_UNKNOWN: typing.ClassVar[FileGeometry]  # value = <FileGeometry.CONTENTS_UNKNOWN: 0>
    __members__: typing.ClassVar[dict[str, FileGeometry]]  # value = {'CONTENTS_UNKNOWN': <FileGeometry.CONTENTS_UNKNOWN: 0>, 'CONTAINS_POINTS': <FileGeometry.CONTAINS_POINTS: 1>, 'CONTAINS_LINES': <FileGeometry.CONTAINS_LINES: 2>, 'CONTAINS_TRIANGLES': <FileGeometry.CONTAINS_TRIANGLES: 4>}
    def __and__(self, other: typing.Any) -> typing.Any:
        ...
    def __eq__(self, other: typing.Any) -> bool:
        ...
    def __ge__(self, other: typing.Any) -> bool:
        ...
    def __getstate__(self) -> int:
        ...
    def __gt__(self, other: typing.Any) -> bool:
        ...
    def __hash__(self) -> int:
        ...
    def __index__(self) -> int:
        ...
    def __init__(self, value: int) -> None:
        ...
    def __int__(self) -> int:
        ...
    def __invert__(self) -> typing.Any:
        ...
    def __le__(self, other: typing.Any) -> bool:
        ...
    def __lt__(self, other: typing.Any) -> bool:
        ...
    def __ne__(self, other: typing.Any) -> bool:
        ...
    def __or__(self, other: typing.Any) -> typing.Any:
        ...
    def __rand__(self, other: typing.Any) -> typing.Any:
        ...
    def __repr__(self) -> str:
        ...
    def __ror__(self, other: typing.Any) -> typing.Any:
        ...
    def __rxor__(self, other: typing.Any) -> typing.Any:
        ...
    def __setstate__(self, state: int) -> None:
        ...
    def __str__(self) -> str:
        ...
    def __xor__(self, other: typing.Any) -> typing.Any:
        ...
    @property
    def name(self) -> str:
        ...
    @property
    def value(self) -> int:
        ...
def read_feature(filename):
    """
    Function to read registration.Feature from file
    
    Args:
        filename (str): Path to file.
    
    Returns:
        open3d.pipelines.registration.Feature
    """
def read_file_geometry_type(arg0: str) -> FileGeometry:
    """
    Returns the type of geometry of the file. This is a faster way of determining the file type than attempting to read the file as a point cloud, mesh, or line set in turn.
    """
def read_image(filename):
    """
    Function to read Image from file
    
    Args:
        filename (str): Path to file.
    
    Returns:
        open3d.geometry.Image
    """
def read_line_set(filename, format = 'auto', print_progress = False):
    """
    Function to read LineSet from file
    
    Args:
        filename (str): Path to file.
        format (str, optional, default='auto'): The format of the input file. When not specified or set as ``auto``, the format is inferred from file extension name.
        print_progress (bool, optional, default=False): If set to true a progress bar is visualized in the console
    
    Returns:
        open3d.geometry.LineSet
    """
def read_octree(filename, format = 'auto'):
    """
    Function to read Octree from file
    
    Args:
        filename (str): Path to file.
        format (str, optional, default='auto'): The format of the input file. When not specified or set as ``auto``, the format is inferred from file extension name.
    
    Returns:
        open3d.geometry.Octree
    """
def read_pinhole_camera_intrinsic(filename):
    """
    Function to read PinholeCameraIntrinsic from file
    
    Args:
        filename (str): Path to file.
    
    Returns:
        open3d.camera.PinholeCameraIntrinsic
    """
def read_pinhole_camera_parameters(filename):
    """
    Function to read PinholeCameraParameters from file
    
    Args:
        filename (str): Path to file.
    
    Returns:
        open3d.camera.PinholeCameraParameters
    """
def read_pinhole_camera_trajectory(filename):
    """
    Function to read PinholeCameraTrajectory from file
    
    Args:
        filename (str): Path to file.
    
    Returns:
        open3d.camera.PinholeCameraTrajectory
    """
def read_point_cloud(filename, format = 'auto', remove_nan_points = False, remove_infinite_points = False, print_progress = False):
    """
    Function to read PointCloud from file
    
    Args:
        filename (str): Path to file.
        format (str, optional, default='auto'): The format of the input file. When not specified or set as ``auto``, the format is inferred from file extension name.
        remove_nan_points (bool, optional, default=False): If true, all points that include a NaN are removed from the PointCloud.
        remove_infinite_points (bool, optional, default=False): If true, all points that include an infinite value are removed from the PointCloud.
        print_progress (bool, optional, default=False): If set to true a progress bar is visualized in the console
    
    Returns:
        open3d.geometry.PointCloud
    """
def read_point_cloud_from_bytes(bytes, format = 'auto', remove_nan_points = False, remove_infinite_points = False, print_progress = False):
    """
    Function to read PointCloud from memory
    
    Args:
        bytes (bytes)
        format (str, optional, default='auto'): The format of the input file. When not specified or set as ``auto``, the format is inferred from file extension name.
        remove_nan_points (bool, optional, default=False): If true, all points that include a NaN are removed from the PointCloud.
        remove_infinite_points (bool, optional, default=False): If true, all points that include an infinite value are removed from the PointCloud.
        print_progress (bool, optional, default=False): If set to true a progress bar is visualized in the console
    
    Returns:
        open3d.geometry.PointCloud
    """
def read_pose_graph(filename):
    """
    Function to read PoseGraph from file
    
    Args:
        filename (str): Path to file.
    
    Returns:
        open3d.pipelines.registration.PoseGraph
    """
def read_triangle_mesh(filename, enable_post_processing = False, print_progress = False):
    """
    Function to read TriangleMesh from file
    
    Args:
        filename (str): Path to file.
        enable_post_processing (bool, optional, default=False)
        print_progress (bool, optional, default=False): If set to true a progress bar is visualized in the console
    
    Returns:
        open3d.geometry.TriangleMesh
    """
def read_triangle_model(filename, print_progress = False):
    """
    Function to read visualization.rendering.TriangleMeshModel from file
    
    Args:
        filename (str): Path to file.
        print_progress (bool, optional, default=False): If set to true a progress bar is visualized in the console
    
    Returns:
        open3d.visualization.rendering.TriangleMeshModel
    """
def read_voxel_grid(filename, format = 'auto', print_progress = False):
    """
    Function to read VoxelGrid from file
    
    Args:
        filename (str): Path to file.
        format (str, optional, default='auto'): The format of the input file. When not specified or set as ``auto``, the format is inferred from file extension name.
        print_progress (bool, optional, default=False): If set to true a progress bar is visualized in the console
    
    Returns:
        open3d.geometry.VoxelGrid
    """
def write_feature(filename, feature):
    """
    Function to write Feature to file
    
    Args:
        filename (str): Path to file.
        feature (open3d.pipelines.registration.Feature): The ``Feature`` object for I/O
    
    Returns:
        bool
    """
def write_image(filename, image, quality = -1):
    """
    Function to write Image to file
    
    Args:
        filename (str): Path to file.
        image (open3d.geometry.Image): The ``Image`` object for I/O
        quality (int, optional, default=-1): Quality of the output file.
    
    Returns:
        bool
    """
def write_line_set(filename, line_set, write_ascii = False, compressed = False, print_progress = False):
    """
    Function to write LineSet to file
    
    Args:
        filename (str): Path to file.
        line_set (open3d.geometry.LineSet): The ``LineSet`` object for I/O
        write_ascii (bool, optional, default=False): Set to ``True`` to output in ascii format, otherwise binary format will be used.
        compressed (bool, optional, default=False): Set to ``True`` to write in compressed format.
        print_progress (bool, optional, default=False): If set to true a progress bar is visualized in the console
    
    Returns:
        bool
    """
def write_octree(filename, octree):
    """
    Function to write Octree to file
    
    Args:
        filename (str): Path to file.
        octree (open3d.geometry.Octree): The ``Octree`` object for I/O
    
    Returns:
        bool
    """
def write_pinhole_camera_intrinsic(filename, intrinsic):
    """
    Function to write PinholeCameraIntrinsic to file
    
    Args:
        filename (str): Path to file.
        intrinsic (open3d.camera.PinholeCameraIntrinsic): The ``PinholeCameraIntrinsic`` object for I/O
    
    Returns:
        bool
    """
def write_pinhole_camera_parameters(filename, parameters):
    """
    Function to write PinholeCameraParameters to file
    
    Args:
        filename (str): Path to file.
        parameters (open3d.camera.PinholeCameraParameters): The ``PinholeCameraParameters`` object for I/O
    
    Returns:
        bool
    """
def write_pinhole_camera_trajectory(filename, trajectory):
    """
    Function to write PinholeCameraTrajectory to file
    
    Args:
        filename (str): Path to file.
        trajectory (open3d.camera.PinholeCameraTrajectory): The ``PinholeCameraTrajectory`` object for I/O
    
    Returns:
        bool
    """
def write_point_cloud(filename, pointcloud, format = 'auto', write_ascii = False, compressed = False, print_progress = False):
    """
    Function to write PointCloud to file
    
    Args:
        filename (str): Path to file.
        pointcloud (open3d.geometry.PointCloud): The ``PointCloud`` object for I/O
        format (str, optional, default='auto'): The format of the input file. When not specified or set as ``auto``, the format is inferred from file extension name.
        write_ascii (bool, optional, default=False): Set to ``True`` to output in ascii format, otherwise binary format will be used.
        compressed (bool, optional, default=False): Set to ``True`` to write in compressed format.
        print_progress (bool, optional, default=False): If set to true a progress bar is visualized in the console
    
    Returns:
        bool
    """
def write_point_cloud_to_bytes(pointcloud, format = 'auto', write_ascii = False, compressed = False, print_progress = False):
    """
    Function to write PointCloud to memory
    
    Args:
        pointcloud (open3d.geometry.PointCloud): The ``PointCloud`` object for I/O
        format (str, optional, default='auto'): The format of the input file. When not specified or set as ``auto``, the format is inferred from file extension name.
        write_ascii (bool, optional, default=False): Set to ``True`` to output in ascii format, otherwise binary format will be used.
        compressed (bool, optional, default=False): Set to ``True`` to write in compressed format.
        print_progress (bool, optional, default=False): If set to true a progress bar is visualized in the console
    
    Returns:
        bytes
    """
def write_pose_graph(filename, pose_graph):
    """
    Function to write PoseGraph to file
    
    Args:
        filename (str): Path to file.
        pose_graph (open3d.pipelines.registration.PoseGraph): The ``PoseGraph`` object for I/O
    
    Returns:
        None
    """
def write_triangle_mesh(filename, mesh, write_ascii = False, compressed = False, write_vertex_normals = True, write_vertex_colors = True, write_triangle_uvs = True, print_progress = False):
    """
    Function to write TriangleMesh to file
    
    Args:
        filename (str): Path to file.
        mesh (open3d.geometry.TriangleMesh): The ``TriangleMesh`` object for I/O
        write_ascii (bool, optional, default=False): Set to ``True`` to output in ascii format, otherwise binary format will be used.
        compressed (bool, optional, default=False): Set to ``True`` to write in compressed format.
        write_vertex_normals (bool, optional, default=True): Set to ``False`` to not write any vertex normals, even if present on the mesh
        write_vertex_colors (bool, optional, default=True): Set to ``False`` to not write any vertex colors, even if present on the mesh
        write_triangle_uvs (bool, optional, default=True): Set to ``False`` to not write any triangle uvs, even if present on the mesh. For ``obj`` format, mtl file is saved only when ``True`` is set
        print_progress (bool, optional, default=False): If set to true a progress bar is visualized in the console
    
    Returns:
        bool
    """
def write_voxel_grid(filename, voxel_grid, write_ascii = False, compressed = False, print_progress = False):
    """
    Function to write VoxelGrid to file
    
    Args:
        filename (str): Path to file.
        voxel_grid (open3d.geometry.VoxelGrid): The ``VoxelGrid`` object for I/O
        write_ascii (bool, optional, default=False): Set to ``True`` to output in ascii format, otherwise binary format will be used.
        compressed (bool, optional, default=False): Set to ``True`` to write in compressed format.
        print_progress (bool, optional, default=False): If set to true a progress bar is visualized in the console
    
    Returns:
        bool
    """
CONTAINS_LINES: FileGeometry  # value = <FileGeometry.CONTAINS_LINES: 2>
CONTAINS_POINTS: FileGeometry  # value = <FileGeometry.CONTAINS_POINTS: 1>
CONTAINS_TRIANGLES: FileGeometry  # value = <FileGeometry.CONTAINS_TRIANGLES: 4>
CONTENTS_UNKNOWN: FileGeometry  # value = <FileGeometry.CONTENTS_UNKNOWN: 0>
