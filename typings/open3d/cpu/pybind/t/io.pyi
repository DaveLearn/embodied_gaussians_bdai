"""
Tensor-based input-output handling module.
"""
from __future__ import annotations
import open3d.cpu.pybind.camera
import open3d.cpu.pybind.core
import open3d.cpu.pybind.t.geometry
import typing
__all__ = ['DepthNoiseSimulator', 'RGBDSensor', 'RGBDVideoMetadata', 'RGBDVideoReader', 'SensorType', 'read_image', 'read_point_cloud', 'read_triangle_mesh', 'write_image', 'write_point_cloud', 'write_triangle_mesh']
class DepthNoiseSimulator:
    """
    Simulate depth image noise from a given noise distortion model. The distortion model is based on *Teichman et. al. "Unsupervised intrinsic calibration of depth sensors via SLAM" RSS 2009*. Also see <http://redwood-data.org/indoor/dataset.html>__
    
    Example::
    
        import open3d as o3d
    
        # Redwood Indoor LivingRoom1 (Augmented ICL-NUIM)
        # http://redwood-data.org/indoor/
        data = o3d.data.RedwoodIndoorLivingRoom1()
        noise_model_path = data.noise_model_path
        im_src_path = data.depth_paths[0]
        depth_scale = 1000.0
    
        # Read clean depth image (uint16)
        im_src = o3d.t.io.read_image(im_src_path)
    
        # Run noise model simulation
        simulator = o3d.t.io.DepthNoiseSimulator(noise_model_path)
        im_dst = simulator.simulate(im_src, depth_scale=depth_scale)
    
        # Save noisy depth image (uint16)
        o3d.t.io.write_image("noisy_depth.png", im_dst)
                
    """
    def __init__(self, noise_model_path):
        """
        Args:
            noise_model_path (str): Path to the noise model file. See http://redwood-data.org/indoor/dataset.html for the format. Or, you may use one of our example datasets, e.g., RedwoodIndoorLivingRoom1.
        """
    def enable_deterministic_debug_mode(self):
        """
        Enable deterministic debug mode. All normally distributed noise will be replaced by 0.
        
        Returns:
            None
        """
    def simulate(self, im_src, depth_scale = 1000.0):
        """
        Apply noise model to a depth image.
        
        Args:
            im_src (open3d.t.geometry.Image): Source depth image, must be with dtype UInt16 or Float32, channels==1.
            depth_scale (float, optional, default=1000.0): Scale factor to the depth image. As a sanity check, if the dtype is Float32, the depth_scale must be 1.0. If the dtype is is UInt16, the depth_scale is typically larger than 1.0, e.g. it can be 1000.0.
        
        Returns:
            open3d.t.geometry.Image
        """
    @property
    def noise_model(self) -> open3d.cpu.pybind.core.Tensor:
        """
        The noise model tensor.
        """
class RGBDSensor:
    """
    Interface class for control of RGBD cameras.
    """
    def __repr__(self) -> str:
        ...
class RGBDVideoMetadata:
    """
    RGBD Video metadata.
    """
    def __init__(self) -> None:
        ...
    def __repr__(self) -> str:
        ...
    @property
    def color_channels(self) -> int:
        """
        Number of color channels.
        """
    @color_channels.setter
    def color_channels(self, arg0: int) -> None:
        ...
    @property
    def color_dt(self) -> open3d.cpu.pybind.core.Dtype:
        """
        Pixel Dtype for color data.
        """
    @color_dt.setter
    def color_dt(self, arg0: open3d.cpu.pybind.core.Dtype) -> None:
        ...
    @property
    def color_format(self) -> str:
        """
        Pixel format for color data
        """
    @color_format.setter
    def color_format(self, arg0: str) -> None:
        ...
    @property
    def depth_dt(self) -> open3d.cpu.pybind.core.Dtype:
        """
        Pixel Dtype for depth data.
        """
    @depth_dt.setter
    def depth_dt(self, arg0: open3d.cpu.pybind.core.Dtype) -> None:
        ...
    @property
    def depth_format(self) -> str:
        """
        Pixel format for depth data
        """
    @depth_format.setter
    def depth_format(self, arg0: str) -> None:
        ...
    @property
    def depth_scale(self) -> float:
        """
        Number of depth units per meter (depth in m = depth_pixel_value/depth_scale).
        """
    @depth_scale.setter
    def depth_scale(self, arg0: float) -> None:
        ...
    @property
    def device_name(self) -> str:
        """
        Capture device name
        """
    @device_name.setter
    def device_name(self, arg0: str) -> None:
        ...
    @property
    def fps(self) -> float:
        """
        Video frame rate (common for both color and depth)
        """
    @fps.setter
    def fps(self, arg0: float) -> None:
        ...
    @property
    def height(self) -> int:
        """
        Height of the video
        """
    @height.setter
    def height(self, arg0: int) -> None:
        ...
    @property
    def intrinsics(self) -> open3d.cpu.pybind.camera.PinholeCameraIntrinsic:
        """
        Shared intrinsics between RGB & depth
        """
    @intrinsics.setter
    def intrinsics(self, arg0: open3d.cpu.pybind.camera.PinholeCameraIntrinsic) -> None:
        ...
    @property
    def serial_number(self) -> str:
        """
        Capture device serial number
        """
    @serial_number.setter
    def serial_number(self, arg0: str) -> None:
        ...
    @property
    def stream_length_usec(self) -> int:
        """
        Length of the video (usec). 0 for live capture.
        """
    @stream_length_usec.setter
    def stream_length_usec(self, arg0: int) -> None:
        ...
    @property
    def width(self) -> int:
        """
        Width of the video
        """
    @width.setter
    def width(self, arg0: int) -> None:
        ...
class RGBDVideoReader:
    """
    RGBD Video file reader.
    """
    @staticmethod
    def create(filename):
        """
        Create RGBD video reader based on filename
        
        Args:
            filename (str): Path to the RGBD video file.
        
        Returns:
            open3d.t.io.RGBDVideoReader
        """
    def __init__(self) -> None:
        ...
    def __repr__(self) -> str:
        ...
    def save_frames(self, frame_path, start_time_us = 0, end_time_us = 18446744073709551615):
        """
        Save synchronized and aligned individual frames to subfolders.
        
        Args:
            frame_path (str): Frames will be stored in stream subfolders 'color' and 'depth' here. The intrinsic camera calibration for the color stream will be saved in 'intrinsic.json'
            start_time_us (int, optional, default=0): Start saving frames from this time (us)
            end_time_us (int, optional, default=18446744073709551615): (default video length) Save frames till this time (us)
        
        Returns:
            None
        """
class SensorType:
    """
    Sensor type
    
    Members:
    
      AZURE_KINECT
    
      REAL_SENSE
    """
    AZURE_KINECT: typing.ClassVar[SensorType]  # value = <SensorType.AZURE_KINECT: 0>
    REAL_SENSE: typing.ClassVar[SensorType]  # value = <SensorType.REAL_SENSE: 1>
    __members__: typing.ClassVar[dict[str, SensorType]]  # value = {'AZURE_KINECT': <SensorType.AZURE_KINECT: 0>, 'REAL_SENSE': <SensorType.REAL_SENSE: 1>}
    def __eq__(self, other: typing.Any) -> bool:
        ...
    def __getstate__(self) -> int:
        ...
    def __hash__(self) -> int:
        ...
    def __index__(self) -> int:
        ...
    def __init__(self, value: int) -> None:
        ...
    def __int__(self) -> int:
        ...
    def __ne__(self, other: typing.Any) -> bool:
        ...
    def __repr__(self) -> str:
        ...
    def __setstate__(self, state: int) -> None:
        ...
    def __str__(self) -> str:
        ...
    @property
    def name(self) -> str:
        ...
    @property
    def value(self) -> int:
        ...
def read_image(filename):
    """
    Function to read image from file.
    
    Args:
        filename (str): Path to file.
    
    Returns:
        open3d.t.geometry.Image
    """
def read_point_cloud(filename, format = 'auto', remove_nan_points = False, remove_infinite_points = False, print_progress = False):
    """
    Function to read PointCloud with tensor attributes from file.
    
    Args:
        filename (str): Path to file.
        format (str, optional, default='auto'): The format of the input file. When not specified or set as ``auto``, the format is inferred from file extension name.
        remove_nan_points (bool, optional, default=False): If true, all points that include a NaN are removed from the PointCloud.
        remove_infinite_points (bool, optional, default=False): If true, all points that include an infinite value are removed from the PointCloud.
        print_progress (bool, optional, default=False): If set to true a progress bar is visualized in the console.
    
    Returns:
        open3d.t.geometry.PointCloud
    """
def read_triangle_mesh(filename: str, enable_post_processing: bool = False, print_progress: bool = False) -> open3d.cpu.pybind.t.geometry.TriangleMesh:
    """
    The general entrance for reading a TriangleMesh from a file.
    The function calls read functions based on the extension name of filename.
    Supported formats are `obj, ply, stl, off, gltf, glb, fbx`.
    
    The following example reads a triangle mesh with the .ply extension::
        import open3d as o3d
        mesh = o3d.t.io.read_triangle_mesh('mesh.ply')
    
    Args:
        filename (str): Path to the mesh file.
        enable_post_processing (bool): If True enables post-processing. 
            Post-processing will 
              - triangulate meshes with polygonal faces
              - remove redundant materials
              - pretransform vertices
              - generate face normals if needed
            
            For more information see ASSIMPs documentation on the flags
            `aiProcessPreset_TargetRealtime_Fast, aiProcess_RemoveRedundantMaterials, 
            aiProcess_OptimizeMeshes, aiProcess_PreTransformVertices`.
            
            Note that identical vertices will always be joined regardless of whether
            post-processing is enabled or not, which changes the number of vertices 
            in the mesh.
    
            The `ply`-format is not affected by the post-processing.
    
        print_progress (bool): If True print the reading progress to the terminal.
        
    Returns:
        Returns the mesh object. On failure an empty mesh is returned.
    """
def write_image(filename, image, quality = -1):
    """
    Function to write Image to file.
    
    Args:
        filename (str): Path to file.
        image (open3d.t.geometry.Image): The ``Image`` object for I/O.
        quality (int, optional, default=-1): Quality of the output file.
    
    Returns:
        bool
    """
def write_point_cloud(filename, pointcloud, write_ascii = False, compressed = False, print_progress = False):
    """
    Function to write PointCloud with tensor attributes to file.
    
    Args:
        filename (str): Path to file.
        pointcloud (open3d.t.geometry.PointCloud): The ``PointCloud`` object for I/O.
        write_ascii (bool, optional, default=False): Set to ``True`` to output in ascii format, otherwise binary format will be used.
        compressed (bool, optional, default=False): Set to ``True`` to write in compressed format.
        print_progress (bool, optional, default=False): If set to true a progress bar is visualized in the console.
    
    Returns:
        bool
    """
def write_triangle_mesh(filename, mesh, write_ascii = False, compressed = False, write_vertex_normals = True, write_vertex_colors = True, write_triangle_uvs = True, print_progress = False):
    """
    Function to write TriangleMesh to file
    
    Args:
        filename (str): Path to file.
        mesh (open3d.t.geometry.TriangleMesh): The ``TriangleMesh`` object for I/O.
        write_ascii (bool, optional, default=False): Set to ``True`` to output in ascii format, otherwise binary format will be used.
        compressed (bool, optional, default=False): Set to ``True`` to write in compressed format.
        write_vertex_normals (bool, optional, default=True): Set to ``False`` to not write any vertex normals, even if present on the mesh.
        write_vertex_colors (bool, optional, default=True): Set to ``False`` to not write any vertex colors, even if present on the mesh.
        write_triangle_uvs (bool, optional, default=True): Set to ``False`` to not write any triangle uvs, even if present on the mesh. For ``obj`` format, mtl file is saved only when ``True`` is set.
        print_progress (bool, optional, default=False): If set to true a progress bar is visualized in the console.
    
    Returns:
        bool
    """
