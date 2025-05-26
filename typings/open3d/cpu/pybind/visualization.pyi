from __future__ import annotations
import numpy
import open3d.cpu.pybind.geometry
import open3d.cpu.pybind.t.geometry
import open3d.cpu.pybind.utility
import typing
__all__ = ['Color', 'Default', 'Material', 'MeshColorOption', 'MeshShadeOption', 'Normal', 'PickedPoint', 'PointColorOption', 'RenderOption', 'ScalarProperties', 'SelectionPolygonVolume', 'TextureMaps', 'VectorProperties', 'ViewControl', 'Visualizer', 'VisualizerWithEditing', 'VisualizerWithKeyCallback', 'VisualizerWithVertexSelection', 'XCoordinate', 'YCoordinate', 'ZCoordinate', 'draw_geometries', 'draw_geometries_with_animation_callback', 'draw_geometries_with_custom_animation', 'draw_geometries_with_editing', 'draw_geometries_with_key_callbacks', 'draw_geometries_with_vertex_selection', 'read_selection_polygon_volume']
class Material:
    """
    Properties (texture maps, scalar and vector) related to visualization. Materials are optionally set for 3D geometries such as TriangleMesh, LineSets, and PointClouds
    """
    material_name: str
    @typing.overload
    def __init__(self) -> None:
        ...
    @typing.overload
    def __init__(self, mat: Material) -> None:
        ...
    @typing.overload
    def __init__(self, material_name: str) -> None:
        ...
    def is_valid(self) -> bool:
        """
        Returns false if material is an empty material
        """
    def set_default_properties(self) -> None:
        """
        Fills material with defaults for common PBR material properties used by Open3D
        """
    @property
    def scalar_properties(self) -> ScalarProperties:
        ...
    @property
    def texture_maps(self) -> TextureMaps:
        ...
    @property
    def vector_properties(self) -> VectorProperties:
        ...
class MeshColorOption:
    """
    Enum class for color for ``TriangleMesh``.
    """
    Color: typing.ClassVar[MeshColorOption]  # value = <MeshColorOption.Color: 1>
    Default: typing.ClassVar[MeshColorOption]  # value = <MeshColorOption.Default: 0>
    Normal: typing.ClassVar[MeshColorOption]  # value = <MeshColorOption.Normal: 9>
    XCoordinate: typing.ClassVar[MeshColorOption]  # value = <MeshColorOption.XCoordinate: 2>
    YCoordinate: typing.ClassVar[MeshColorOption]  # value = <MeshColorOption.YCoordinate: 3>
    ZCoordinate: typing.ClassVar[MeshColorOption]  # value = <MeshColorOption.ZCoordinate: 4>
    __members__: typing.ClassVar[dict[str, MeshColorOption]]  # value = {'Default': <MeshColorOption.Default: 0>, 'Color': <MeshColorOption.Color: 1>, 'XCoordinate': <MeshColorOption.XCoordinate: 2>, 'YCoordinate': <MeshColorOption.YCoordinate: 3>, 'ZCoordinate': <MeshColorOption.ZCoordinate: 4>, 'Normal': <MeshColorOption.Normal: 9>}
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
    def __le__(self, other: typing.Any) -> bool:
        ...
    def __lt__(self, other: typing.Any) -> bool:
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
class MeshShadeOption:
    """
    Enum class for mesh shading for ``TriangleMesh``.
    """
    Color: typing.ClassVar[MeshShadeOption]  # value = <MeshShadeOption.Color: 1>
    Default: typing.ClassVar[MeshShadeOption]  # value = <MeshShadeOption.Default: 0>
    __members__: typing.ClassVar[dict[str, MeshShadeOption]]  # value = {'Default': <MeshShadeOption.Default: 0>, 'Color': <MeshShadeOption.Color: 1>}
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
    def __le__(self, other: typing.Any) -> bool:
        ...
    def __lt__(self, other: typing.Any) -> bool:
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
class PickedPoint:
    coord: numpy.ndarray[numpy.float64[3, 1]]
    index: int
    def __init__(self) -> None:
        ...
class PointColorOption:
    """
    Enum class for point color for ``PointCloud``.
    """
    Color: typing.ClassVar[PointColorOption]  # value = <PointColorOption.Color: 1>
    Default: typing.ClassVar[PointColorOption]  # value = <PointColorOption.Default: 0>
    Normal: typing.ClassVar[PointColorOption]  # value = <PointColorOption.Normal: 9>
    XCoordinate: typing.ClassVar[PointColorOption]  # value = <PointColorOption.XCoordinate: 2>
    YCoordinate: typing.ClassVar[PointColorOption]  # value = <PointColorOption.YCoordinate: 3>
    ZCoordinate: typing.ClassVar[PointColorOption]  # value = <PointColorOption.ZCoordinate: 4>
    __members__: typing.ClassVar[dict[str, PointColorOption]]  # value = {'Default': <PointColorOption.Default: 0>, 'Color': <PointColorOption.Color: 1>, 'XCoordinate': <PointColorOption.XCoordinate: 2>, 'YCoordinate': <PointColorOption.YCoordinate: 3>, 'ZCoordinate': <PointColorOption.ZCoordinate: 4>, 'Normal': <PointColorOption.Normal: 9>}
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
    def __le__(self, other: typing.Any) -> bool:
        ...
    def __lt__(self, other: typing.Any) -> bool:
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
class RenderOption:
    """
    Defines rendering options for visualizer.
    """
    def __init__(self) -> None:
        """
        Default constructor
        """
    def __repr__(self) -> str:
        ...
    def load_from_json(self, filename):
        """
        Function to load RenderOption from a JSON file.
        
        Args:
            filename (str): Path to file.
        
        Returns:
            None
        """
    def save_to_json(self, filename):
        """
        Function to save RenderOption to a JSON file.
        
        Args:
            filename (str): Path to file.
        
        Returns:
            None
        """
    @property
    def background_color(self) -> numpy.ndarray[numpy.float64[3, 1]]:
        """
        float numpy array of size ``(3,)``: Background RGB color.
        """
    @background_color.setter
    def background_color(self, arg0: numpy.ndarray[numpy.float64[3, 1]]) -> None:
        ...
    @property
    def light_on(self) -> bool:
        """
        bool: Whether to turn on Phong lighting.
        """
    @light_on.setter
    def light_on(self, arg0: bool) -> None:
        ...
    @property
    def line_width(self) -> float:
        """
        float: Line width for ``LineSet``.
        """
    @line_width.setter
    def line_width(self, arg0: float) -> None:
        ...
    @property
    def mesh_color_option(self) -> ...:
        """
        ``MeshColorOption``: Color option for ``TriangleMesh``.
        """
    @mesh_color_option.setter
    def mesh_color_option(self, arg0: ...) -> None:
        ...
    @property
    def mesh_shade_option(self) -> ...:
        """
        ``MeshShadeOption``: Mesh shading option for ``TriangleMesh``.
        """
    @mesh_shade_option.setter
    def mesh_shade_option(self, arg0: ...) -> None:
        ...
    @property
    def mesh_show_back_face(self) -> bool:
        """
        bool: Whether to show back faces for ``TriangleMesh``.
        """
    @mesh_show_back_face.setter
    def mesh_show_back_face(self, arg0: bool) -> None:
        ...
    @property
    def mesh_show_wireframe(self) -> bool:
        """
        bool: Whether to show wireframe for ``TriangleMesh``.
        """
    @mesh_show_wireframe.setter
    def mesh_show_wireframe(self, arg0: bool) -> None:
        ...
    @property
    def point_color_option(self) -> ...:
        """
        ``PointColorOption``: Point color option for ``PointCloud``.
        """
    @point_color_option.setter
    def point_color_option(self, arg0: ...) -> None:
        ...
    @property
    def point_show_normal(self) -> bool:
        """
        bool: Whether to show normal for ``PointCloud``.
        """
    @point_show_normal.setter
    def point_show_normal(self, arg0: bool) -> None:
        ...
    @property
    def point_size(self) -> float:
        """
        float: Point size for ``PointCloud``.
        """
    @point_size.setter
    def point_size(self, arg0: float) -> None:
        ...
    @property
    def show_coordinate_frame(self) -> bool:
        """
        bool: Whether to show coordinate frame.
        """
    @show_coordinate_frame.setter
    def show_coordinate_frame(self, arg0: bool) -> None:
        ...
class ScalarProperties:
    def __bool__(self) -> bool:
        """
        Check whether the map is nonempty
        """
    @typing.overload
    def __contains__(self, arg0: str) -> bool:
        ...
    @typing.overload
    def __contains__(self, arg0: typing.Any) -> bool:
        ...
    def __delitem__(self, arg0: str) -> None:
        ...
    def __getitem__(self, arg0: str) -> float:
        ...
    def __init__(self) -> None:
        ...
    def __iter__(self) -> typing.Iterator[str]:
        ...
    def __len__(self) -> int:
        ...
    def __repr__(self) -> str:
        """
        Return the canonical string representation of this map.
        """
    def __setitem__(self, arg0: str, arg1: float) -> None:
        ...
    def items(self) -> typing.ItemsView:
        ...
    def keys(self) -> typing.KeysView:
        ...
    def values(self) -> typing.ValuesView:
        ...
class SelectionPolygonVolume:
    """
    Select a polygon volume for cropping.
    """
    def __copy__(self) -> SelectionPolygonVolume:
        ...
    def __deepcopy__(self, arg0: dict) -> SelectionPolygonVolume:
        ...
    @typing.overload
    def __init__(self) -> None:
        """
        Default constructor
        """
    @typing.overload
    def __init__(self, arg0: SelectionPolygonVolume) -> None:
        """
        Copy constructor
        """
    def __repr__(self) -> str:
        ...
    def crop_in_polygon(self, input):
        """
        Function to crop 3d point clouds.
        
        Args:
            input (open3d.geometry.PointCloud): The input point cloud xyz.
        
        Returns:
            list[int]
        """
    def crop_point_cloud(self, input):
        """
        Function to crop point cloud.
        
        Args:
            input (open3d.geometry.PointCloud): The input point cloud.
        
        Returns:
            open3d.geometry.PointCloud
        """
    def crop_triangle_mesh(self, input):
        """
        Function to crop crop triangle mesh.
        
        Args:
            input (open3d.geometry.TriangleMesh): The input triangle mesh.
        
        Returns:
            open3d.geometry.TriangleMesh
        """
    @property
    def axis_max(self) -> float:
        """
        float: Maximum axis value.
        """
    @axis_max.setter
    def axis_max(self, arg0: float) -> None:
        ...
    @property
    def axis_min(self) -> float:
        """
        float: Minimum axis value.
        """
    @axis_min.setter
    def axis_min(self, arg0: float) -> None:
        ...
    @property
    def bounding_polygon(self) -> open3d.cpu.pybind.utility.Vector3dVector:
        """
        ``(n, 3)`` float64 numpy array: Bounding polygon boundary.
        """
    @bounding_polygon.setter
    def bounding_polygon(self, arg0: open3d.cpu.pybind.utility.Vector3dVector) -> None:
        ...
    @property
    def orthogonal_axis(self) -> str:
        """
        string: one of ``{x, y, z}``.
        """
    @orthogonal_axis.setter
    def orthogonal_axis(self, arg0: str) -> None:
        ...
class TextureMaps:
    def __bool__(self) -> bool:
        """
        Check whether the map is nonempty
        """
    @typing.overload
    def __contains__(self, arg0: str) -> bool:
        ...
    @typing.overload
    def __contains__(self, arg0: typing.Any) -> bool:
        ...
    def __delitem__(self, arg0: str) -> None:
        ...
    def __getitem__(self, arg0: str) -> open3d.cpu.pybind.t.geometry.Image:
        ...
    def __init__(self) -> None:
        ...
    def __iter__(self) -> typing.Iterator[str]:
        ...
    def __len__(self) -> int:
        ...
    def __setitem__(self, arg0: str, arg1: open3d.cpu.pybind.t.geometry.Image) -> None:
        ...
    def items(self) -> typing.ItemsView:
        ...
    def keys(self) -> typing.KeysView:
        ...
    def values(self) -> typing.ValuesView:
        ...
class VectorProperties:
    def __bool__(self) -> bool:
        """
        Check whether the map is nonempty
        """
    @typing.overload
    def __contains__(self, arg0: str) -> bool:
        ...
    @typing.overload
    def __contains__(self, arg0: typing.Any) -> bool:
        ...
    def __delitem__(self, arg0: str) -> None:
        ...
    def __getitem__(self, arg0: str) -> numpy.ndarray[numpy.float32[4, 1]]:
        ...
    def __init__(self) -> None:
        ...
    def __iter__(self) -> typing.Iterator[str]:
        ...
    def __len__(self) -> int:
        ...
    def __repr__(self) -> str:
        """
        Return the canonical string representation of this map.
        """
    def __setitem__(self, arg0: str, arg1: numpy.ndarray[numpy.float32[4, 1]]) -> None:
        ...
    def items(self) -> typing.ItemsView:
        ...
    def keys(self) -> typing.KeysView:
        ...
    def values(self) -> typing.ValuesView:
        ...
class ViewControl:
    """
    View controller for visualizer.
    """
    def __init__(self) -> None:
        """
        Default constructor
        """
    def __repr__(self) -> str:
        ...
    def camera_local_rotate(self, x: float, y: float, xo: float = 0.0, yo: float = 0.0) -> None:
        """
        Function to process rotation of camera in a localcoordinate frame
        """
    def camera_local_translate(self, forward: float, right: float, up: float) -> None:
        """
        Function to process translation of camera
        """
    def change_field_of_view(self, step = 0.45):
        """
        Function to change field of view
        
        Args:
            step (float, optional, default=0.45): The step to change field of view.
        
        Returns:
            None
        """
    def convert_from_pinhole_camera_parameters(self, parameter, allow_arbitrary = False):
        """
        Args:
            parameter (open3d.camera.PinholeCameraParameters): The pinhole camera parameter to convert from.
            allow_arbitrary (bool, optional, default=False)
        
        Returns:
            bool
        """
    def convert_to_pinhole_camera_parameters(self):
        """
        Function to convert ViewControl to camera.PinholeCameraParameters
        
        Returns:
            open3d.camera.PinholeCameraParameters
        """
    def get_field_of_view(self):
        """
        Function to get field of view
        
        Returns:
            float
        """
    def reset_camera_local_rotate(self) -> None:
        """
        Resets the coordinate frame for local camera rotations
        """
    def rotate(self, x, y, xo = 0.0, yo = 0.0):
        """
        Function to process rotation
        
        Args:
            x (float): Distance the mouse cursor has moved in x-axis.
            y (float): Distance the mouse cursor has moved in y-axis.
            xo (float, optional, default=0.0): Original point coordinate of the mouse in x-axis.
            yo (float, optional, default=0.0): Original point coordinate of the mouse in y-axis.
        
        Returns:
            None
        """
    def scale(self, scale):
        """
        Function to process scaling
        
        Args:
            scale (float): Scale ratio.
        
        Returns:
            None
        """
    def set_constant_z_far(self, z_far):
        """
        Function to change the far z-plane of the visualizer to a constant value, i.e., independent of zoom and bounding box size.
        
        Args:
            z_far (float): The depth of the far z-plane of the visualizer.
        
        Returns:
            None
        """
    def set_constant_z_near(self, z_near):
        """
        Function to change the near z-plane of the visualizer to a constant value, i.e., independent of zoom and bounding box size.
        
        Args:
            z_near (float): The depth of the near z-plane of the visualizer.
        
        Returns:
            None
        """
    def set_front(self, front: numpy.ndarray[numpy.float64[3, 1]]) -> None:
        """
        Set the front vector of the visualizer
        """
    def set_lookat(self, lookat: numpy.ndarray[numpy.float64[3, 1]]) -> None:
        """
        Set the lookat vector of the visualizer
        """
    def set_up(self, up: numpy.ndarray[numpy.float64[3, 1]]) -> None:
        """
        Set the up vector of the visualizer
        """
    def set_zoom(self, zoom: float) -> None:
        """
        Set the zoom of the visualizer
        """
    def translate(self, x, y, xo = 0.0, yo = 0.0):
        """
        Function to process translation
        
        Args:
            x (float): Distance the mouse cursor has moved in x-axis.
            y (float): Distance the mouse cursor has moved in y-axis.
            xo (float, optional, default=0.0): Original point coordinate of the mouse in x-axis.
            yo (float, optional, default=0.0): Original point coordinate of the mouse in y-axis.
        
        Returns:
            None
        """
    def unset_constant_z_far(self):
        """
        Function to remove a previously set constant z far value, i.e., far z-plane of the visualizer is dynamically set dependent on zoom and bounding box size.
        
        Returns:
            None
        """
    def unset_constant_z_near(self):
        """
        Function to remove a previously set constant z near value, i.e., near z-plane of the visualizer is dynamically set dependent on zoom and bounding box size.
        
        Returns:
            None
        """
class Visualizer:
    """
    The main Visualizer class.
    """
    def __init__(self) -> None:
        """
        Default constructor
        """
    def __repr__(self) -> str:
        ...
    def add_geometry(self, geometry, reset_bounding_box = True):
        """
        Function to add geometry to the scene and create corresponding shaders
        
        Args:
            geometry (open3d.geometry.Geometry): The ``Geometry`` object.
            reset_bounding_box (bool, optional, default=True): Set to ``False`` to keep current viewpoint
        
        Returns:
            bool
        """
    def capture_depth_float_buffer(self, do_render = False):
        """
        Function to capture depth in a float buffer
        
        Args:
            do_render (bool, optional, default=False): Set to ``True`` to do render.
        
        Returns:
            open3d.geometry.Image
        """
    def capture_depth_image(self, filename, do_render = False, depth_scale = 1000.0):
        """
        Function to capture and save a depth image
        
        Args:
            filename (str): Path to file.
            do_render (bool, optional, default=False): Set to ``True`` to do render.
            depth_scale (float, optional, default=1000.0): Scale depth value when capturing the depth image.
        
        Returns:
            None
        """
    def capture_depth_point_cloud(self, filename, do_render = False, convert_to_world_coordinate = False):
        """
        Function to capture and save local point cloud
        
        Args:
            filename (str): Path to file.
            do_render (bool, optional, default=False): Set to ``True`` to do render.
            convert_to_world_coordinate (bool, optional, default=False): Set to ``True`` to convert to world coordinates
        
        Returns:
            None
        """
    def capture_screen_float_buffer(self, do_render = False):
        """
        Function to capture screen and store RGB in a float buffer
        
        Args:
            do_render (bool, optional, default=False): Set to ``True`` to do render.
        
        Returns:
            open3d.geometry.Image
        """
    def capture_screen_image(self, filename, do_render = False):
        """
        Function to capture and save a screen image
        
        Args:
            filename (str): Path to file.
            do_render (bool, optional, default=False): Set to ``True`` to do render.
        
        Returns:
            None
        """
    def clear_geometries(self) -> bool:
        """
        Function to clear geometries from the visualizer
        """
    def close(self):
        """
        Function to notify the window to be closed
        
        Returns:
            None
        """
    def create_window(self, window_name = 'Open3D', width = 1920, height = 1080, left = 50, top = 50, visible = True):
        """
        Function to create a window and initialize GLFW
        
        Args:
            window_name (str, optional, default='Open3D'): Window title name.
            width (int, optional, default=1920): Width of the window.
            height (int, optional, default=1080): Height of window.
            left (int, optional, default=50): Left margin of the window to the screen.
            top (int, optional, default=50): Top margin of the window to the screen.
            visible (bool, optional, default=True): Whether the window is visible.
        
        Returns:
            bool
        """
    def destroy_window(self):
        """
        Function to destroy a window. This function MUST be called from the main thread.
        
        Returns:
            None
        """
    def get_render_option(self):
        """
        Function to retrieve the associated ``RenderOption``
        
        Returns:
            open3d.visualization.RenderOption
        """
    def get_view_control(self):
        """
        Function to retrieve the associated ``ViewControl``
        
        Returns:
            open3d.visualization.ViewControl
        """
    def get_view_status(self) -> str:
        """
        Get the current view status as a json string of ViewTrajectory.
        """
    def get_window_name(self):
        """
        Returns:
            str
        """
    def is_full_screen(self):
        """
        Function to query whether in fullscreen mode
        
        Returns:
            bool
        """
    def poll_events(self):
        """
        Function to poll events
        
        Returns:
            bool
        """
    def register_animation_callback(self, callback_func):
        """
        Function to register a callback function for animation. The callback function returns if UpdateGeometry() needs to be run.
        
        Args:
            callback_func (Callable[[open3d.visualization.Visualizer], bool]): The call back function.
        
        Returns:
            None
        """
    def remove_geometry(self, geometry, reset_bounding_box = True):
        """
        Function to remove geometry
        
        Args:
            geometry (open3d.geometry.Geometry): The ``Geometry`` object.
            reset_bounding_box (bool, optional, default=True): Set to ``False`` to keep current viewpoint
        
        Returns:
            bool
        """
    def reset_view_point(self, reset_bounding_box = False):
        """
        Function to reset view point
        
        Args:
            reset_bounding_box (bool, optional, default=False): Set to ``False`` to keep current viewpoint
        
        Returns:
            None
        """
    def run(self):
        """
        Function to activate the window. This function will block the current thread until the window is closed.
        
        Returns:
            None
        """
    def set_full_screen(self, fullscreen):
        """
        Function to change between fullscreen and windowed
        
        Args:
            fullscreen (bool)
        
        Returns:
            None
        """
    def set_view_status(self, view_status_str: str) -> None:
        """
        Set the current view status from a json string of ViewTrajectory.
        """
    def toggle_full_screen(self):
        """
        Function to toggle between fullscreen and windowed
        
        Returns:
            None
        """
    def update_geometry(self, geometry):
        """
        Function to update geometry. This function must be called when geometry has been changed. Otherwise the behavior of Visualizer is undefined.
        
        Args:
            geometry (open3d.geometry.Geometry): The ``Geometry`` object.
        
        Returns:
            bool
        """
    def update_renderer(self):
        """
        Function to inform render needed to be updated
        
        Returns:
            None
        """
class VisualizerWithEditing(Visualizer):
    """
    Visualizer with editing capabilities.
    """
    @typing.overload
    def __init__(self) -> None:
        """
        Default constructor
        """
    @typing.overload
    def __init__(self, voxel_size: float, use_dialog: bool, directory: str) -> None:
        ...
    def __repr__(self) -> str:
        ...
    def get_cropped_geometry(self) -> open3d.cpu.pybind.geometry.Geometry:
        """
        Function to get cropped geometry
        """
    def get_picked_points(self) -> list[int]:
        """
        Function to get picked points
        """
class VisualizerWithKeyCallback(Visualizer):
    """
    Visualizer with custom key callback capabilities.
    """
    def __init__(self) -> None:
        """
        Default constructor
        """
    def __repr__(self) -> str:
        ...
    def register_key_action_callback(self, key: int, callback_func: typing.Callable[[Visualizer, int, int], bool]) -> None:
        """
        Function to register a callback function for a key action event. The callback function takes Visualizer, action and mods as input and returns a boolean indicating if UpdateGeometry() needs to be run.
        """
    def register_key_callback(self, key: int, callback_func: typing.Callable[[Visualizer], bool]) -> None:
        """
        Function to register a callback function for a key press event
        """
class VisualizerWithVertexSelection(Visualizer):
    """
    Visualizer with vertex selection capabilities.
    """
    @typing.overload
    def __init__(self) -> None:
        """
        Default constructor
        """
    @typing.overload
    def __init__(self) -> None:
        ...
    def __repr__(self) -> str:
        ...
    def add_picked_points(self, indices: open3d.cpu.pybind.utility.IntVector) -> None:
        """
        Function to add picked points
        """
    def clear_picked_points(self) -> None:
        """
        Function to clear picked points
        """
    def get_picked_points(self) -> list[...]:
        """
        Function to get picked points
        """
    def pick_points(self, x: float, y: float, w: float, h: float) -> open3d.cpu.pybind.utility.IntVector:
        """
        Function to pick points
        """
    def register_selection_changed_callback(self, f: typing.Callable[[], None]) -> None:
        """
        Registers a function to be called when selection changes
        """
    def register_selection_moved_callback(self, f: typing.Callable[[], None]) -> None:
        """
        Registers a function to be called after selection moves
        """
    def register_selection_moving_callback(self, f: typing.Callable[[], None]) -> None:
        """
        Registers a function to be called while selection moves. Geometry's vertex values can be changed, but do not changethe number of vertices.
        """
    def remove_picked_points(self, indices: open3d.cpu.pybind.utility.IntVector) -> None:
        """
        Function to remove picked points
        """
def draw_geometries(*args, **kwargs):
    """
    draw_geometries(*args, **kwargs)
    Overloaded function.
    
    
    1. draw_geometries(geometry_list, window_name='Open3D', width=1920, height=1080, left=50, top=50, point_show_normal=False, mesh_show_wireframe=False, mesh_show_back_face=False)
        Function to draw a list of geometry.Geometry objects
    
    Args:
        geometry_list (list[open3d.geometry.Geometry]): List of geometries to be visualized.
        window_name (str, optional, default='Open3D'): The displayed title of the visualization window.
        width (int, optional, default=1920): The width of the visualization window.
        height (int, optional, default=1080): The height of the visualization window.
        left (int, optional, default=50): The left margin of the visualization window.
        top (int, optional, default=50): The top margin of the visualization window.
        point_show_normal (bool, optional, default=False): Visualize point normals if set to true.
        mesh_show_wireframe (bool, optional, default=False): Visualize mesh wireframe if set to true.
        mesh_show_back_face (bool, optional, default=False): Visualize also the back face of the mesh triangles.
    
    Returns:
        None
    
    2. draw_geometries(geometry_list, window_name='Open3D', width=1920, height=1080, left=50, top=50, point_show_normal=False, mesh_show_wireframe=False, mesh_show_back_face=False, lookat, up, front, zoom)
        Function to draw a list of geometry.Geometry objects
    
    Args:
        geometry_list (list[open3d.geometry.Geometry]): List of geometries to be visualized.
        window_name (str, optional, default='Open3D'): The displayed title of the visualization window.
        width (int, optional, default=1920): The width of the visualization window.
        height (int, optional, default=1080): The height of the visualization window.
        left (int, optional, default=50): The left margin of the visualization window.
        top (int, optional, default=50): The top margin of the visualization window.
        point_show_normal (bool, optional, default=False): Visualize point normals if set to true.
        mesh_show_wireframe (bool, optional, default=False): Visualize mesh wireframe if set to true.
        mesh_show_back_face (bool, optional, default=False): Visualize also the back face of the mesh triangles.
        lookat (numpy.ndarray[numpy.float64[3, 1]]): The lookat vector of the camera.
        up (numpy.ndarray[numpy.float64[3, 1]]): The up vector of the camera.
        front (numpy.ndarray[numpy.float64[3, 1]]): The front vector of the camera.
        zoom (float): The zoom of the camera.
    
    Returns:
        None
    """
def draw_geometries_with_animation_callback(geometry_list, callback_function, window_name = 'Open3D', width = 1920, height = 1080, left = 50, top = 50):
    """
    Function to draw a list of geometry.Geometry objects with a customized animation callback function
    
    Args:
        geometry_list (list[open3d.geometry.Geometry]): List of geometries to be visualized.
        callback_function (Callable[[open3d.visualization.Visualizer], bool]): Call back function to be triggered at a key press event.
        window_name (str, optional, default='Open3D'): The displayed title of the visualization window.
        width (int, optional, default=1920): The width of the visualization window.
        height (int, optional, default=1080): The height of the visualization window.
        left (int, optional, default=50): The left margin of the visualization window.
        top (int, optional, default=50): The top margin of the visualization window.
    
    Returns:
        None
    """
def draw_geometries_with_custom_animation(geometry_list, window_name = 'Open3D', width = 1920, height = 1080, left = 50, top = 50, optional_view_trajectory_json_file = ''):
    """
    Function to draw a list of geometry.Geometry objects with a GUI that supports animation
    
    Args:
        geometry_list (list[open3d.geometry.Geometry]): List of geometries to be visualized.
        window_name (str, optional, default='Open3D'): The displayed title of the visualization window.
        width (int, optional, default=1920): The width of the visualization window.
        height (int, optional, default=1080): The height of the visualization window.
        left (int, optional, default=50): The left margin of the visualization window.
        top (int, optional, default=50): The top margin of the visualization window.
        optional_view_trajectory_json_file (str, optional, default=''): Camera trajectory json file path for custom animation.
    
    Returns:
        None
    """
def draw_geometries_with_editing(geometry_list, window_name = 'Open3D', width = 1920, height = 1080, left = 50, top = 50):
    """
    Function to draw a list of geometry.Geometry providing user interaction
    
    Args:
        geometry_list (list[open3d.geometry.Geometry]): List of geometries to be visualized.
        window_name (str, optional, default='Open3D'): The displayed title of the visualization window.
        width (int, optional, default=1920): The width of the visualization window.
        height (int, optional, default=1080): The height of the visualization window.
        left (int, optional, default=50): The left margin of the visualization window.
        top (int, optional, default=50): The top margin of the visualization window.
    
    Returns:
        None
    """
def draw_geometries_with_key_callbacks(geometry_list, key_to_callback, window_name = 'Open3D', width = 1920, height = 1080, left = 50, top = 50):
    """
    Function to draw a list of geometry.Geometry objects with a customized key-callback mapping
    
    Args:
        geometry_list (list[open3d.geometry.Geometry]): List of geometries to be visualized.
        key_to_callback (dict[int, Callable[[open3d.visualization.Visualizer], bool]]): Map of key to call back functions.
        window_name (str, optional, default='Open3D'): The displayed title of the visualization window.
        width (int, optional, default=1920): The width of the visualization window.
        height (int, optional, default=1080): The height of the visualization window.
        left (int, optional, default=50): The left margin of the visualization window.
        top (int, optional, default=50): The top margin of the visualization window.
    
    Returns:
        None
    """
def draw_geometries_with_vertex_selection(geometry_list, window_name = 'Open3D', width = 1920, height = 1080, left = 50, top = 50):
    """
    Function to draw a list of geometry.Geometry providing ability for user to select points
    
    Args:
        geometry_list (list[open3d.geometry.Geometry]): List of geometries to be visualized.
        window_name (str, optional, default='Open3D'): The displayed title of the visualization window.
        width (int, optional, default=1920): The width of the visualization window.
        height (int, optional, default=1080): The height of the visualization window.
        left (int, optional, default=50): The left margin of the visualization window.
        top (int, optional, default=50): The top margin of the visualization window.
    
    Returns:
        None
    """
def read_selection_polygon_volume(filename):
    """
    Function to read SelectionPolygonVolume from file
    
    Args:
        filename (str): The file path.
    
    Returns:
        open3d.visualization.SelectionPolygonVolume
    """
Color: MeshColorOption  # value = <MeshColorOption.Color: 1>
Default: MeshColorOption  # value = <MeshColorOption.Default: 0>
Normal: MeshColorOption  # value = <MeshColorOption.Normal: 9>
XCoordinate: MeshColorOption  # value = <MeshColorOption.XCoordinate: 2>
YCoordinate: MeshColorOption  # value = <MeshColorOption.YCoordinate: 3>
ZCoordinate: MeshColorOption  # value = <MeshColorOption.ZCoordinate: 4>
