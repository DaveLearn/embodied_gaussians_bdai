from __future__ import annotations
import open3d.cpu.pybind.core
import typing
__all__ = ['NearestNeighborSearch']
class NearestNeighborSearch:
    """
    NearestNeighborSearch class for nearest neighbor search. Construct a NearestNeighborSearch object with input dataset_points of shape {n_dataset, d}.
    """
    def __init__(self, dataset_points: open3d.cpu.pybind.core.Tensor, index_dtype: open3d.cpu.pybind.core.Dtype = ...) -> None:
        ...
    def fixed_radius_index(self, radius: float | None = None) -> bool:
        ...
    def fixed_radius_search(self, query_points, radius, sort = None):
        """
        Args:
            query_points (open3d.core.Tensor): The query tensor of shape {n_query, d}.
            radius (float): Radius value for radius search.
            sort (Optional[bool], optional, default=None)
        
        Returns:
            tuple[open3d.core.Tensor, open3d.core.Tensor, open3d.core.Tensor]
        """
    def hybrid_index(self, radius: float | None = None) -> bool:
        ...
    def hybrid_search(self, query_points, radius, max_knn):
        """
        Perform hybrid search.
        
        Args:
            query_points (open3d.core.Tensor): The query tensor of shape {n_query, d}.
            radius (float): Radius value for radius search.
            max_knn (int): Maximum number of neighbors to search per query point.
        
        Returns:
            tuple[open3d.core.Tensor, open3d.core.Tensor, open3d.core.Tensor]
        """
    def knn_index(self) -> bool:
        """
        Set index for knn search.
        """
    def knn_search(self, query_points, knn):
        """
        Perform knn search.
        
        Args:
            query_points (open3d.core.Tensor): The query tensor of shape {n_query, d}.
            knn (int): Number of neighbors to search per query point.
        
        Returns:
            tuple[open3d.core.Tensor, open3d.core.Tensor]
        """
    def multi_radius_index(self) -> bool:
        """
        Set index for multi-radius search.
        """
    def multi_radius_search(self, query_points, radii):
        """
        Perform multi-radius search. Each query point has an independent radius.
        
        Args:
            query_points (open3d.core.Tensor): The query tensor of shape {n_query, d}.
            radii (open3d.core.Tensor): Tensor of shape {n_query,} containing multiple radii, one for each query point.
        
        Returns:
            tuple[open3d.core.Tensor, open3d.core.Tensor, open3d.core.Tensor]
        """
