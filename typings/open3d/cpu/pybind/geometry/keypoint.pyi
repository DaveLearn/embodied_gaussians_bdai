"""
Keypoint Detectors.
"""
from __future__ import annotations
__all__ = ['compute_iss_keypoints']
def compute_iss_keypoints(input, salient_radius = 0.0, non_max_radius = 0.0, gamma_21 = 0.975, gamma_32 = 0.975, min_neighbors = 5):
    """
    Function that computes the ISS keypoints from an input point cloud. This implements the keypoint detection modules proposed in Yu Zhong, 'Intrinsic Shape Signatures: A Shape Descriptor for 3D Object Recognition', 2009.
    
    Args:
        input (open3d.geometry.PointCloud): The Input point cloud.
        salient_radius (float, optional, default=0.0): The radius of the spherical neighborhood used to detect keypoints.
        non_max_radius (float, optional, default=0.0): The non maxima suppression radius
        gamma_21 (float, optional, default=0.975): The upper bound on the ratio between the second and the first eigenvalue returned by the EVD
        gamma_32 (float, optional, default=0.975): The upper bound on the ratio between the third and the second eigenvalue returned by the EVD
        min_neighbors (int, optional, default=5): Minimum number of neighbors that has to be found to consider a keypoint
    
    Returns:
        open3d.geometry.PointCloud
    """
