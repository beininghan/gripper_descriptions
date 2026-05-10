# Copyright (c) 2025, NVIDIA CORPORATION. All rights reserved.
#
# NVIDIA CORPORATION and its licensors retain all intellectual property
# and proprietary rights in and to this software, related documentation
# and any modifications thereto. Any use, reproduction, disclosure or
# distribution of this software and related documentation without an express
# license agreement from NVIDIA CORPORATION is strictly prohibited.
#
"""
Utility functions for visualization using viser.
This module provides visualization helpers for meshes, point clouds, grasps, and coordinate frames.
"""
from typing import Dict, List, Optional, Tuple, Union

import numpy as np
import trimesh
import trimesh.transformations as tra
import viser
import viser.transforms as vtf


def is_rotation_matrix(M, tol=1e-4):
    """Check if matrix M is a valid rotation matrix."""
    tag = False
    I = np.identity(M.shape[0])

    if (np.linalg.norm((np.matmul(M, M.T) - I)) < tol) and (
        np.abs(np.linalg.det(M) - 1) < tol
    ):
        tag = True

    if tag is False:
        print("M @ M.T:\n", np.matmul(M, M.T))
        print("det:", np.linalg.det(M))

    return tag


def get_color_from_score(labels, use_255_scale=False):
    """Convert score labels to RGB colors (red=low, green=high)."""
    scale = 255.0 if use_255_scale else 1.0
    if type(labels) in [np.float32, float]:
        return scale * np.array([1 - labels, labels, 0])
    else:
        score = scale * np.stack(
            [np.ones(labels.shape[0]) - labels, labels, np.zeros(labels.shape[0])],
            axis=1,
        )
        return score.astype(np.int32)


def rgb2hex(rgb: Tuple[int, int, int]) -> str:
    """
    Converts rgb color to hex.

    Args:
        rgb: color in rgb, e.g. (255,0,0)
    """
    return "0x%02x%02x%02x" % (rgb)


def matrix_to_wxyz_position(T: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """
    Convert a 4x4 homogeneous transformation matrix to wxyz quaternion and position.

    Args:
        T: 4x4 homogeneous transformation matrix

    Returns:
        Tuple of (wxyz quaternion, position)
    """
    rotation_matrix = T[:3, :3]
    so3 = vtf.SO3.from_matrix(rotation_matrix)
    wxyz = so3.wxyz
    position = T[:3, 3]
    return wxyz, position


def create_visualizer(clear=True, port: int = 8080) -> viser.ViserServer:
    """
    Create a viser server for visualization.

    Args:
        clear: If True, clear existing scene content
        port: Port number for the viser server (default: 8080)

    Returns:
        viser.ViserServer instance
    """
    print(f"Starting viser server on http://localhost:{port}")
    server = viser.ViserServer(port=port)
    if clear:
        server.scene.reset()
    print(f"Viser server running at http://localhost:{port}")
    return server


def make_frame(
    vis: viser.ViserServer,
    name: str,
    h: float = 0.15,
    radius: float = 0.01,
    o: float = 1.0,
    T: Optional[np.ndarray] = None,
):
    """Add a red-green-blue triad to the Viser visualizer.

    Args:
        vis (viser.ViserServer): the visualizer
        name (string): name for this frame (should be unique)
        h (float): height of frame visualization (axes_length)
        radius (float): radius of frame visualization (axes_radius)
        o (float): opacity (not used in viser frames, kept for API compatibility)
        T (4x4 numpy.array): (optional) transform to apply to this geometry
    """
    if vis is None:
        return

    wxyz = (1.0, 0.0, 0.0, 0.0)
    position = (0.0, 0.0, 0.0)

    if T is not None:
        is_valid = is_rotation_matrix(T[:3, :3])
        if not is_valid:
            raise ValueError("viser_utils: attempted to visualize invalid transform T")
        wxyz, position = matrix_to_wxyz_position(T)

    vis.scene.add_frame(
        name,
        show_axes=True,
        axes_length=h,
        axes_radius=radius,
        wxyz=wxyz,
        position=position,
    )


def visualize_mesh(
    vis: viser.ViserServer,
    name: str,
    mesh: trimesh.Trimesh,
    color: Optional[List[int]] = None,
    transform: Optional[np.ndarray] = None,
):
    """Visualize a mesh in viser.

    Args:
        vis: viser server object
        name: unique name for the mesh
        mesh: trimesh.Trimesh object
        color: RGB color list [R, G, B] in range [0, 255]
        transform: 4x4 homogeneous transform to apply
    """
    if vis is None:
        return None

    if color is None:
        color = np.random.randint(low=0, high=256, size=3).tolist()

    if isinstance(color, np.ndarray):
        color = color.tolist()
    color_tuple = tuple(int(c) for c in color[:3])

    wxyz = (1.0, 0.0, 0.0, 0.0)
    position = (0.0, 0.0, 0.0)

    if transform is not None:
        wxyz, position = matrix_to_wxyz_position(transform)

    return vis.scene.add_mesh_simple(
        name,
        vertices=mesh.vertices.astype(np.float32),
        faces=mesh.faces.astype(np.uint32),
        color=color_tuple,
        wxyz=wxyz,
        position=position,
    )


def visualize_bbox(
    vis: viser.ViserServer,
    name: str,
    dims: np.ndarray,
    T: Optional[np.ndarray] = None,
    color: Optional[List[int]] = None,
):
    """Visualize a bounding box using a wireframe.

    Args:
        vis (viser.ViserServer): the visualizer
        name (string): name for this frame (should be unique)
        dims (array-like): shape (3,), dimensions of the bounding box
        T (4x4 numpy.array): (optional) transform to apply to this geometry
        color: RGB color tuple [R, G, B] in range [0, 255]
    """
    if vis is None:
        return

    if color is None:
        color = [255, 0, 0]

    if isinstance(color, np.ndarray):
        color = color.tolist()
    color_tuple = tuple(int(c) for c in color[:3])

    wxyz = (1.0, 0.0, 0.0, 0.0)
    position = (0.0, 0.0, 0.0)

    if T is not None:
        wxyz, position = matrix_to_wxyz_position(T)

    if isinstance(dims, np.ndarray):
        dims = tuple(float(d) for d in dims)

    vis.scene.add_box(
        name,
        color=color_tuple,
        dimensions=dims,
        wireframe=True,
        wxyz=wxyz,
        position=position,
    )


def visualize_pointcloud(
    vis: viser.ViserServer,
    name: str,
    pc: np.ndarray,
    color: Optional[np.ndarray] = None,
    transform: Optional[np.ndarray] = None,
    size: float = 0.01,
    **kwargs,
):
    """Visualize a point cloud in viser.

    Args:
        vis: viser server object
        name: str
        pc: Nx3 or HxWx3
        color: (optional) same shape as pc[0 - 255] scale or just rgb tuple
        transform: (optional) 4x4 homogeneous transform
        size: point size (default 0.01)
    """
    if vis is None:
        return
    if pc.ndim == 3:
        pc = pc.reshape(-1, pc.shape[-1])

    if pc.shape[-1] != 3:
        pc = pc[:, :3]

    num_points = pc.shape[0]

    if color is not None:
        if isinstance(color, list):
            color = np.array(color)
        color = np.array(color)

        if color.ndim == 3:
            color = color.reshape(-1, color.shape[-1])

        if color.ndim == 1:
            single_color = np.array(color).flatten()[:3]
            color = np.tile(single_color, (num_points, 1))
        elif color.ndim == 2:
            if color.shape[-1] > 3:
                color = color[:, :3]
            if color.shape[0] != num_points:
                if color.shape[0] > num_points:
                    color = color[:num_points]
                else:
                    padding = np.tile(color[-1:], (num_points - color.shape[0], 1))
                    color = np.vstack([color, padding])

        color = color.astype(np.float32)
        if color.size > 0 and color.max() <= 1.0:
            color = (color * 255).astype(np.uint8)
        else:
            color = np.clip(color, 0, 255).astype(np.uint8)
    else:
        color = np.full((num_points, 3), 255, dtype=np.uint8)

    assert color.shape == (num_points, 3), f"Color shape {color.shape} doesn't match expected ({num_points}, 3)"

    wxyz = (1.0, 0.0, 0.0, 0.0)
    position = (0.0, 0.0, 0.0)

    if transform is not None:
        wxyz, position = matrix_to_wxyz_position(transform)

    vis.scene.add_point_cloud(
        name,
        points=pc.astype(np.float32),
        colors=color,
        point_size=size,
        wxyz=wxyz,
        position=position,
    )
