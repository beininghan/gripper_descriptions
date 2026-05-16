#!/usr/bin/env python3
# Copyright (c) 2025, NVIDIA CORPORATION. All rights reserved.
#
# NVIDIA CORPORATION and its licensors retain all intellectual property
# and proprietary rights in and to this software, related documentation
# and any modifications thereto. Any use, reproduction, disclosure or
# distribution of this software and related documentation without an express
# license agreement from NVIDIA CORPORATION is strictly prohibited.
"""
Render PNG images of each gripper's visual mesh (open configuration).

Uses pyrender with EGL backend for headless offscreen rendering.
Each gripper's vis_mesh.obj (merged visual mesh at open config) is loaded,
rendered with a single-color metallic material, and saved as a PNG.

Usage:
    # Render all grippers
    python -m gripper_descriptions.scripts.render_gripper_images

    # Render a specific gripper
    python -m gripper_descriptions.scripts.render_gripper_images --gripper robotiq_2f_85

    # Custom output directory and resolution
    python -m gripper_descriptions.scripts.render_gripper_images --output-dir my_figs/ --resolution 600
"""

import argparse
import os
import sys

# Force EGL for headless rendering (must be set before pyrender import)
os.environ["PYOPENGL_PLATFORM"] = "egl"

import numpy as np
import pyrender
import trimesh
from PIL import Image

import gripper_descriptions


def _normalize(v, eps=1e-12):
    n = np.linalg.norm(v)
    if n < eps:
        return v
    return v / n


def camera_pose_from_lookat(eye, target, up=(0, 0, 1)):
    """Compute a 4x4 camera-to-world pose matrix for pyrender (OpenGL convention)."""
    eye = np.asarray(eye, dtype=float)
    target = np.asarray(target, dtype=float)
    up = np.asarray(up, dtype=float)

    # pyrender camera looks down -Z in camera space
    z = _normalize(eye - target)
    x = _normalize(np.cross(up, z))
    if np.linalg.norm(x) < 1e-8:
        alt_up = np.array([0, 1, 0]) if abs(up[2]) > 0.9 else np.array([0, 0, 1])
        x = _normalize(np.cross(alt_up, z))
    y = np.cross(z, x)

    pose = np.eye(4)
    pose[:3, 0] = x
    pose[:3, 1] = y
    pose[:3, 2] = z
    pose[:3, 3] = eye
    return pose


def compute_camera_params(mesh):
    """Compute camera eye position and target to frame the mesh nicely."""
    bounds = mesh.bounds  # (2, 3) — min and max corners
    centroid = mesh.centroid
    extents = bounds[1] - bounds[0]
    radius = np.linalg.norm(extents) / 2.0

    # Position camera at an angle that shows the gripper well
    # Slightly elevated, looking from front-right
    distance = radius * 2.8
    eye = centroid + np.array([distance * 0.6, -distance * 0.8, distance * 0.5])
    target = centroid

    return eye, target


def render_gripper(mesh_path, output_path, resolution=400):
    """Render a single gripper vis_mesh.obj to a PNG file."""
    mesh = trimesh.load(mesh_path, force="mesh")

    # Gripper color — gray (#808080)
    gripper_color = np.array([128, 128, 128, 255], dtype=np.uint8) / 255.0

    scene = pyrender.Scene(
        ambient_light=np.array([0.35, 0.35, 0.35, 1.0]),
        bg_color=np.array([1.0, 1.0, 1.0, 1.0]),
    )

    # Add gripper mesh with metallic material
    pm = pyrender.Mesh.from_trimesh(
        mesh,
        material=pyrender.MetallicRoughnessMaterial(
            baseColorFactor=gripper_color,
            metallicFactor=0.3,
            roughnessFactor=0.5,
        ),
    )
    scene.add(pm)

    # Camera
    eye, target = compute_camera_params(mesh)
    cam = pyrender.PerspectiveCamera(yfov=np.pi / 4.0, aspectRatio=1.0)
    cam_pose = camera_pose_from_lookat(eye, target)
    scene.add(cam, pose=cam_pose)

    # Key light (from camera direction)
    key_light = pyrender.DirectionalLight(color=[1.0, 1.0, 1.0], intensity=4.0)
    scene.add(key_light, pose=cam_pose)

    # Fill light (from opposite side)
    fill_eye = np.array([eye[0] * -1, eye[1] * -1, eye[2]])
    fill_pose = camera_pose_from_lookat(fill_eye, target)
    fill_light = pyrender.DirectionalLight(color=[0.8, 0.8, 0.9], intensity=2.0)
    scene.add(fill_light, pose=fill_pose)

    # Rim light (from above/behind)
    rim_eye = target + np.array([0, 0, np.linalg.norm(eye - target) * 1.2])
    rim_pose = camera_pose_from_lookat(rim_eye, target)
    rim_light = pyrender.DirectionalLight(color=[0.9, 0.9, 1.0], intensity=1.5)
    scene.add(rim_light, pose=rim_pose)

    # Render
    renderer = pyrender.OffscreenRenderer(resolution, resolution)
    color, _ = renderer.render(scene)
    renderer.delete()

    Image.fromarray(color).save(output_path)


def main():
    parser = argparse.ArgumentParser(
        description="Render PNG images of gripper visual meshes",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--gripper",
        type=str,
        default=None,
        help="Render a single gripper by name. If omitted, renders all grippers.",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=None,
        help="Output directory for PNGs. Defaults to <repo_root>/figs/",
    )
    parser.add_argument(
        "--resolution",
        type=int,
        default=400,
        help="Image resolution (square). Default: 400",
    )
    parser.add_argument(
        "--list-grippers",
        action="store_true",
        help="List available grippers and exit.",
    )
    args = parser.parse_args()

    if args.list_grippers:
        grippers = gripper_descriptions.list_grippers()
        print(f"Available grippers ({len(grippers)}):")
        for g in grippers:
            print(f"  {g}")
        return

    # Default output dir: <repo_root>/figs/
    if args.output_dir is None:
        repo_root = os.path.dirname(
            os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        )
        output_dir = os.path.join(repo_root, "figs")
    else:
        output_dir = args.output_dir

    os.makedirs(output_dir, exist_ok=True)

    # Determine which grippers to render
    if args.gripper:
        grippers = [args.gripper]
    else:
        grippers = gripper_descriptions.list_grippers()

    print(f"Rendering {len(grippers)} gripper(s) to {output_dir}/")
    print(f"Resolution: {args.resolution}x{args.resolution}")
    print()

    for gripper_name in grippers:
        try:
            gripper_path = gripper_descriptions.get_gripper_path(gripper_name)
        except FileNotFoundError as e:
            print(f"  SKIP {gripper_name}: {e}")
            continue

        mesh_path = os.path.join(gripper_path, "vis_mesh.obj")
        if not os.path.exists(mesh_path):
            print(f"  SKIP {gripper_name}: vis_mesh.obj not found")
            continue

        output_path = os.path.join(output_dir, f"{gripper_name}.png")
        try:
            render_gripper(mesh_path, output_path, resolution=args.resolution)
            print(f"  OK   {gripper_name} -> {output_path}")
        except Exception as e:
            print(f"  FAIL {gripper_name}: {e}")

    print(f"\nDone. Images saved to {output_dir}/")


if __name__ == "__main__":
    main()
