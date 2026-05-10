#!/usr/bin/env python3
# Copyright (c) 2025, NVIDIA CORPORATION. All rights reserved.
#
# NVIDIA CORPORATION and its licensors retain all intellectual property
# and proprietary rights in and to this software, related documentation
# and any modifications thereto. Any use, reproduction, disclosure or
# distribution of this software and related documentation without an express
# license agreement from NVIDIA CORPORATION is strictly prohibited.
#
"""
Visualize grippers from gripper_descriptions assets using Viser.

Usage:
    python -m gripper_descriptions.scripts.vis_gripper --gripper robotiq_2f_85
    python -m gripper_descriptions.scripts.vis_gripper --gripper robotiq_2f_85 --state close
    python -m gripper_descriptions.scripts.vis_gripper --gripper robotiq_2f_85 --show-sweep-volume
    python -m gripper_descriptions.scripts.vis_gripper --gripper robotiq_2f_85 --animate
"""

import argparse
import json
import os
import sys
from typing import Dict, List, Optional, Tuple

import numpy as np
import trimesh
import trimesh.transformations as tra
import yourdfpy
import viser

import gripper_descriptions
from gripper_descriptions.viser_utils import (
    create_visualizer,
    make_frame,
    visualize_bbox,
    visualize_mesh,
    visualize_pointcloud,
)


def get_traj_js(gripper_config: Dict, num_steps: int = 10) -> List[Dict]:
    """Generate trajectory of joint states from open to close."""
    gripper_open = gripper_config['open']
    gripper_close = gripper_config['close']

    trajs = []
    for s in range(num_steps + 1):
        js = dict()
        for k in gripper_open.keys():
            open_js = gripper_open[k]
            close_js = gripper_close[k]
            js[k] = open_js + (close_js - open_js) * s / num_steps
        trajs.append(js)

    return trajs


def get_link_colors(gripper_name: str, num_links: int) -> List[List[int]]:
    """Get colors for gripper links based on gripper type."""
    base_color = [80, 80, 80]
    finger_color = [50, 180, 50]
    default_color = [120, 120, 120]

    colors = []
    for i in range(num_links):
        if i <= 1:
            colors.append(base_color)
        elif gripper_name.startswith('parallel') or gripper_name.startswith('revolute'):
            colors.append(finger_color if i >= 2 else default_color)
        else:
            colors.append(default_color)

    return colors


def load_urdf_scene(urdf_path: str) -> yourdfpy.URDF:
    """Load URDF using yourdfpy."""
    scene = yourdfpy.URDF.load(
        urdf_path,
        build_scene_graph=True,
        load_meshes=True,
        build_collision_scene_graph=False,
        load_collision_meshes=False,
        force_mesh=False,
        force_collision_mesh=False,
    )
    return scene


def visualize_gripper(
    vis: viser.ViserServer,
    robot: yourdfpy.URDF,
    js_cfg: Dict,
    gripper_name: str,
    name_prefix: str = "gripper",
    show_frames: bool = False,
):
    """Visualize the gripper URDF with given joint configuration."""
    robot.update_cfg(js_cfg)
    scene = robot.scene
    geometry_names = list(scene.geometry.keys())
    colors = get_link_colors(gripper_name, len(geometry_names))

    for i, geom_name in enumerate(geometry_names):
        mesh = scene.geometry[geom_name]
        transform = scene.graph.get(geom_name)[0]
        transformed_mesh = mesh.copy()
        transformed_mesh.apply_transform(transform)

        visualize_mesh(
            vis,
            f"{name_prefix}/link_{i}",
            transformed_mesh,
            color=colors[i] if i < len(colors) else [120, 120, 120],
        )

        if show_frames:
            make_frame(vis, f"{name_prefix}/frame_{i}", T=transform, h=0.02, radius=0.001)


def visualize_sweep_volume(
    vis: viser.ViserServer,
    config: Dict,
    name: str = "sweep_volume",
    use_half: bool = False,
):
    """Visualize the gripper sweep volume as a bounding box."""
    sv = config.get('sweep_volume', None)
    if sv is None:
        print("No sweep volume in config")
        return

    if use_half:
        extents = sv.get('extents2', sv['extents'])
        offset = sv.get('offset2', sv['offset'])
    else:
        extents = sv['extents']
        offset = sv['offset']

    T = tra.translation_matrix(offset)
    visualize_bbox(vis, name, np.array(extents), T=T, color=[0, 100, 255])


def visualize_fingertip(
    vis: viser.ViserServer,
    config: Dict,
    name: str = "fingertip",
):
    """Visualize the fingertip position as a small sphere/frame."""
    fingertip = config.get('fingertip', None)
    if fingertip is None:
        print("No fingertip in config")
        return

    T = tra.translation_matrix(fingertip)
    make_frame(vis, name, T=T, h=0.02, radius=0.002)


def load_gripper(gripper_name: str) -> Tuple[yourdfpy.URDF, Dict, str]:
    """Load gripper URDF and config from gripper_descriptions assets."""
    gripper_path = gripper_descriptions.get_gripper_path(gripper_name)

    urdf_path = os.path.join(gripper_path, "gripper.urdf")
    config_path = os.path.join(gripper_path, "config.json")

    if not os.path.exists(urdf_path):
        raise ValueError(f"URDF not found: {urdf_path}")
    if not os.path.exists(config_path):
        raise ValueError(f"Config not found: {config_path}")

    robot = load_urdf_scene(urdf_path)
    with open(config_path, 'r') as f:
        config = json.load(f)

    return robot, config, gripper_path


def list_available_grippers() -> List[str]:
    """List all available grippers."""
    return gripper_descriptions.list_grippers()


def parse_args():
    parser = argparse.ArgumentParser(
        description="Visualize gripper using Viser",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    # Basic visualization (open state)
    python -m gripper_descriptions.scripts.vis_gripper --gripper robotiq_2f_85

    # Visualize closed state with sweep volume
    python -m gripper_descriptions.scripts.vis_gripper --gripper robotiq_2f_85 --state close --show-sweep-volume

    # Animate the gripper opening/closing
    python -m gripper_descriptions.scripts.vis_gripper --gripper robotiq_2f_85 --animate

    # List all available grippers
    python -m gripper_descriptions.scripts.vis_gripper --list-grippers
        """
    )
    parser.add_argument(
        "--gripper",
        type=str,
        default="robotiq_2f_85",
        help="Gripper name (folder name in assets/x_grippers)",
    )
    parser.add_argument(
        "--state",
        type=str,
        default="open",
        choices=["open", "close", "half"],
        help="Gripper state to visualize",
    )
    parser.add_argument(
        "--show-sweep-volume",
        action="store_true",
        help="Show the sweep volume box",
    )
    parser.add_argument(
        "--show-fingertip",
        action="store_true",
        help="Show the fingertip position",
    )
    parser.add_argument(
        "--show-frames",
        action="store_true",
        help="Show coordinate frames for each link",
    )
    parser.add_argument(
        "--show-world-frame",
        action="store_true",
        help="Show world coordinate frame",
    )
    parser.add_argument(
        "--animate",
        action="store_true",
        help="Animate gripper opening and closing",
    )
    parser.add_argument(
        "--animation-speed",
        type=float,
        default=0.05,
        help="Animation speed (seconds per frame)",
    )
    parser.add_argument(
        "--port",
        type=int,
        default=8080,
        help="Port for viser server",
    )
    parser.add_argument(
        "--list-grippers",
        action="store_true",
        help="List all available grippers and exit",
    )
    return parser.parse_args()


def main():
    args = parse_args()

    # List grippers if requested
    if args.list_grippers:
        grippers = list_available_grippers()
        print(f"\n=== Available Grippers ({len(grippers)}) ===")
        for g in grippers:
            print(f"  {g}")
        return

    # Load gripper
    print(f"Loading gripper: {args.gripper}")
    try:
        robot, config, gripper_path = load_gripper(args.gripper)
    except (ValueError, FileNotFoundError) as e:
        print(f"Error: {e}")
        print("\nUse --list-grippers to see available grippers")
        return
    print(f"Loaded from: {gripper_path}")

    # Create visualizer
    vis = create_visualizer(port=args.port)

    # Get joint state
    if args.state == "open":
        js = config['open']
    elif args.state == "close":
        js = config['close']
    elif args.state == "half":
        traj = get_traj_js(config, num_steps=2)
        js = traj[1]  # Middle state

    # Visualize world frame
    if args.show_world_frame:
        make_frame(vis, "world_frame", h=0.1, radius=0.003)

    # Visualize gripper
    visualize_gripper(
        vis, robot, js, args.gripper,
        name_prefix="gripper",
        show_frames=args.show_frames,
    )

    # Visualize sweep volume
    if args.show_sweep_volume:
        use_half = (args.state == "half")
        visualize_sweep_volume(vis, config, use_half=use_half)

    # Visualize fingertip
    if args.show_fingertip:
        visualize_fingertip(vis, config)

    # Print info
    print(f"\nGripper type: {config.get('type', 'unknown')}")
    print(f"State: {args.state}")
    print(f"Joint config: {js}")
    if 'bbox' in config:
        print(f"Bounding box: {config['bbox']}")

    if args.animate:
        import time
        print("\nAnimating gripper (Ctrl+C to stop)...")
        traj = get_traj_js(config, num_steps=20)
        try:
            while True:
                # Open to close
                for js in traj:
                    visualize_gripper(vis, robot, js, args.gripper, show_frames=args.show_frames)
                    time.sleep(args.animation_speed)
                # Close to open
                for js in reversed(traj):
                    visualize_gripper(vis, robot, js, args.gripper, show_frames=args.show_frames)
                    time.sleep(args.animation_speed)
        except KeyboardInterrupt:
            print("\nAnimation stopped")
    else:
        print("\nVisualization server running. Press Ctrl+C to exit...")
        try:
            while True:
                pass
        except KeyboardInterrupt:
            print("\nExiting...")


if __name__ == "__main__":
    main()
