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
Visualize gripper URDFs from gripper_descriptions assets using Viser.

Displays all grippers in a grid with a configurable number of columns (default 5).

Usage:
    python -m gripper_descriptions.scripts.vis_all_grippers
    python -m gripper_descriptions.scripts.vis_all_grippers --filter parallel
    python -m gripper_descriptions.scripts.vis_all_grippers --state close
    python -m gripper_descriptions.scripts.vis_all_grippers --show-sweep-volume
    python -m gripper_descriptions.scripts.vis_all_grippers --cols 8
"""

import argparse
import json
import os
import time
from typing import Dict, List, Optional, Tuple

import numpy as np
import yourdfpy

import gripper_descriptions
from gripper_descriptions.viser_utils import (
    create_visualizer,
    make_frame,
    visualize_bbox,
    visualize_mesh,
)


def load_urdf_scene(urdf_path: str) -> yourdfpy.URDF:
    return yourdfpy.URDF.load(
        urdf_path,
        build_scene_graph=True,
        load_meshes=True,
        build_collision_scene_graph=False,
        load_collision_meshes=False,
        force_mesh=False,
        force_collision_mesh=False,
    )


def get_link_colors(gripper_name: str, num_links: int) -> List[List[int]]:
    base_color = [80, 80, 80]
    finger_color = [50, 180, 50]
    default_color = [120, 120, 120]

    colors = []
    for i in range(num_links):
        if i <= 1:
            colors.append(base_color)
        elif gripper_name.startswith("parallel") or gripper_name.startswith("revolute"):
            colors.append(finger_color if i >= 2 else default_color)
        else:
            colors.append(default_color)
    return colors


def discover_grippers(
    name_filter: Optional[str] = None,
) -> List[Tuple[str, str]]:
    """Discover available grippers.

    Returns list of (gripper_name, gripper_dir) tuples.
    """
    grippers = []
    assets_path = gripper_descriptions.get_assets_path()

    for name in gripper_descriptions.list_grippers():
        gdir = os.path.join(assets_path, name)
        if not os.path.exists(os.path.join(gdir, "gripper.urdf")):
            continue
        if not os.path.exists(os.path.join(gdir, "config.json")):
            continue
        if name_filter and name_filter not in name:
            continue
        grippers.append((name, gdir))

    return grippers


def load_gripper(gripper_dir: str) -> Tuple[yourdfpy.URDF, Dict]:
    urdf_path = os.path.join(gripper_dir, "gripper.urdf")
    config_path = os.path.join(gripper_dir, "config.json")
    robot = load_urdf_scene(urdf_path)
    with open(config_path, "r") as f:
        config = json.load(f)
    return robot, config


def get_joint_state(config: Dict, state: str) -> Dict:
    if state == "open":
        return config["open"]
    elif state == "close":
        return config["close"]
    elif state == "half":
        js = {}
        for k in config["open"]:
            o = config["open"][k]
            c = config["close"][k]
            js[k] = o + (c - o) * 0.5
        return js
    raise ValueError(f"Unknown state: {state}")


def visualize_gripper_at_offset(
    vis,
    robot: yourdfpy.URDF,
    js_cfg: Dict,
    gripper_name: str,
    offset: np.ndarray,
    name_prefix: str = "gripper",
):
    """Render a single gripper translated by offset."""
    robot.update_cfg(js_cfg)
    scene = robot.scene
    geometry_names = list(scene.geometry.keys())
    colors = get_link_colors(gripper_name, len(geometry_names))

    for i, geom_name in enumerate(geometry_names):
        mesh = scene.geometry[geom_name]
        local_tf = scene.graph.get(geom_name)[0]

        world_tf = np.eye(4)
        world_tf[:3, 3] = offset
        combined = world_tf @ local_tf

        transformed_mesh = mesh.copy()
        transformed_mesh.apply_transform(combined)

        visualize_mesh(
            vis,
            f"{name_prefix}/link_{i}",
            transformed_mesh,
            color=colors[i] if i < len(colors) else [120, 120, 120],
        )


def visualize_grid(
    vis,
    grippers: List[Tuple[str, str]],
    state: str,
    spacing: float,
    cols: int,
    show_sweep_volume: bool,
    show_sweep_volume_v2: bool,
) -> Tuple[int, int]:
    """Load and display all grippers in a grid layout.

    Returns (loaded, failed) counts.
    """
    vis.scene.reset()
    make_frame(vis, "world", h=0.10, radius=0.002)

    loaded = 0
    failed = 0

    for idx, (name, gdir) in enumerate(grippers):
        col = idx % cols
        row = idx // cols
        offset = np.array([col * spacing, row * spacing, 0.0])
        prefix = f"grippers/{name}"

        try:
            robot, config = load_gripper(gdir)
        except Exception as e:
            print(f"  SKIP {name}: {e}")
            failed += 1
            continue

        js = get_joint_state(config, state)
        visualize_gripper_at_offset(vis, robot, js, name, offset, name_prefix=prefix)

        sv = config.get("sweep_volume")

        if show_sweep_volume and sv is not None:
            sv_offset = np.array(sv["offset"])
            tf = np.eye(4)
            tf[:3, 3] = offset + sv_offset
            visualize_bbox(
                vis,
                f"sweep_volumes/{name}",
                np.array(sv["extents"]),
                T=tf,
                color=[0, 100, 255],
            )

        if show_sweep_volume_v2 and sv is not None:
            if "extents2" in sv and "offset2" in sv:
                sv_offset_v2 = np.array(sv["offset2"])
                tf_v2 = np.eye(4)
                tf_v2[:3, 3] = offset + sv_offset_v2
                visualize_bbox(
                    vis,
                    f"sweep_volumes_v2/{name}",
                    np.array(sv["extents2"]),
                    T=tf_v2,
                    color=[255, 165, 0],
                )

        label_pos = offset + np.array([0.0, 0.0, -0.04])
        vis.scene.add_label(
            f"labels/{name}",
            text=name,
            wxyz=(1.0, 0.0, 0.0, 0.0),
            position=tuple(label_pos),
        )

        loaded += 1

    return loaded, failed


def parse_args():
    parser = argparse.ArgumentParser(
        description="Visualize gripper URDFs in a grid using Viser",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--filter",
        type=str,
        default=None,
        help="Only show grippers whose name contains this substring",
    )
    parser.add_argument(
        "--state",
        type=str,
        default="open",
        choices=["open", "close", "half"],
        help="Joint state to display",
    )
    parser.add_argument(
        "--spacing",
        type=float,
        default=0.35,
        help="Spacing between grippers in a row (meters)",
    )
    parser.add_argument(
        "--cols",
        type=int,
        default=5,
        help="Number of columns in the grid (default: 5)",
    )
    parser.add_argument(
        "--show-sweep-volume",
        action="store_true",
        help="Show open-state sweep volume wireframes (blue)",
    )
    parser.add_argument(
        "--show-sweep-volume-v2",
        action="store_true",
        help="Show mid-state sweep volume v2 wireframes (orange)",
    )
    parser.add_argument(
        "--show-world-frame",
        action="store_true",
        help="Show world coordinate frame at origin",
    )
    parser.add_argument(
        "--port",
        type=int,
        default=8080,
        help="Port for viser server",
    )
    parser.add_argument(
        "--list",
        action="store_true",
        help="List discovered grippers and exit",
    )
    return parser.parse_args()


def main():
    args = parse_args()

    grippers = discover_grippers(name_filter=args.filter)

    if not grippers:
        print("No grippers found. Check filter.")
        return

    if args.list:
        print(f"\n{'='*60}")
        print(f"Discovered {len(grippers)} grippers")
        print(f"{'='*60}")
        for name, gdir in grippers:
            print(f"  {name}")
        print()
        return

    total = len(grippers)
    rows = (total + args.cols - 1) // args.cols

    print(f"Discovered {total} grippers — displaying in {rows}x{args.cols} grid")

    vis = create_visualizer(port=args.port)

    loaded, failed = visualize_grid(
        vis, grippers, args.state, args.spacing, args.cols,
        args.show_sweep_volume, args.show_sweep_volume_v2,
    )
    print(f"  Loaded {loaded}, failed {failed}")
    print(f"  View at: http://localhost:{args.port}")

    print("\nPress Ctrl+C to exit.")
    try:
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        print("\nExiting...")


if __name__ == "__main__":
    main()
