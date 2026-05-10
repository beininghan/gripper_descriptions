# Copyright (c) 2025, NVIDIA CORPORATION. All rights reserved.
#
# NVIDIA CORPORATION and its licensors retain all intellectual property
# and proprietary rights in and to this software, related documentation
# and any modifications thereto. Any use, reproduction, disclosure or
# distribution of this software and related documentation without an express
# license agreement from NVIDIA CORPORATION is strictly prohibited.
"""
gripper_descriptions — Real-world gripper URDF descriptions, meshes, and configs.

Provides 29 real-world gripper assets for use with GraspGenX.
"""
import os
from pathlib import Path
from typing import List

_PACKAGE_DIR = Path(__file__).resolve().parent
_ASSETS_DIR = _PACKAGE_DIR / "assets" / "x_grippers"


def get_assets_path() -> str:
    """Return the absolute path to the x_grippers asset directory.

    This is the directory containing one subdirectory per gripper
    (e.g. ``franka_panda/``, ``robotiq_2f_85/``, ...).
    """
    return str(_ASSETS_DIR)


def get_gripper_path(name: str) -> str:
    """Return the absolute path to a specific gripper's asset directory.

    Args:
        name: Gripper folder name (e.g. ``'robotiq_2f_85'``).

    Raises:
        FileNotFoundError: If the gripper directory does not exist.
    """
    p = _ASSETS_DIR / name
    if not p.is_dir():
        raise FileNotFoundError(
            f"Gripper '{name}' not found at {p}. "
            f"Available: {', '.join(list_grippers())}"
        )
    return str(p)


def list_grippers() -> List[str]:
    """Return sorted list of available gripper names."""
    if not _ASSETS_DIR.is_dir():
        return []
    return sorted(
        d.name
        for d in _ASSETS_DIR.iterdir()
        if d.is_dir() and d.name != "utils"
    )


AVAILABLE_GRIPPERS: List[str] = list_grippers()
