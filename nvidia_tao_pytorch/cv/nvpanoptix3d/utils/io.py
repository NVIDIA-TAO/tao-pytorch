# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

""" IO functions for NVPanoptix3D. """

import os
import torch
import numpy as np
from plyfile import PlyData
from matplotlib import pyplot as plt
from typing import Union, List, Tuple


def read_ply(ply_file):
    """Read a mesh/pointcloud from a PLY file.

    Args:
        ply_file: Path-like to the PLY file to read.

    Returns:
        A tuple ``(points, faces, colors)`` where:
        - ``points``: ``(N, 3)`` float array of XYZ coordinates
        - ``faces``: ``(M, 3)`` int array of triangle vertex indices
        - ``colors``: ``(N, 3)`` uint8/int array of per-vertex RGB values

    Raises:
        KeyError: If the expected PLY elements/fields are missing.
        OSError: If the file cannot be opened/read.
    """
    with open(ply_file, "rb") as file:
        ply_data = PlyData.read(file)

    points = []
    colors = []
    indices = []

    for x, y, z, r, g, b in ply_data["vertex"]:
        points.append([x, y, z])
        colors.append([r, g, b])

    for face in ply_data["face"]:
        indices.append([face[0][0], face[0][1], face[0][2]])

    points = np.array(points)
    colors = np.array(colors)
    indices = np.array(indices)

    return points, indices, colors


def write_ply(
    vertices: Union[np.array, torch.Tensor],
    colors: Union[np.array, torch.Tensor, List, Tuple],
    faces: Union[np.array, torch.Tensor], output_file: os.PathLike
) -> None:
    """Write vertices (and optional colors/faces) to an ASCII PLY file.

    Args:
        vertices: Vertex positions as ``(N, 3)`` array/tensor (XYZ). Tensors are
            detached and moved to CPU before writing.
        colors: Optional vertex colors. Accepts:
            - ``(N, 3)`` array/tensor of RGB values
            - a length-3 list/tuple interpreted as a constant color for all vertices
            - None to omit color fields
        faces: Optional triangle indices as ``(M, 3)`` array/tensor. If None, an
            empty face list is written.
        output_file: Output path for the PLY file.

    Returns:
        None
    """
    if isinstance(vertices, torch.Tensor):
        vertices = vertices.detach().cpu().numpy()

    if isinstance(colors, torch.Tensor):
        colors = colors.detach().cpu().numpy()

    if isinstance(faces, torch.Tensor):
        faces = faces.detach().cpu().numpy()

    if colors is not None:
        if isinstance(colors, (list, tuple)):
            colors = np.ones_like(vertices) * np.array(colors)

    if faces is None:
        faces = []

    with open(output_file, "w") as file:
        file.write("ply \n")
        file.write("format ascii 1.0\n")
        file.write(f"element vertex {len(vertices):d}\n")
        file.write("property float x\n")
        file.write("property float y\n")
        file.write("property float z\n")

        if colors is not None:
            file.write("property uchar red\n")
            file.write("property uchar green\n")
            file.write("property uchar blue\n")

        if faces is not None:
            file.write(f"element face {len(faces):d}\n")
            file.write("property list uchar uint vertex_indices\n")
        file.write("end_header\n")

        if colors is not None:
            for vertex, color in zip(vertices, colors):
                file.write(f"{vertex[0]:f} {vertex[1]:f} {vertex[2]:f} ")
                file.write(f"{int(color[0]):d} {int(color[1]):d} {int(color[2]):d}\n")
        else:
            for vertex in vertices:
                file.write(f"{vertex[0]:f} {vertex[1]:f} {vertex[2]:f}\n")

        for face in faces:
            file.write(f"3 {face[0]:d} {face[1]:d} {face[2]:d}\n")


def write_image(
    image: Union[np.array, torch.Tensor],
    output_file: os.PathLike, kwargs=None
) -> None:
    """Save an image array/tensor to disk using Matplotlib.

    Args:
        image: Image in a format accepted by ``matplotlib.pyplot.imsave`` (typically
            ``(H, W)``, ``(H, W, 3)``, or ``(H, W, 4)``). Torch tensors are detached
            and moved to CPU before saving.
        output_file: Output path for the image file.
        kwargs: Optional dict of keyword arguments forwarded to ``plt.imsave``.
    """
    if isinstance(image, torch.Tensor):
        image = image.detach().cpu().numpy()

    if kwargs is None:
        kwargs = {}
    plt.imsave(output_file, image, **kwargs)


def assemble_frame_name(frame_name, type_name: str, extension: str, drop_yaw: bool = False):
    """Assemble an output filename from a canonical NVPanoptix3D frame name.

    The input ``frame_name`` is expected to be in the form
    ``"<scene>_<angle>_<rot>"``. The returned filename inserts ``type_name`` and
    appends ``extension``.

    Args:
        frame_name: Base frame name with underscore-separated parts.
        type_name: Token to insert into the filename (e.g. ``"rgb"`` or ``"depth"``).
        extension: File extension including leading dot (e.g. ``".png"``).
        drop_yaw: If True, omit the rotation part from the assembled name.

    Returns:
        The assembled filename as a string.
    """
    frame_parts = frame_name.split("_")
    frame_name = frame_parts[0]
    frame_angle = frame_parts[1]
    frame_rot = frame_parts[2]

    if drop_yaw:
        file_name = f"{frame_name}_{type_name}{frame_angle}{extension}"
    else:
        file_name = f"{frame_name}_{type_name}{frame_angle}_{frame_rot}{extension}"

    return file_name


def write_pointcloud(
    points: Union[np.array, torch.Tensor],
    colors: Union[np.array, torch.Tensor, List, Tuple],
    output_file: os.PathLike
) -> None:
    """Write a pointcloud PLY (no faces).

    Args:
        points: ``(N, 3)`` point coordinates.
        colors: Per-point colors or a constant RGB color (see :func:`write_ply`).
        output_file: Output path for the PLY file.
    """
    write_ply(points, colors, None, output_file)
