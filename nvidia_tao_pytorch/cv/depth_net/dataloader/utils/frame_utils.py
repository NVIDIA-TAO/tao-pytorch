# Copyright (c) 2025, NVIDIA CORPORATION.  All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""DepthNet frame related utils module."""

import numpy as np
import os
import cv2
import re
import torch
from PIL import Image
from os.path import splitext
import nvidia_tao_pytorch.core.loggers.api_logging as status_logging


def depth_to_uint8_encoding(depth, scale=1000):
    """
    Encodes a floating-point depth map into a 3-channel 8-bit unsigned integer (uint8) image.
    This method effectively stores a higher precision depth value in 3 bytes per pixel,
    similar to how some datasets (e.g., Middlebury) encode 16-bit or 24-bit depth.

    The encoding works as:
    depth_value * scale = R * 255*255 + G * 255 + B
    where R, G, B are the channel values (0-255).

    Args:
        depth (numpy.ndarray): The input floating-point depth map (H, W).
        scale (int, optional): A scaling factor applied to the depth before encoding.
                               Defaults to 1000.

    Returns:
        numpy.ndarray: The encoded depth map as a 3-channel uint8 image (H, W, 3).
    """
    depth = depth * scale
    H, W = depth.shape
    out = np.zeros((H, W, 3), dtype=float)
    out[..., 0] = depth // (255 * 255)  # Red channel for the most significant part
    out[..., 1] = (depth - out[..., 0] * 255 * 255) // 255  # Green channel
    out[..., 2] = depth - out[..., 0] * 255 * 255 - out[..., 1] * 255  # Blue channel

    if not (out[..., 2] <= 255).all():
        print(f"Warning: Blue channel values exceeded 255 during encoding. Min: {out[..., 2].min()}, Max: {out[..., 2].max()}")
    return out.astype(np.uint8)


def depth_uint8_decoding(depth_uint8, scale=1000):
    """Decode an 8-bit depth image to floating point values.

    Args:
        depth_uint8 (np.ndarray): 8-bit depth image.
        scale (int): Scale factor.

    Returns:
        out (np.ndarray): Floating point depth image.
    """
    depth_uint8 = depth_uint8.astype(float)
    out = depth_uint8[..., 0] * 255 * 255 + depth_uint8[..., 1] * 255 + depth_uint8[..., 2]
    return out / float(scale)


def distance_to_depth(npy_distance, int_width=1024, int_height=768, flt_focal=886.81):
    """Convert a hypersim distance map to a depth map.

    Args:
        npy_distance (np.ndarray): Hypersim distance map.
        int_width (int): Width of the image.
        int_height (int): Height of the image.
        flt_focal (float): Focal length.

    Returns:
        npy_depth (np.ndarray): Depth map.
    """
    npy_imageplane_x = np.linspace((-0.5 * int_width) + 0.5, (0.5 * int_width) - 0.5, int_width).reshape(
        1, int_width).repeat(int_height, 0).astype(np.float32)[:, :, None]
    npy_imageplane_y = np.linspace((-0.5 * int_height) + 0.5, (0.5 * int_height) - 0.5,
                                   int_height).reshape(int_height, 1).repeat(int_width, 1).astype(np.float32)[:, :, None]
    npy_imageplane_z = np.full([int_height, int_width, 1], flt_focal, np.float32)
    npy_imageplane = np.concatenate(
        [npy_imageplane_x, npy_imageplane_y, npy_imageplane_z], 2)

    npy_depth = npy_distance / np.linalg.norm(npy_imageplane, 2, 2) * flt_focal
    return npy_depth


def disparity_to_depth(disparity, focal_x, stereo_baseline, doffs):
    """Convert disparity to depth using camera parameters.

    Args:
        disparity (np.ndarray): Disparity map.
        focal_x (float): Focal length.
        stereo_baseline (float): Stereo baseline.
        doffs (float): Depth offset.

    Returns:
        depth (np.ndarray): Depth map.
    """
    depth = focal_x * stereo_baseline / (disparity + doffs)
    return depth


def read_flow(file_name):
    """Read .flo file in Middlebury format.

    Args:
        file_name (str): File name.

    Returns:
        data (np.ndarray): Flow map.
    """
    with open(file_name, 'rb') as f:
        magic = np.fromfile(f, np.float32, count=1)
        if 202021.25 != magic:
            status_logging.get_status_logger().write(
                message="Magic number incorrect. Invalid .flo file",
                status_level=status_logging.Status.RUNNING
            )
            return None
        else:
            w = np.fromfile(f, np.int32, count=1)
            h = np.fromfile(f, np.int32, count=1)
            data = np.fromfile(f, np.float32, count=2 * int(w) * int(h))
            # Reshape data into 3D array (columns, rows, bands)
            # The reshape here is for visualization, the original code is (w,h,2)
            return np.resize(data, (int(h), int(w), 2))


def read_image(file_name):
    """Read an image from a file, handling various extensions.

    Args:
        file_name (str): File name.

    Returns:
        img (np.ndarray): Image.
    """
    ext = os.path.splitext(file_name)[-1]
    if ext in ['.png', '.jpg', '.jpeg']:
        img = cv2.imread(file_name)
        if img is None:
            status_logging.get_status_logger().write(
                message=f"RGB image is None for file: {file_name}",
                status_level=status_logging.Status.RUNNING
            )
            raise ValueError(f"RGB image is None for file: {file_name}")
        img = img[..., :3]
        if len(img.shape) == 3:
            img = img[..., ::-1]
        elif len(img.shape) == 2:
            img = np.tile(img[..., None], (1, 1, 3))
        else:
            status_logging.get_status_logger().write(
                message=f"Invalid image format: {file_name}",
                status_level=status_logging.Status.RUNNING
            )
            raise ValueError(f"Invalid image format: {file_name}")
        return img
    elif ext == '.hdf5':
        img = cv2.imread(file_name)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB) / 255.0
        return img
    else:
        status_logging.get_status_logger().write(
            message=f"Unsupported file extension: {file_name}",
            status_level=status_logging.Status.RUNNING
        )
        raise NotImplementedError


def read_pfm(file_name, flip_up_down=False):
    """Read a PFM (Portable Float Map) file.

    Args:
        file_name (str): File name.
        flip_up_down (bool): Flip up down.

    Returns:
        data (np.ndarray): Data.
    """
    pfm_file = open(file_name, 'rb')
    color = None
    width = None
    height = None
    scale = None
    endian = None

    header = pfm_file.readline().rstrip()
    if header.decode('ascii') == 'PF':
        color = True
    elif header.decode('ascii') == 'Pf':
        color = False
    else:
        raise TypeError(f'Not a PFM file. {header} file: {file_name}')

    dim_match = re.search(r'(\d+)\s(\d+)', pfm_file.readline().decode('ascii'))
    if dim_match:
        width, height = map(int, dim_match.groups())
    else:
        raise RuntimeError(f'Malformed PFM header. {pfm_file}')

    scale = float(pfm_file.readline().rstrip())
    if scale < 0:  # little-endian
        endian = '<'
        scale = -scale
    else:
        endian = '>'  # big-endian

    data = np.fromfile(pfm_file, endian + 'f')
    shape = (height, width, 3) if color else (height, width)
    data = np.reshape(data, shape)
    if flip_up_down:
        data = np.flipud(data)  # Flip vertically to match image coordinates
    return data, scale


def depth_to_disparity(depth, return_mask=False):
    """Convert a depth map to a disparity map.

    Args:
        depth (np.ndarray): Depth map.
        return_mask (bool): Return mask.

    Returns:
        disparity (np.ndarray): Disparity map.
        non_negtive_mask (np.ndarray): Non-negative mask.
    """
    if isinstance(depth, torch.Tensor):
        disparity = torch.zeros_like(depth)
    elif isinstance(depth, np.ndarray):
        disparity = np.zeros_like(depth)
    else:
        raise TypeError(f'Invalid depth type: {type(depth)}')
    non_negtive_mask = depth > 0
    disparity[non_negtive_mask] = 1.0 / depth[non_negtive_mask]
    if return_mask:
        return disparity, non_negtive_mask
    else:
        return disparity


def read_gt_3dvlm(file_name, normalize_depth=False, return_disparity=True):
    """Read 3DVLM ground truth depth data.

    Args:
        file_name (str): File name.
        normalize_depth (bool): Normalize depth.
        return_disparity (bool): Return disparity.

    Returns:
        depth (np.ndarray): Depth map.
    """
    depth = np.load(file_name, allow_pickle=True)
    depth = depth.astype(np.float32)
    depth *= 1000.

    if return_disparity:
        depth = depth_to_disparity(depth, return_mask=False)

    if normalize_depth:
        depth = (depth - depth.min()) / (depth.max() - depth.min())

    return depth


def read_gt_fsdv3(disp_file, normalize_depth=False):
    """Read FSDV3 ground truth disparity data.

    Args:
        disp_file (str): Disparity file name.
        normalize_depth (bool): Normalize depth.

    Returns:
        disp (np.ndarray): Disparity map.
    """
    disp_uint = cv2.imread(disp_file)[..., ::-1]
    disp = depth_uint8_decoding(disp_uint)
    disp[disp == 0] = np.inf
    if normalize_depth:
        disp = (disp - disp.min()) / (disp.max() - disp.min())
    return disp


def read_gt_issac_stereo(file_name, normalize_depth=False):
    """Read Issac Stereo ground truth data.

    Args:
        file_name (str): File name.
        normalize_depth (bool): Normalize depth.

    Returns:
        pseduo_depth (np.ndarray): Pseudo depth map.
    """
    pseduo_depth, _ = read_pfm(file_name, flip_up_down=True)
    if normalize_depth:
        pseduo_depth = (pseduo_depth - pseduo_depth.min()) / (pseduo_depth.max() - pseduo_depth.min())
    return pseduo_depth


def read_gt_nvclip(file_name, normalize_depth=False):
    """Read NVCLIP ground truth data.

    Args:
        file_name (str): File name.
        normalize_depth (bool): Normalize depth.

    Returns:
        pseduo_depth (np.ndarray): Pseudo depth map.
    """
    pseduo_depth, _ = read_pfm(file_name, flip_up_down=False)
    if normalize_depth:
        pseduo_depth = (pseduo_depth - pseduo_depth.min()) / (pseduo_depth.max() - pseduo_depth.min())
    return pseduo_depth


def read_gt_surf(file_name, normalize_depth=False, return_disparity=True):
    """Read SURF ground truth data.

    Args:
        file_name (str): File name.
        normalize_depth (bool): Normalize depth.
        return_disparity (bool): Return disparity.

    Returns:
        depth (np.ndarray): Depth map.
    """
    depth = cv2.imread(file_name, cv2.IMREAD_UNCHANGED)
    depth = depth.astype(np.float32)
    depth /= 1000.
    # clip depth value
    depth = np.clip(depth, 0.1, 100)

    if return_disparity:
        depth = depth_to_disparity(depth, return_mask=False)

    if normalize_depth:
        depth = (depth - depth.min()) / (depth.max() - depth.min())

    return depth


def read_gt_crestereo(disp_path, normalize_depth=False):
    """Read CREStereo ground truth disparity data.

    Args:
        disp_path (str): Disparity file name.
        normalize_depth (bool): Normalize depth.

    Returns:
        disp (np.ndarray): Disparity map.
    """
    disp = cv2.imread(disp_path, cv2.IMREAD_UNCHANGED)
    disp = disp.astype(np.float32) / 32
    if normalize_depth:
        disp = (disp - disp.min()) / (disp.max() - disp.min())
    return disp


def read_gt_nyudv2(depth_path, normalize_depth=False, return_disparity=True):
    """Read NYUDv2 ground truth depth data.

    Args:
        depth_path (str): Depth file name.
        normalize_depth (bool): Normalize depth.
        return_disparity (bool): Return disparity.

    Returns:
        depth (np.ndarray): Depth map.
    """
    depth = cv2.imread(depth_path, -1)
    depth = depth.astype(np.float32) / 1000.0

    if return_disparity:
        depth = depth_to_disparity(depth, return_mask=False)

    if normalize_depth:
        depth = (depth - depth.min()) / (depth.max() - depth.min())

    return depth


def read_gt_middlebury(disp_path, mask_path=None, normalize_depth=False):
    """Read Middlebury ground truth disparity and mask data.

    Args:
        disp_path (str): Disparity file name.
        mask_path (str): Mask file name.
        normalize_depth (bool): Normalize depth.

    Returns:
        disp (np.ndarray): Disparity map.
    """
    if '.pfm' in disp_path:
        disp, _ = read_pfm(disp_path, flip_up_down=True)
    elif '.png' in disp_path:
        disp = cv2.imread(disp_path, -1).astype(np.float32)
        disp[disp == 0] = np.inf  # NOTE: 0 means invalid (https://vision.middlebury.edu/stereo/data/scenes2005/)  # noqa: E261
    else:
        raise NotImplementedError(f"Unsupported file extension: {disp_path}")

    if mask_path is not None:
        if os.path.exists(mask_path):
            occ_mask = Image.open(mask_path).convert('L')
            occ_mask = np.asarray(occ_mask, dtype=np.float32)
            disp[occ_mask != 255] = np.inf

    if normalize_depth:
        disp = (disp - disp.min()) / (disp.max() - disp.min())
    return disp


def read_depth(disp_path,  normalize_depth=False):
    """Read a depth map from a file, handling png and pfm extensions.

    Args:
        disp_path (str): Disparity file name.
        normalize_depth (bool): Normalize depth.

    Returns:
        disp (np.ndarray): Disparity map.
    """
    if '.pfm' in disp_path:
        disp, _ = read_pfm(disp_path, flip_up_down=True)
    elif '.png' in disp_path:
        disp = cv2.imread(disp_path, -1).astype(np.float32)
        disp[disp == 0] = np.inf  # NOTE: 0 means invalid (https://vision.middlebury.edu/stereo/data/scenes2005/)  # noqa: E261
    else:
        raise NotImplementedError(f"Unsupported file extension: {disp_path}")

    if normalize_depth:
        disp = (disp - disp.min()) / (disp.max() - disp.min())

    return disp


def read_disparity(file_name):
    """
    Reads a disparity map from a given file path, supporting various formats.

    Handles PNG (encoded disparity) and PFM formats. Sets zero disparity
    values to infinity (invalid).

    Args:
        file_name (str): The path to the disparity file.

    Returns:
        numpy.ndarray: The loaded disparity map as a NumPy array (H, W).
                       Data type is float32.

    Raises:
        NotImplementedError: If the file extension is not supported.
    """
    ext = splitext(file_name)[-1]
    if ext.lower() == '.png':
        disp_uint = cv2.imread(file_name)
        if len(disp_uint.shape) == 3:
            disp_uint = disp_uint[..., ::-1]  # Convert BGR to RGB for decoding
        elif len(disp_uint.shape) == 2:
            disp_uint = np.tile(disp_uint[..., None], (1, 1, 3))  # Convert grayscale to 3-channel for decoding
        disp = depth_uint8_decoding(disp_uint)  # Uses the same decoding logic as depth
        disp[disp == 0] = np.inf  # Set 0 disparity to infinity (invalid)
        return disp
    elif ext.lower() == '.pfm':
        disp = read_pfm(file_name, flip_up_down=True)[0].astype(np.float32)
        if len(disp.shape) == 2:
            return disp  # Single channel PFM
        else:
            # If PFM is 3-channel, assume disparity is in the first channel or
            # the last channel is an alpha/mask channel, and we take the disparity channels
            return disp[:, :, :-1]  # Assuming PFM with (u, v) or (disp, ?) and we want the first channels

    else:
        raise NotImplementedError(f"Disparity file extension '{ext}' not supported for reading.")
