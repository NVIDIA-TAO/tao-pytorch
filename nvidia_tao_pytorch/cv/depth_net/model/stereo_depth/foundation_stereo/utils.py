# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Model utils meant to provide helper functions for generic modules"""

import math
import itertools
from collections import OrderedDict
import numpy as np
from scipy import interpolate
import torch
import torchvision
from PIL import Image
import torch.nn.functional as F


class InputPadder:
    """Pads images such that dimensions are divisible by a given factor."""

    def __init__(self, dims, mode='sintel', divis_by=8, force_square=False):
        """
        Initializes the InputPadder with padding dimensions and mode.

        Args:
            dims (tuple): Dimensions of the input image, typically (B, C, H, W).
            mode (str, optional): Padding mode. 'sintel' pads equally on all sides to center the image,
                                  while other modes (default) pad only on the right and bottom.
                                  Defaults to 'sintel'.
            divis_by (int, optional): The divisor to ensure image dimensions are divisible by.
                                      Defaults to 8.
            force_square (bool, optional): If True, pads the image to make it square while
                                           maintaining divisibility. Defaults to False.
        """
        self.ht, self.wd = dims[-2:]
        if force_square:
            max_side = max(self.ht, self.wd)
            pad_ht = ((max_side // divis_by) + 1) * divis_by - self.ht
            pad_wd = ((max_side // divis_by) + 1) * divis_by - self.wd
        else:
            pad_ht = (((self.ht // divis_by) + 1) * divis_by - self.ht) % divis_by
            pad_wd = (((self.wd // divis_by) + 1) * divis_by - self.wd) % divis_by

        if mode == 'sintel':
            # Pad [left, right, top, bottom]
            self._pad = [pad_wd // 2, pad_wd - pad_wd // 2, pad_ht // 2, pad_ht - pad_ht // 2]
        else:
            # Pad [left, right, top, bottom]
            self._pad = [pad_wd // 2, pad_wd - pad_wd // 2, 0, pad_ht]

    def pad(self, *inputs):
        """
        Pads input tensors according to the initialized padding scheme.

        Args:
            *inputs (torch.Tensor): One or more input tensors to be padded.
                                    All tensors must have 4 dimensions (B, C, H, W).

        Returns:
            list: A list of padded tensors.
        """
        assert all(x.ndim == 4 for x in inputs), "All inputs must have 4 dimensions (B, C, H, W)."
        return [F.pad(x, self._pad, mode='replicate') for x in inputs]

    def unpad(self, x):
        """
        Removes padding from the input tensor to restore original dimensions.

        Args:
            x (torch.Tensor): The padded input tensor, must have 4 dimensions (B, C, H, W).

        Returns:
            torch.Tensor: The unpadded tensor.
        """
        assert x.ndim == 4, "Input must have 4 dimensions (B, C, H, W)."
        ht, wd = x.shape[-2:]
        # Calculate crop coordinates [top, bottom, left, right]
        c = [self._pad[2], ht - self._pad[3], self._pad[0], wd - self._pad[1]]
        return x[..., c[0]: c[1], c[2]: c[3]]


class CenterPadding(torch.nn.Module):
    """
    Applies center padding to input tensors to make their spatial dimensions
    multiples of a given factor.

    This module is designed to ensure that the height, width, and optionally depth
    dimensions of an input tensor are perfectly divisible by a specified `multiple`.
    Padding is applied symmetrically (half on one side, half on the other) to
    keep the original content centered within the new, padded dimensions. This is
    useful for preparing inputs for architectures like U-Nets or models with
    downsampling/upsampling layers that require specific input divisibility.
    """

    def __init__(self, multiple):
        """
        Initializes the CenterPadding module.

        Args:
            multiple (int): The integer value by which the spatial dimensions
                            (H, W, and optionally D for 3D tensors) of the
                            input tensor should be divisible after padding.
        """
        super().__init__()
        self.multiple = multiple

    def _get_pad(self, size):
        """
        Calculates the required padding for a single spatial dimension.

        Given an original dimension size, it computes the smallest new size
        that is a multiple of `self.multiple` and then determines how much
        padding needs to be added to the left/top and right/bottom sides.

        Args:
            size (int): The current size of a spatial dimension (e.g., H or W).

        Returns:
            tuple: A tuple `(pad_size_left, pad_size_right)` indicating the
                   number of pixels/voxels to add to the left/top side and
                   the right/bottom side of that dimension, respectively.
        """
        # Calculate the new size, which is the smallest multiple of self.multiple
        # that is greater than or equal to the current size.
        new_size = math.ceil(size / self.multiple) * self.multiple

        # Calculate the total padding needed for this dimension.
        pad_size = new_size - size

        # Distribute padding symmetrically: half on the left/top side, the rest on the right/bottom.
        pad_size_left = pad_size // 2
        pad_size_right = pad_size - pad_size_left
        return pad_size_left, pad_size_right

    @torch.inference_mode()
    def forward(self, x):
        """
        Applies center padding to the input tensor's spatial dimensions.

        The padding is applied in reverse order of dimensions (W, H, then D if present)
        as expected by `torch.nn.functional.pad`.

        Args:
            x (torch.Tensor): The input tensor. This can be of shape (B, C, H, W)
                              for 2D images or (B, C, D, H, W) for 3D data.
                              Padding is applied to the dimensions starting from the
                              last spatial dimension backwards.

        Returns:
            torch.Tensor: The padded tensor, with its spatial dimensions
                          now multiples of `self.multiple`.
        """
        # Calculate padding for each spatial dimension.
        # `x.shape[:1:-1]` gets the spatial dimensions in reverse order (W, H, and potentially D).
        # `itertools.chain.from_iterable` flattens the list of (left_pad, right_pad) tuples
        # into a single list required by F.pad (e.g., [pad_w_left, pad_w_right, pad_h_left, pad_h_right]).
        pads = list(itertools.chain.from_iterable(self._get_pad(m) for m in x.shape[:1:-1]))

        # Apply padding using `torch.nn.functional.pad` in replication mode.
        # 'replicate' mode fills new pixels by copying the values from the edge pixels.
        return F.pad(x, pads, mode='replicate')


def get_resize_keep_aspect_ratio(height, width, divider=16, max_h=1232, max_w=1232):
    """
    Resizes an image while keeping the aspect ratio
    and ensuring divisibility by a given factor, with optional maximum dimensions.

    This function calculates new dimensions for an image such that its aspect
    ratio is preserved, both height and width are divisible by `divider`, and
    neither dimension exceeds `max_h` or `max_w` respectively.

    Args:
        height (int): Original height of the image.
        width (int): Original width of the image.
        divider (int, optional): The factor by which dimensions must be divisible. Defaults to 16.
        max_h (int, optional): Maximum allowed height after resizing. Defaults to 1232.
        max_w (int, optional): Maximum allowed width after resizing. Defaults to 1232.

    Returns:
        tuple: A tuple (h_resize, w_resize) representing the new height and width.

    Raises:
        AssertionError: If max_h or max_w are not divisible by the divider,
                        ensuring that the maximum limits themselves are valid
                        targets for rounding.
    """
    assert max_h % divider == 0, "max_h must be divisible by divider."
    assert max_w % divider == 0, "max_w must be divisible by divider."

    def round_by_divider(x):
        """Helper function to round a dimension up to the nearest multiple of the divider."""
        return int(np.ceil(x / divider) * divider)

    # Initial resizing to ensure divisibility by `divider`
    h_resize = round_by_divider(height)
    w_resize = round_by_divider(width)

    # Adjust if resized dimensions exceed maximum allowed dimensions while preserving aspect ratio
    if h_resize > max_h or w_resize > max_w:
        if h_resize > w_resize:
            # If height is the limiting factor, scale width proportionally and round
            w_resize = round_by_divider(w_resize * max_h / h_resize)
            h_resize = max_h
        else:
            # If width is the limiting factor, scale height proportionally and round
            h_resize = round_by_divider(h_resize * max_w / w_resize)
            w_resize = max_w

    return int(h_resize), int(w_resize)


def forward_interpolate(flow):
    """
    Performs forward interpolation on an optical flow field.

    This function takes an optical flow field and warps it based on its own
    values. It simulates moving pixels from their original positions (x0, y0)
    to new positions (x1, y1) defined by the flow (dx, dy). Pixels in the
    output grid that do not have a corresponding source pixel are filled using
    nearest neighbor interpolation, effectively handling occlusions or areas
    that become visible due to the flow.

    Args:
        flow (torch.Tensor): The input optical flow field, typically of shape (2, H, W).
                             The first channel `flow[0]` represents the horizontal displacement (dx),
                             and the second channel `flow[1]` represents the vertical displacement (dy).
                             The input is expected to be a PyTorch tensor.

    Returns:
        torch.Tensor: The interpolated flow field, with the same shape as the input.
                      This tensor will be a float32 PyTorch tensor.
    """
    # Detach from computation graph, move to CPU, and convert to NumPy array for SciPy operations
    flow = flow.detach().cpu().numpy()

    # Separate the flow components
    dx, dy = flow[0], flow[1]

    # Get height and width of the flow field
    ht, wd = dx.shape

    # Create meshgrid of original pixel coordinates (x0, y0)
    x0, y0 = np.meshgrid(np.arange(wd), np.arange(ht))

    # Calculate the target coordinates (x1, y1) by adding flow components to original coordinates
    x1 = x0 + dx
    y1 = y0 + dy

    # Flatten all coordinate and flow arrays for `griddata`
    x1, y1, dx, dy = x1.ravel(), y1.ravel(), dx.ravel(), dy.ravel()

    # Identify valid points within the image boundaries after displacement
    valid = (x1 >= 0) & (x1 < wd) & (y1 >= 0) & (y1 < ht)

    # Filter coordinates and flow values to keep only valid points
    x1, y1, dx, dy = x1[valid], y1[valid], dx[valid], dy[valid]

    # Perform nearest neighbor interpolation. `griddata` interpolates values (dx, dy)
    # from scattered points (x1, y1) onto a regular grid (x0, y0).
    # `fill_value=0` means areas not covered by valid points will be filled with 0.
    flow_x = interpolate.griddata((x1, y1), dx, (x0, y0), method='nearest', fill_value=0)
    flow_y = interpolate.griddata((x1, y1), dy, (x0, y0), method='nearest', fill_value=0)

    # Stack the interpolated flow components back into a single flow field
    flow = np.stack([flow_x, flow_y], axis=0)

    # Convert the NumPy array back to a PyTorch float tensor
    return torch.from_numpy(flow).float()


def freeze_model(model):
    """
    Freezes a PyTorch model by setting all parameters and buffers to non-trainable.

    This function is commonly used to prevent a pre-trained model or specific
    parts of a model from being updated during training. It sets the model to
    evaluation mode and disables gradient computation for all its parameters,
    making them effectively 'frozen'. Buffers, such as those in BatchNorm layers,
    are also affected by `model.eval()` as their running statistics are no longer
    updated.

    Args:
        model (torch.nn.Module): The PyTorch model to freeze. This model will be
                                 modified in-place.

    Returns:
        torch.nn.Module: The frozen model.
    """
    model.eval()  # Set the model to evaluation mode. This affects BatchNorm and Dropout layers.
    for param in model.parameters():
        param.requires_grad = False  # Disable gradient computation for all model parameters.
    for buffer in model.buffers():
        buffer.requires_grad = False
    return model


def normalize_image(img):
    """
    Normalizes an image tensor using ImageNet mean and standard deviation.

    This is a standard preprocessing step for many deep learning models that
    have been pre-trained on the ImageNet dataset. The normalization converts
    pixel values to a standard range, which can help with model convergence.
    The input image is first scaled from the typical [0, 255] range to [0, 1]
    before applying the channel-wise normalization.

    Args:
        img (torch.Tensor): The input image tensor. Expected to have pixel
                            values in the range [0, 255] and typically of shape
                            (C, H, W) for a single image or (B, C, H, W) for a batch.

    Returns:
        torch.Tensor: The normalized image tensor. It will have the same shape
                      as the input but with pixel values adjusted according to
                      ImageNet statistics. The tensor will be contiguous in memory.
    """
    # Define the normalization transform with ImageNet's mean and standard deviation
    transform = torchvision.transforms.Normalize(
        mean=[0.485, 0.456, 0.406],  # Mean for R, G, B channels
        std=[0.229, 0.224, 0.225],   # Standard deviation for R, G, B channels
        inplace=False                # Do not perform normalization in-place
    )
    # First, scale the image pixel values from [0, 255] to [0, 1]
    # Then apply the defined normalization transform.
    # .contiguous() ensures the tensor is stored contiguously in memory, which can improve performance.
    transform_image = transform(img / 255.0).contiguous()
    return transform_image


def unnormalize(normalized_tensor, mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]):
    """
    Unnormalizes a tensor image with channel-wise mean and standard deviation.

    This function reverses the normalization process performed by `normalize_image`
    (or similar operations). It's useful for visualizing normalized images or
    converting them back to a displayable pixel value range (e.g., [0, 1] or [0, 255])
    after model inference.

    The unnormalization formula is:
    `unnormalized_value = normalized_value * std + mean`

    Args:
        normalized_tensor (torch.Tensor): Normalized image tensor, typically of shape
                                          (C, H, W) or (B, C, H, W).
        mean (sequence, optional): Sequence of means (one for each channel) used during
                                   the original normalization. Defaults to ImageNet means.
        std (sequence, optional): Sequence of standard deviations (one for each channel)
                                  used during the original normalization. Defaults to ImageNet stds.

    Returns:
        torch.Tensor: Unnormalized image tensor. Its pixel values will be
                      restored to the range they were in *before* the mean/std
                      normalization (e.g., typically [0, 1]).
    """
    # Convert mean and std lists to PyTorch tensors and reshape them
    # to (C, 1, 1) for broadcasting across H and W dimensions.
    mean = torch.tensor(mean).reshape(-1, 1, 1)
    std = torch.tensor(std).reshape(-1, 1, 1)

    # If the input tensor is on a CUDA device, move mean and std to CUDA as well
    if normalized_tensor.is_cuda:
        mean = mean.to('cuda')
        std = std.to('cuda')

    # Apply the unnormalization formula
    unnormalized_tensor = normalized_tensor * std + mean
    return unnormalized_tensor


def write_image(image, save_path):
    """
    Writes an image to the specified file path using OpenCV.

    This function takes an image (assumed to be in RGB format) and
    then attempts to save the image to the given path.

    Args:
        image (numpy.ndarray): The image array to save. Expected to be a NumPy array
                               in RGB format (e.g., shape (H, W, 3) or (H, W)).
                               Pixel values are typically expected to be in the range [0, 255].
        save_path (str): The full path including the filename and extension
                         where the image will be saved (e.g., "output/my_image.png").
    """
    pil_image = Image.fromarray(image)
    pil_image.save(save_path)


def get_filename_from_path(full_path, data_name):
    """
    Extracts a standardized filename from a given full path based on the dataset name.

    This function acts as a dispatcher, calling specific helper functions to
    extract filenames according to the conventions of different datasets.
    It supports 'middlebury', 'eth3d', 'kitti', and 'stereodataset'.

    Args:
        full_path (str): The complete file path of an image, including its name and extension.
        data_name (str): The name of the dataset as a string (case-insensitive).
                         Supported values are 'middlebury', 'eth3d', 'kitti', 'stereodataset'.

    Returns:
        str: The extracted filename, formatted according to the specific dataset's conventions.

    Raises:
        NotImplementedError: If the provided `data_name` is not recognized or supported.
    """
    # Dictionary mapping dataset names to their respective filename extraction functions
    DATASET = {'middlebury': get_middlebury_filenames,
               'eth3d': get_eth3d_filenames,
               'kitti': get_kitti_filenames,
               'fsd': get_stereodata_filenames,
               'isaacrealdataset': get_stereodata_filenames,
               'genericdataset': get_stereodata_filenames,
               }

    # Convert data_name to lowercase for case-insensitive matching
    data_name_lower = data_name.lower()

    # Check if the requested data_name is supported
    if data_name_lower not in DATASET:
        raise NotImplementedError(f'{data_name} data file name not implemented!')

    # Call the appropriate filename extraction function and return its result
    return DATASET[data_name_lower](full_path)


def get_eth3d_filenames(full_path):
    """
    Extracts the dataset-specific filename for ETH3D images from their full path.

    This function identifies known ETH3D sequence names within the path components.
    If a known sequence name is found, it constructs the filename with a '.png' extension.
    Otherwise, it defaults to using the last component of the path as the filename.

    Args:
        full_path (str): The full path to an ETH3D image file. This typically includes
                         directories like '/path/to/ETH3D/dataset_name/image_file.png'.

    Returns:
        str: The extracted ETH3D filename (e.g., 'delivery_area_1l.png' or 'image.png').
    """
    # A list of predefined ETH3D scene/file names
    file_names = ['delivery_area_1l', 'electro_1l', 'facade_1s', 'playground_2s',
                  'terrains_1s', 'delivery_area_1s', ' electro_1s',  'forest_1s',
                  'playground_3l', 'terrains_2l', 'delivery_area_2l', 'electro_2l',
                  'forest_2s', 'playground_3s', 'terrains_2s', 'delivery_area_2s',
                  'electro_2s', 'playground_1l', 'terrace_1s', 'delivery_area_3l',
                  'electro_3l', 'playground_1s', 'terrace_2s', 'delivery_area_3s',
                  'electro_3s', 'playground_2l', 'terrains_1l']

    set_name = None
    # Split the full path into components to check for known dataset names
    path_components = full_path.split('/')

    # Iterate through the known file names to find a match in the path components
    for name in file_names:
        if name in path_components:
            set_name = name + '.png'  # If found, construct the filename with .png
            break  # Exit loop once a match is found

    # If no known file name was found in the path, default to the last component of the path
    if set_name is None:
        set_name = path_components[-1]

    return set_name


def get_kitti_filenames(full_path):
    """
    Extracts the filename for KITTI images from their full path.

    For KITTI, the filename is typically the last component of the file path.
    This function splits the path by '/' and returns the last element.

    Args:
        full_path (str): The complete file path of a KITTI image, e.g., '/path/to/kitti/dataset/image_000000.png'.

    Returns:
        str: The extracted filename, e.g., 'image_000000.png'.
    """
    return full_path.split('/')[-1]


def get_stereodata_filenames(full_path):
    """Return the input path stripped of its leading separator.

    Caller is expected to ``os.makedirs(os.path.dirname(target))`` before
    writing, mirroring the mono path which recreates the directory tree
    under the output root. This keeps outputs collision-free at any
    nesting depth (e.g. ``<seq>/<view>/<file>``).

    Args:
        full_path (str): Absolute or relative input path.

    Returns:
        str: ``full_path`` with the leading ``/`` removed.
    """
    return full_path.lstrip('/')


def get_middlebury_filenames(full_path):
    """
    Extracts the dataset-specific filename for Middlebury evaluation images from their full path.

    This function attempts to find a known Middlebury scene name within the full path.
    If a match is found, it constructs the filename using that scene name appended with '.png'.
    If no specific scene name is matched, it defaults to using the last component of the path
    as the filename.

    Args:
        full_path (str): The complete file path of a Middlebury image, e.g., '/path/to/Middlebury/Adirondack/im0.png'.

    Returns:
        str: The extracted Middlebury filename (e.g., 'Adirondack.png' or 'im0.png').
    """
    # A list of predefined Middlebury scene names
    file_names = ['Adirondack', 'ArtL', 'Jadeplant', 'Motorcycle',
                  'MotorcycleE', 'Piano', 'PianoL', 'Pipes', 'Playroom',
                  'Playtable', 'PlaytableP', 'Recycle', 'Shelves', 'Teddy', 'Vintage']

    set_name = None

    # Iterate through the known file names to find if any are present in the full path
    full_path_check = full_path.split('/')[-3:]  # check the last three path names
    for name in file_names:
        if name in full_path_check:
            set_name = name + '.' + full_path_check[-1].split('.')[-1]  # If a known scene name is found
            break  # Once a match is found, no need to check further

    # If no specific Middlebury scene name was found in the path,
    # default to using the last component of the path (the actual file name)
    if set_name is None:
        set_name = full_path.split('/')[-1]

    return set_name


class LayerNorm2d(torch.nn.LayerNorm):
    """LayerNorm for channels_first tensors with 2D spatial dimensions (N, C, H, W)."""

    def __init__(self, normalized_shape, eps=1e-6):
        """Initializes LayerNorm2d.

        Args:
            normalized_shape (int): Channel dimension.
            eps (float, optional): Small constant for numerical stability. Defaults to 1e-6.
        """
        super().__init__(normalized_shape, eps=eps)

    def forward(self, x) -> torch.Tensor:
        """Forward pass of LayerNorm2d.

        Args:
            x (torch.Tensor): Input tensor of shape (B, C, H, W).

        Returns:
            torch.Tensor: Normalized output tensor.
        """
        if x.is_contiguous():
            return (F.layer_norm(x.permute(0, 2, 3, 1),
                    self.normalized_shape,
                    self.weight, self.bias, self.eps).permute(0, 3, 1, 2).contiguous()
                    )
        s, u = torch.var_mean(x, dim=1, keepdim=True)
        x = (x - u) * torch.rsqrt(s + self.eps)
        x = x * self.weight[:, None, None] + self.bias[:, None, None]
        return x


class LayerNorm3d(torch.nn.LayerNorm):
    """LayerNorm for channels_first tensors with 3D spatial dimensions (N, C, D, H, W)."""

    def __init__(self, normalized_shape, eps=1e-6):
        super().__init__(normalized_shape, eps=eps)

    def forward(self, x) -> torch.Tensor:
        """Forward pass of LayerNorm3d.

        Args:
            x (torch.Tensor): Input tensor of shape (B, C, D, H, W).

        Returns:
            torch.Tensor: Normalized output tensor.
        """
        return (
            F.layer_norm(
                x.permute(0, 2, 3, 4, 1).contiguous(),
                self.normalized_shape, self.weight, self.bias, self.eps)
            .permute(0, 4, 1, 2, 3)
            .contiguous()
        )


def bilinear_sampler(img, coords, mask=False, low_memory=False):
    """
    Wrapper for torch.nn.functional.grid_sample, uses pixel coordinates.

    This function performs bilinear sampling on an image tensor using a grid
    of pixel coordinates. It normalizes the coordinates from pixel space to
    the [-1, 1] range required by `grid_sample`. It includes an option for
    low-memory processing for large batches by breaking down the operation.
    This implementation is specifically optimized for stereo problems where
    the input image height `H` is expected to be 1 and `ygrid` is uniform.

    Args:
        img (torch.Tensor): The input image tensor (B, C, H, W).
        coords (torch.Tensor): The grid of sampling coordinates (B, H_out, W_out, 2),
                               where the last dimension contains (x, y) pixel coordinates.
        mask (bool, optional): If True, also returns a mask indicating valid sampled points
                               (i.e., points that fall within the original image boundaries).
                               Defaults to False.
        low_memory (bool, optional): If True, processes the batch in smaller chunks
                                     to conserve GPU memory. This is beneficial for
                                     very large input batches. Defaults to False.

    Returns:
        torch.Tensor or tuple:
            - If `mask` is False: The sampled image tensor (B, C, H_out, W_out).
            - If `mask` is True: A tuple containing (sampled_img, mask), where
                                 `sampled_img` is the sampled image tensor and `mask`
                                 is a float tensor of shape (B, H_out, W_out, 1)
                                 indicating valid sample locations (1.0 for valid, 0.0 for invalid).

    Raises:
        AssertionError: If `torch.unique(ygrid).numel()` is not 1 or `H` is not 1,
                        indicating that the input does not conform to the expected
                        stereo problem characteristics for this optimized sampler.
    """
    H, W = img.shape[-2:]
    xgrid, ygrid = coords.split([1, 1], dim=-1)

    # Normalize x coordinates from [0, W-1] to [-1, 1]
    xgrid = 2 * xgrid / (W - 1) - 1

    # Assertions specific to stereo problems where height is 1 and ygrid is uniform
    assert torch.unique(ygrid).numel() == 1 and H == 1, \
        "This function is designed for stereo problems where H=1 and ygrid is constant."

    # Concatenate x and y grids to form the sampling grid for F.grid_sample
    grid = torch.cat([xgrid, ygrid], dim=-1).to(img.dtype)

    if low_memory:
        B = img.shape[0]
        out = []
        bs = 102400  # Batch size for low memory processing
        # Process the batch in chunks
        for b in np.arange(0, B, bs):
            tmp = F.grid_sample(img[b:b + bs], grid[b:b + bs], align_corners=True)
            out.append(tmp)
        img = torch.cat(out, dim=0)
    else:
        # Perform grid sampling directly
        img = F.grid_sample(img, grid, align_corners=True)

    if mask:
        # Create a mask for valid sampled points in normalized coordinate space
        mask_valid_x = (xgrid > -1) & (xgrid < 1)
        mask_valid_y = (ygrid > -1) & (ygrid < 1)
        mask = mask_valid_x & mask_valid_y
        return img, mask.float()
    return img


def coords_grid(batch, ht, wd):
    """
    Generates a 2D coordinate grid for a given batch size, height, and width.

    The grid contains pixel coordinates (x, y) for each location in a 2D plane.
    The coordinates start from (0,0) at the top-left corner.

    Args:
        batch (int): The number of grids to generate (batch size).
        ht (int): The height of the grid.
        wd (int): The width of the grid.

    Returns:
        torch.Tensor: A tensor of shape (batch, 2, ht, wd) containing the
                      (x, y) coordinates for each pixel. The first channel
                      corresponds to x-coordinates, and the second to y-coordinates.
    """
    # Create 1D tensors for y (height) and x (width) coordinates
    # torch.meshgrid returns two 2D tensors: one with y-coordinates varying by row,
    # and one with x-coordinates varying by column.
    coords = torch.meshgrid(torch.arange(ht), torch.arange(wd), indexing='ij')

    # Stack the meshgrid outputs:
    # `coords[::-1]` reverses the order, so it becomes (x_coords_2d, y_coords_2d)
    # torch.stack then stacks them into a (2, ht, wd) tensor.
    coords = torch.stack(coords[::-1], dim=0).float()

    # Add a batch dimension at the beginning and repeat it `batch` times.
    # The final shape will be (batch, 2, ht, wd).
    return coords[None].repeat(batch, 1, 1, 1)


def disparity_regression(x, maxdisp):
    """
    Computes disparity regression from a probability volume.

    This function calculates the expected disparity value for each pixel
    given a probability distribution over possible disparities. It assumes `x`
    represents a soft-maxed probability distribution or a similar confidence
    measure across the disparity dimension. The output is a single disparity value
    per pixel, derived as a weighted sum of possible disparity values.

    Args:
        x (torch.Tensor): The input probability volume, typically of shape (B, D, H, W),
                          where D is the number of possible disparity values (maxdisp).
                          `x[b, d, h, w]` represents the probability or confidence of
                          disparity `d` at spatial location (h, w) for batch item `b`.
        maxdisp (int): The maximum possible disparity value. This should correspond
                       to the size of the disparity dimension `D` in `x`.

    Returns:
        torch.Tensor: A tensor of shape (B, 1, H, W) representing the regressed
                      disparity map. Each element contains the expected disparity
                      value for the corresponding pixel.

    Raises:
        AssertionError: If the input tensor `x` does not have exactly 4 dimensions.
    """
    assert len(x.shape) == 4, "Input to disparity_regression must be a 4D tensor (B, D, H, W)."

    # Create a tensor representing the actual disparity values [0, 1, ..., maxdisp-1]
    disp_values = torch.arange(0, maxdisp, dtype=x.dtype, device=x.device)

    # Reshape `disp_values` to (1, maxdisp, 1, 1) to enable broadcasting
    # when multiplied with the input probability volume `x`.
    disp_values = disp_values.reshape(1, maxdisp, 1, 1)

    # Perform element-wise multiplication of probabilities with disparity values
    # and then sum along the disparity dimension (dim=1) to get the expected disparity.
    # `keepdim=True` ensures the output maintains the (B, 1, H, W) shape.
    return torch.sum(x * disp_values, 1, keepdim=True)


def upsample_disp_raft(disp, mask, scale_factor=4):
    """
    Upsamples a disparity field using a learned mask, inspired by the RAFT architecture.

    This function leverages a high-resolution "mask" to perform a detailed
    upsampling of a low-resolution disparity map. The mask typically represents
    weights for a 3x3 local neighborhood contribution for each upsampled pixel.
    The process involves unfolding the low-resolution disparity, applying the
    soft-maxed mask as weights, and then reshaping to the high resolution.

    Args:
        disp (torch.Tensor): The low-resolution disparity map of shape (N, 1, H, W).
        mask (torch.Tensor): The upsampling mask of shape (N, 1, 9 * S*S, H, W),
                             where S is the `scale_factor`. This mask typically
                             contains weights for 9 (3x3) neighborhood elements,
                             and then `S*S` sub-pixel locations within each upsampled grid cell.
        scale_factor (int, optional): The integer factor by which to upsample the
                                      disparity map (e.g., 2, 4, 8). Defaults to 4.

    Returns:
        torch.Tensor: The high-resolution upsampled disparity map of shape
                      (N, 1, H * scale_factor, W * scale_factor).
    """
    N, _, H, W = disp.shape

    # Reshape the mask to explicitly separate the 9-element neighborhood and
    # the scale_factor*scale_factor sub-pixel contributions.
    mask = mask.view(N, 1, 9, scale_factor, scale_factor, H, W)

    # Apply softmax over the 9-element neighborhood dimension (dim=2) to
    # ensure that the weights for each local patch sum to 1.
    mask = torch.softmax(mask, dim=2)

    # Unfold the low-resolution disparity map.
    # F.unfold takes a 3x3 sliding window with padding=1, effectively creating
    # a feature map where each 'pixel' contains the 9 elements from its 3x3 neighborhood.
    # We multiply by scale_factor *before* unfolding, assuming disparities are scaled.
    up_disp = F.unfold(scale_factor * disp, [3, 3], padding=1)

    # Reshape the unfolded disparity to match the dimensions of the mask for element-wise multiplication.
    # The -1 in reshape infers the correct size for the second dimension (C*k*k, where C=1, k=3).
    up_disp = up_disp.view(N, 1, 9, 1, 1, H, W)

    # Perform the weighted sum. This sums the 9 neighborhood contributions based on the mask weights.
    up_disp = torch.sum(mask * up_disp, dim=2)

    # Permute and reshape the tensor to interleave the upsampled pixels correctly
    # to form the final high-resolution disparity map.
    up_disp = up_disp.permute(0, 1, 4, 2, 5, 3)  # N, 1, H_upscaled, W_upscaled, factors -> N, 1, H*S, W*S
    return up_disp.reshape(N, 1, scale_factor * H, scale_factor * W)


def context_upsample(disp_low, up_weights):
    """
    Upsamples a low-resolution disparity map using context-based upsampling weights.

    This function performs an implicit upsampling by applying a set of learned
    `up_weights` to local neighborhoods of the `disp_low` map. The `up_weights`
    determine how the 3x3 neighborhood around each low-resolution pixel contributes
    to the corresponding high-resolution `scale_factor` x `scale_factor` block.

    Args:
        disp_low (torch.Tensor): The low-resolution disparity map of shape (B, C, H, W).
                                 Typically, C=1 for disparity.
        up_weights (torch.Tensor): The upsampling weights, typically of shape (B, 9, H_out, W_out),
                                   where `H_out` and `W_out` are the target high resolution
                                   dimensions (e.g., H * 4, W * 4). These weights are applied
                                   to the 9 elements of a 3x3 neighborhood.

    Returns:
        torch.Tensor: The high-resolution upsampled disparity map of shape
                      (B, H_out, W_out). The channel dimension is squeezed out
                      as the output is a single disparity value per pixel.
    """
    b, c, h, w = disp_low.shape

    # Unfold the low-resolution disparity map into 3x3 local patches.
    # `F.unfold` extracts sliding windows from the input.
    # The output shape will be (B, C * kernel_H * kernel_W, L),
    # where L is (H*W) after reshaping to (B, C*k*k, H, W).
    disp_unfold = F.unfold(disp_low.reshape(b, c, h, w), kernel_size=3, padding=1, stride=1).reshape(b, -1, h, w)

    # Interpolate the unfolded patches to the target high resolution.
    # Using 'nearest' mode for interpolation. The target size is assumed to be 4x larger.
    # After interpolation, reshape back to (B, 9, H*4, W*4) to align with `up_weights`.
    disp_unfold = F.interpolate(
        disp_unfold, (h * 4, w * 4), mode='nearest').reshape(b, 9, h * 4, w * 4)

    # Perform the weighted sum. Element-wise multiply the unfolded disparities with
    # the upsampling weights and sum along the second dimension (the 9 neighborhood elements).
    disp = (disp_unfold * up_weights).sum(1)

    return disp


def build_correlation_volume(refimg_feature, targetimg_feature, maxdisp):
    """
    Builds a correlation volume between a reference and a target feature map.

    The correlation volume represents the similarity (correlation score)
    between features from the `refimg_fea` at a given pixel and features from
    `targetimg_fea` shifted by various disparity values up to `maxdisp`.
    This is a fundamental component in many stereo matching algorithms.

    Args:
        refimg_feature (torch.Tensor): Feature map from the reference (left) image,
                                   of shape (B, C, H, W).
        targetimg_feature (torch.Tensor): Feature map from the target (right) image,
                                    of shape (B, C, H, W).
        maxdisp (int): The maximum disparity value to consider for building the volume.
                       The volume will have `maxdisp` disparity levels.

    Returns:
        torch.Tensor: The correlation volume of shape (B, 1, maxdisp, H, W).
                      Each entry `volume[b, 0, d, h, w]` contains the correlation
                      score between `refimg_fea[b, :, h, w]` and
                      `targetimg_fea[b, :, h, w-d]`.
    """
    B, _, H, W = refimg_feature.shape

    # Initialize an empty volume to store correlation costs.
    # The channel dimension is 1 as correlation typically results in a scalar similarity.
    volume = refimg_feature.new_zeros([B, 1, maxdisp, H, W])

    # Iterate through each possible disparity value
    for i in range(maxdisp):
        if i > 0:
            # For non-zero disparities, calculate correlation between the reference
            # features and the target features shifted by `i` pixels to the left.
            # `i:` for ref_img_fea means starting from column `i`.
            # `:-i` for target_img_fea means ending before the last `i` columns.
            volume[:, :, i, :, i:] = correlation(
                refimg_feature[:, :, :, i:], targetimg_feature[:, :, :, :-i])
        else:
            # For disparity 0, calculate correlation without any shift.
            volume[:, :, i, :, :] = correlation(refimg_feature, targetimg_feature)

    return volume.contiguous()  # Ensure the tensor is contiguous in memory


def build_concat_volume(refimg_feature, targetimg_feature, maxdisp):
    """
    Builds a concatenated feature volume for stereo matching.

    This volume is constructed by concatenating the reference image's feature map
    with a shifted version of the target image's feature map along the channel dimension
    for each possible disparity level. This provides a rich context for a subsequent
    3D convolution network to learn disparity.

    Args:
        refimg_feature (torch.Tensor): Feature map from the reference (left) image,
                                   of shape (B, C, H, W).
        targetimg_feature (torch.Tensor): Feature map from the target (right) image,
                                    of shape (B, C, H, W).
        maxdisp (int): The maximum disparity value to consider. The volume will
                       have `maxdisp` disparity levels.

    Returns:
        torch.Tensor: The concatenated volume of shape (B, 2 * C, maxdisp, H, W).
                      For each disparity `d`, `volume[b, :C, d, h, w]` contains
                      `refimg_fea[b, :, h, w]` and `volume[b, C:, d, h, w]`
                      contains `targetimg_fea[b, :, h, w-d]`.
    """
    B, C, H, W = refimg_feature.shape

    # Initialize an empty volume to store the concatenated features.
    # The channel dimension is 2 * C because we are concatenating two C-channel feature maps.
    volume = refimg_feature.new_zeros([B, 2 * C, maxdisp, H, W])

    # Iterate through each possible disparity value
    for i in range(maxdisp):
        if i > 0:
            # For non-zero disparities, fill the first half of channels with
            # reference features and the second half with shifted target features.
            volume[:, :C, i, :, :] = refimg_feature[:, :, :, :]
            volume[:, C:, i, :, i:] = targetimg_feature[:, :, :, :-i]
        else:
            # For disparity 0, fill with reference and unshifted target features.
            volume[:, :C, i, :, :] = refimg_feature
            volume[:, C:, i, :, :] = targetimg_feature

    volume = volume.contiguous()  # Ensure the tensor is contiguous in memory
    return volume


def correlation(feature1, feature2):
    """
    Computes the correlation cost between two feature maps.

    This function calculates the element-wise product of two feature maps
    and then sums the result along the channel dimension. This operation
    measures the similarity between corresponding feature vectors at each
    spatial location.

    Args:
        feature1 (torch.Tensor): The first feature map (e.g., from the reference image),
                             of shape (B, C, H, W).
        feature2 (torch.Tensor): The second feature map (e.g., from the target image),
                             of shape (B, C, H, W).

    Returns:
        torch.Tensor: The correlation cost tensor of shape (B, 1, H, W).
                      Each element represents the similarity score at a given
                      spatial location.
    """
    # Perform element-wise multiplication and then sum along the channel dimension (dim=1).
    # `keepdim=True` ensures the output maintains a channel dimension of size 1.
    return torch.sum(feature1 * feature2, dim=1, keepdim=True)


def groupwise_correlation(feature1, feature2, num_groups, normalize: bool = True):
    """
    Computes group-wise correlation between two feature maps.

    This method divides the channels of the feature maps into `num_groups`
    and then computes the correlation independently within each group.
    When ``normalize=True`` (default), features within each group are
    L2-normalized before the dot product (cosine similarity). When
    ``normalize=False``, raw inner-product correlation is used — required
    by the FFS commercial bp2 distillation, which was trained without
    feature normalization.

    Args:
        feature1 (torch.Tensor): The first feature map (B, C, H, W).
        feature2 (torch.Tensor): The second feature map (B, C, H, W).
        num_groups (int): The number of groups to divide the channels into.
                          The total number of channels `C` must be divisible by `num_groups`.
        normalize (bool, optional): When True (default) L2-normalize each
            group before the dot product. False = raw correlation.

    Returns:
        torch.Tensor: The group-wise correlation cost tensor of shape (B, num_groups, H, W).
                      Each channel in the output corresponds to the correlation
                      score for one group.
    """
    B, C, H, W = feature1.shape

    # Assert that the number of channels is divisible by the number of groups.
    # The original code had this assertion commented out, but it's crucial for correct reshaping.
    assert C % num_groups == 0, f"Number of channels C ({C}) must be divisible by num_groups ({num_groups})."

    channels_per_group = C // num_groups

    # Reshape features to introduce the group dimension: (B, num_groups, channels_per_group, H, W)
    feature1 = feature1.reshape(B, num_groups, channels_per_group, H, W)
    feature2 = feature2.reshape(B, num_groups, channels_per_group, H, W)

    # Use `torch.amp.autocast('cuda', enabled=False)` to ensure that normalization
    # and summation are performed in full precision (float32) even if Automatic Mixed Precision (AMP)
    # is enabled, which can be important for numerical stability of normalization.
    with torch.amp.autocast('cuda', enabled=False):
        # Normalize features within each group along their channel dimension (dim=2)
        # Then, perform element-wise multiplication and sum along the channel_per_group dimension.
        if normalize:
            cost = (F.normalize(feature1.float(), dim=2) * F.normalize(feature2.float(), dim=2)).sum(dim=2)
        else:
            cost = (feature1.float() * feature2.float()).sum(dim=2)

    # The output `cost` should have shape (B, num_groups, H, W).
    # The original code had this assertion commented out, but it's useful for verification.
    # assert cost.shape == (B, num_groups, H, W), f"Expected cost shape ({B}, {num_groups}, {H}, {W}), but got {cost.shape}"

    return cost


def build_gwc_volume(refimg_feature, targetimg_feature, maxdisp, num_groups, stride=1,
                     normalize: bool = True):
    """
    Builds a Group-Wise Correlation (GWC) volume for stereo matching.

    This function constructs a 3D volume where each slice along the disparity
    dimension represents the group-wise correlation between the reference feature map
    and a horizontally shifted version of the target feature map. This volume
    serves as a cost aggregation input for subsequent disparity estimation networks.

    Args:
        refimg_feature (torch.Tensor): Feature map from the reference (left) image,
                                       of shape (B, C, H, W).
        targetimg_feature (torch.Tensor): Feature map from the target (right) image,
                                        of shape (B, C, H, W).
        maxdisp (int): The maximum disparity value to consider for the volume.
                       The volume will have `maxdisp // stride` disparity levels.
        num_groups (int): The number of groups to divide the channels into for
                          group-wise correlation. `C` must be divisible by `num_groups`.
        stride (int, optional): The stride for sampling disparity levels. For example,
                                if `stride=2`, only disparities 0, 2, 4, ... are considered.
                                Defaults to 1.

    Returns:
        torch.Tensor: The group-wise correlation volume of shape
                      (B, num_groups, maxdisp // stride, H, W).
                      `volume[b, g, d_idx, h, w]` contains the group-wise correlation
                      score for group `g` at spatial location (h, w) for a disparity
                      corresponding to `d_idx * stride`.

    Raises:
        AssertionError: If `maxdisp` is not divisible by `stride`.
    """
    B, _, H, W = refimg_feature.shape

    # Assert that maxdisp is perfectly divisible by the stride.
    # This ensures that disparity levels are sampled consistently.
    assert maxdisp % stride == 0, "maxdisp must be divisible by stride."

    # Calculate the number of disparity levels in the volume based on maxdisp and stride.
    num_disparity_levels = maxdisp // stride

    # Initialize an empty volume to store the group-wise correlation costs.
    # The channels correspond to `num_groups`, and there's a dimension for disparity levels.
    volume = refimg_feature.new_zeros([B, num_groups, num_disparity_levels, H, W])

    # Iterate through each disparity index for the volume
    for i in range(num_disparity_levels):
        # Calculate the actual disparity value for the current index
        current_disp_value = i * stride

        if current_disp_value > 0:
            # For non-zero disparities, calculate group-wise correlation between
            # the reference features and the target features shifted by `current_disp_value`.
            # `current_disp_value:` for ref_img_feature means starting from column `current_disp_value`.
            # `:-current_disp_value` for target_img_feature means ending before the last `current_disp_value` columns.
            volume[:, :, i, :, current_disp_value:] = groupwise_correlation(
                refimg_feature[:, :, :, current_disp_value:], targetimg_feature[:, :, :, :-current_disp_value],
                num_groups, normalize=normalize)
        else:
            # For disparity 0, calculate group-wise correlation without any shift.
            volume[:, :, i, :, :] = groupwise_correlation(refimg_feature, targetimg_feature, num_groups,
                                                          normalize=normalize)

    return volume.contiguous()  # Ensure the tensor is contiguous in memory


def norm_correlation(feature1, feature2):
    """
    Computes normalized correlation between two feature maps.

    This function calculates the cosine similarity between corresponding feature
    vectors in `feature1` and `feature2`. It first normalizes each feature vector
    by its L2-norm (Euclidean norm) to unit length before computing their
    element-wise product and summing along the channel dimension. This makes
    the similarity measure robust to variations in feature magnitude.

    Args:
        feature1 (torch.Tensor): The first feature map (e.g., from the reference image),
                                 of shape (B, C, H, W).
        feature2 (torch.Tensor): The second feature map (e.g., from the target image),
                                 of shape (B, C, H, W).

    Returns:
        torch.Tensor: The normalized correlation cost tensor of shape (B, 1, H, W).
                      Each element represents the similarity score at a given
                      spatial location, typically in the range [-1, 1].
    """
    # Calculate the L2-norm for each feature vector along the channel dimension (dim=1).
    # `keepdim=True` ensures the output shape is (B, 1, H, W) for broadcasting.
    # A small epsilon (1e-5) is added for numerical stability to prevent division by zero.
    norm1 = torch.norm(feature1, 2, 1, True) + 1e-5
    norm2 = torch.norm(feature2, 2, 1, True) + 1e-5

    # Normalize feature maps
    normalized_feature1 = feature1 / norm1
    normalized_feature2 = feature2 / norm2

    # Compute element-wise product of normalized features and then take the mean
    # along the channel dimension (dim=1). The mean operation here implicitly performs
    # the sum and then divides by the number of channels, effectively averaging the
    # cosine similarities across channels if C > 1. If C=1, it's just the product.
    cost = torch.mean((normalized_feature1 * normalized_feature2), dim=1, keepdim=True)
    return cost


def process_edgenext_state_dict(checkpoint_dict: OrderedDict) -> OrderedDict:
    """
    Processes a state dictionary from an EdgeNeXt model checkpoint to remove a leading 'model.' prefix.

    This function is useful when a pre-trained model's state dictionary has an extra
    prefix, such as 'model.', in its keys. This can happen during saving if the
    model was wrapped inside a container or a larger module. This function iterates
    through all keys in the `checkpoint_dict` and removes this prefix, making the
    state dictionary compatible with a direct instantiation of the `EdgeNeXt` model class.

    Args:
        checkpoint_dict (OrderedDict): The state dictionary loaded from an EdgeNeXt
                                       model checkpoint.

    Returns:
        OrderedDict: The processed state dictionary with the 'model.' prefix removed
                     from the keys.
    """
    new_state_dict = OrderedDict()
    for k, v in checkpoint_dict.items():
        parts = k.split('.')
        # Check if the first part of the key is 'model'.
        if 'model' == parts[0]:
            # If so, reconstruct the key by joining the remaining parts.
            new_key = '.'.join(parts[1:])
            new_state_dict[new_key] = v
        else:
            # Otherwise, keep the key as is.
            new_state_dict[k] = v

    # The original variable name is reassigned to the new dictionary.
    checkpoint_dict = new_state_dict
    return checkpoint_dict


def get_dataset_index(file_path, dataset_sub_config):
    """
    gets dataset index for a dataset listed in the dataset spec config.

    Args:
        file_path (str): The absolute path of the file.

    Returns:
        index (int): The index of the dataset file.
    """
    assert "data_sources" in dataset_sub_config, "dataset_subconfig_should contain 'data_sources' field'"
    sub_dirs_in_file_path = {x.lower(): 0 for x in file_path.split("/")}
    index = 0
    for i in range(len(dataset_sub_config["data_sources"])):
        if dataset_sub_config["data_sources"][i]["dataset_name"].lower() in sub_dirs_in_file_path:
            index = i
            break
    return index
