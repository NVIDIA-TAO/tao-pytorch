# SPDX-FileCopyrightText: Copyright (c) 2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Tool for creating ZIP/PNG based datasets."""

import functools
import io
import json
import os
import re
import zipfile
from pathlib import Path
from typing import Callable, Optional, Tuple, Union
import imageio

import numpy as np
import PIL.Image
from tqdm import tqdm

from nvidia_tao_pytorch.core.tlt_logging import logging
import nvidia_tao_pytorch.core.loggers.api_logging as status_logging


def error(msg):
    """Print an error message and exit the program.

    Args:
        msg (str): The error message to display.
    """
    logging.error(msg)
    status_logging.get_status_logger().write(
        message=msg,
        status_level=status_logging.Status.FAILURE
    )
    raise Exception(msg)


def parse_tuple(s: str) -> Tuple[int, int]:
    """Parse a string formatted as 'M,N' or 'MxN' into a tuple of integers.

    Args:
        s (str): String representing two integers separated by 'x' or ','.

    Returns:
        Tuple[int, int]: A tuple containing two integers.

    Raises:
        ValueError: If the input string cannot be parsed.

    Example:
        '4x2' returns (4, 2)
        '0,1' returns (0, 1)
    """
    m = re.match(r'^(\d+)[x,](\d+)$', s)
    if m:
        return (int(m.group(1)), int(m.group(2)))
    raise ValueError(f'cannot parse tuple {s}')


def maybe_min(a: int, b: Optional[int]) -> int:
    """Return the smaller of two values, if the second value is provided.

    Args:
        a (int): First value.
        b (Optional[int]): Second value, which may be None.

    Returns:
        int: The smaller of a and b, or a if b is None.
    """
    if b is not None:
        return min(a, b)
    return a


def file_ext(name: Union[str, Path]) -> str:
    """Extract the file extension from a filename or path.

    Args:
        name (Union[str, Path]): The filename or path.

    Returns:
        str: The file extension.
    """
    return str(name).rsplit('.', maxsplit=1)[-1]


def is_image_ext(fname: Union[str, Path]) -> bool:
    """Check if a file is an image based on its extension.

    Args:
        fname (Union[str, Path]): The filename or path.

    Returns:
        bool: True if the file extension matches an image format, False otherwise.
    """
    ext = file_ext(fname).lower()
    return f'.{ext}' in PIL.Image.EXTENSION  # type: ignore


def png_to_rgb(arr):
    """Convert a PNG image array with transparency to an RGB format with a white background.

    Args:
        arr (np.ndarray): Image array in PNG format.

    Returns:
        np.ndarray: RGB image array.
    """
    png = PIL.Image.fromarray(arr)
    png.load()  # required for png.split()

    background = PIL.Image.new("RGB", png.size, (255, 255, 255))
    background.paste(png, mask=png.split()[3])  # 3 is the alpha channel

    return np.array(background)


def open_image_folder(source_dir, *, max_images: Optional[int] = None):
    """Open an image folder and load images along with their labels.

    Args:
        source_dir (str): Path to the source directory.
        max_images (Optional[int], optional): Maximum number of images to load. Defaults to None.

    Returns:
        Tuple[int, Callable[[], dict]]: The maximum number of images and an iterator over image dictionaries.
    """
    input_images = [str(f) for f in sorted(Path(source_dir).rglob('*')) if is_image_ext(f) and os.path.isfile(f)]

    # Load labels from dataset.json if it exists; otherwise, create default labels based on sub-folder names.
    labels = {}
    meta_fname = os.path.join(source_dir, 'dataset.json')
    if os.path.isfile(meta_fname):
        with open(meta_fname, 'r') as file:
            labels = json.load(file).get('labels', {})
            if labels:
                labels = {x[0]: x[1] for x in labels}
            else:
                labels = {}
    else:
        # Default label is the sub-folder name for each image
        labels = {str(Path(f).relative_to(source_dir)).replace('\\', '/'): Path(f).parent.name for f in input_images}

    max_idx = maybe_min(len(input_images), max_images)

    def iterate_images():
        for idx, fname in enumerate(input_images):
            arch_fname = os.path.relpath(fname, source_dir).replace('\\', '/')
            img = imageio.imread(fname)

            # Alpha channel conversion
            if img.shape[-1] == 4:
                img = png_to_rgb(img)

            # Get label from the loaded labels or default to sub-folder name
            label = labels.get(arch_fname, Path(fname).parent.name)
            yield dict(img=img, label=label)

            if idx >= max_idx - 1:
                break

    return max_idx, iterate_images()


def make_transform(
    transform: Optional[str],
    output_width: Optional[int],
    output_height: Optional[int]
) -> Callable[[np.ndarray], Optional[np.ndarray]]:
    """Create a transformation function for resizing and cropping images.

    Args:
        transform (Optional[str]): Type of transformation ('center-crop' or None).
        output_width (Optional[int]): Desired output width.
        output_height (Optional[int]): Desired output height.

    Returns:
        Callable[[np.ndarray], Optional[np.ndarray]]: The transformation function.
    """
    def scale(width, height, img):
        w = img.shape[1]
        h = img.shape[0]
        if width == w and height == h:
            return img
        img = PIL.Image.fromarray(img)
        ww = width if width is not None else w
        hh = height if height is not None else h
        img = img.resize((ww, hh), PIL.Image.LANCZOS)
        return np.array(img)

    def center_crop(width, height, img):
        crop = np.min(img.shape[:2])
        img = img[(img.shape[0] - crop) // 2: (img.shape[0] + crop) // 2, (img.shape[1] - crop) // 2: (img.shape[1] + crop) // 2]
        img = PIL.Image.fromarray(img, 'RGB')
        img = img.resize((width, height), PIL.Image.LANCZOS)
        return np.array(img)

    if transform is None:
        return functools.partial(scale, output_width, output_height)
    if transform == 'center-crop':
        if (output_width is None) or (output_height is None):
            error('must specify --resolution=WxH when using ' + transform + 'transform')
        return functools.partial(center_crop, output_width, output_height)

    assert False, 'unknown transform'


def open_dataset(source, max_images: Optional[int]):
    """Open a dataset from a directory and return an image iterator.

    Args:
        source (str): Path to the source directory.
        max_images (Optional[int]): Maximum number of images to load.

    Returns:
        Tuple[int, Callable[[], dict]]: The maximum number of images and an iterator over image dictionaries.
    """
    if os.path.isdir(source):
        return open_image_folder(source, max_images=max_images)
    else:
        return error(f'Missing input directory: {source}')


def open_dest(dest: str) -> Tuple[str, Callable[[str, Union[bytes, str]], None], Callable[[], None]]:
    """Open a destination for writing data, supporting zip and directory output.

    Args:
        dest (str): Destination path.

    Returns:
        Tuple[str, Callable[[str, Union[bytes, str]], None], Callable[[], None]]:
            The destination path, a write function, and a close function.
    """
    dest_ext = file_ext(dest)

    if dest_ext == 'zip':
        if os.path.dirname(dest) != '':
            os.makedirs(os.path.dirname(dest), exist_ok=True)
        zf = zipfile.ZipFile(file=dest, mode='w', compression=zipfile.ZIP_STORED)

        def zip_write_bytes(fname: str, data: Union[bytes, str]):
            zf.writestr(fname, data)
        return '', zip_write_bytes, zf.close
    else:
        # If the output folder already exists, check that is is
        # empty.
        #
        # Note: creating the output directory is not strictly
        # necessary as folder_write_bytes() also mkdirs, but it's better
        # to give an error message earlier in case the dest folder
        # somehow cannot be created.
        if os.path.isdir(dest) and len(os.listdir(dest)) != 0:
            error('--dest folder must be empty')
        os.makedirs(dest, exist_ok=True)

        def folder_write_bytes(fname: str, data: Union[bytes, str]):
            os.makedirs(os.path.dirname(fname), exist_ok=True)
            with open(fname, 'wb') as fout:
                if isinstance(data, str):
                    data = data.encode('utf8')
                fout.write(data)
        return dest, folder_write_bytes, lambda: None


def convert_dataset(
    source: str,
    dest: str,
    max_images: Optional[int],
    transform: Optional[str],
    resolution: Optional[Tuple[int, int]]
):
    """Convert an image dataset into a dataset archive usable with StyleGAN2 ADA PyTorch.

    The input dataset format is guessed from the --source argument:

    \b
    --source path/                      Recursively load all images from path/

    Specifying the output format and path:

    \b
    --dest /path/to/dir                 Save output files under /path/to/dir
    --dest /path/to/dataset.zip         Save output files into /path/to/dataset.zip

    The output dataset format can be either an image folder or an uncompressed zip archive.
    Zip archives makes it easier to move datasets around file servers and clusters, and may
    offer better training performance on network file systems.

    Images within the dataset archive will be stored as uncompressed PNG.
    Uncompresed PNGs can be efficiently decoded in the training loop.

    Class labels are stored in a file called 'dataset.json' that is stored at the
    dataset root folder.  This file has the following structure:

    \b
    {
        "labels": [
            ["00000/img00000000.png",6],
            ["00000/img00000001.png",9],
            ... repeated for every image in the datase
            ["00049/img00049999.png",1]
        ]
    }

    Mapping dictionay which maps string class to intger class is stored in a file called 'label_map.json' that is stored at the
    dataset root folder.  This file has the following structure:

    \b
    {
        "BIRD": 0,
        "DOG": 1,
        "CAT": 2,
        ...
    }

    If the 'dataset.json' file cannot be found, the dataset is interpreted as
    not containing class labels.

    Image scale/crop and resolution requirements:

    Output images must be square-shaped and they must all have the same power-of-two
    dimensions.

    To scale arbitrary input image size to a specific width and height, use the
    --resolution option.  Output resolution will be either the original
    input resolution (if resolution was not specified) or the one specified with
    --resolution option.

    Use the --transform=center-crop option to apply a
    center crop transform on the input image.  These options should be used with the
    --resolution option.  For example:

    \b
    python dataset_tool.py --source LSUN/raw/cat_lmdb --dest /tmp/lsun_cat \\
        --transform=center-crop --resolution=512x384
    """
    PIL.Image.init()  # type: ignore

    if dest == '':
        error('--dest output filename or directory must not be an empty string')

    num_files, input_iter = open_dataset(source, max_images=max_images)
    archive_root_dir, save_bytes, close_dest = open_dest(dest)

    if resolution is None:
        resolution = (None, None)
    transform_image = make_transform(transform, *resolution)

    dataset_attrs = None

    labels = []
    for idx, image in tqdm(enumerate(input_iter), total=num_files):
        idx_str = f'{idx:08d}'
        archive_fname = f'{idx_str[:5]}/img{idx_str}.png'

        # Apply crop and resize.
        img = transform_image(image['img'])

        # Transform may drop images.
        if img is None:
            assert False, 'None of image after transform'

        # Error check to require uniform image attributes across
        # the whole dataset.
        channels = img.shape[2] if img.ndim == 3 else 1
        cur_image_attrs = {
            'width': img.shape[1],
            'height': img.shape[0],
            'channels': channels
        }
        if dataset_attrs is None:
            dataset_attrs = cur_image_attrs
            width = dataset_attrs['width']
            height = dataset_attrs['height']
            if width != height:
                error(f'Image dimensions after scale and crop are required to be square.  Got {width}x{height}')
            if dataset_attrs['channels'] not in [1, 3]:
                error('Input images must be stored as RGB or grayscale')
            if width != 2 ** int(np.floor(np.log2(width))):
                error('Image width/height after scale and crop are required to be power-of-two')
        elif dataset_attrs != cur_image_attrs:
            err = [f'  dataset {k}/cur image {k}: {dataset_attrs[k]}/{cur_image_attrs[k]}' for k in dataset_attrs.keys()]  # pylint: disable=unsubscriptable-object
            error(f'Image {archive_fname} attributes must be equal across all images of the dataset.  Got:\n' + '\n'.join(err))

        # Save the image as an uncompressed PNG.
        img = PIL.Image.fromarray(img, {1: 'L', 3: 'RGB'}[channels])
        image_bits = io.BytesIO()
        img.save(image_bits, format='png', compress_level=0, optimize=False)
        save_bytes(os.path.join(archive_root_dir, archive_fname), image_bits.getbuffer())
        labels.append([archive_fname, image['label']] if image['label'] is not None else None)

    # Step 1: Create a mapping of unique string labels to integers
    label_map = {label: idx for idx, label in enumerate(sorted({label for _, label in labels}))}
    # Step 2: Convert the original labels to integer labels using the label_map
    int_labels = [[filename, label_map[label]] for filename, label in labels]

    # Step 3: Create the metadata with integer labels
    metadata = {
        'labels': int_labels if all(label is not None for _, label in int_labels) else None
    }

    # Step 4: Save the metadata and label map to JSON
    save_bytes(os.path.join(archive_root_dir, 'dataset.json'), json.dumps(metadata, indent=4))

    # Step 5: Export label_map as a JSON file for reference
    save_bytes(os.path.join(archive_root_dir, 'label_map.json'), json.dumps(label_map, indent=4))

    close_dest()


if __name__ == "__main__":
    convert_dataset()  # pylint: disable=no-value-for-parameter
