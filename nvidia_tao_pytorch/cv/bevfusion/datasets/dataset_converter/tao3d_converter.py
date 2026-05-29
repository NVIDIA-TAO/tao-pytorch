# SPDX-FileCopyrightText: Copyright (c) 2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""BEVFusion TAO3Ddataset converter functions."""

import os
import glob
from pathlib import Path
import numpy as np
import math
from skimage import io

import mmengine

import nvidia_tao_pytorch.cv.bevfusion.structures as tao_structures


def get_label_anno(label_path, is_synthetic=False, dimension_order='hwl'):
    """get tao3d label annotation """
    is_empty = False
    annotations = {}
    annotations.update({
        'name': [],
        'truncated': [],
        'occluded': [],
        'alpha': [],
        'bbox': [],
        'dimensions': [],
        'location': [],
        'rotation': []
    })
    with open(label_path, 'r') as f:
        lines = f.readlines()
    content = [line.strip().split(' ') for line in lines]
    if len(content) == 0:
        is_empty = True
    num_objects = len([x[0] for x in content if x[0] != 'DontCare'])
    # hack - assign person class
    annotations['name'] = np.array([x[0] for x in content])
    # annotations['name'] = np.array(['person' for x in content])
    num_gt = len(annotations['name'])
    annotations['truncated'] = np.array([float(x[1]) for x in content])
    annotations['occluded'] = np.array([float(x[2]) for x in content])
    annotations['alpha'] = np.array([float(x[3]) for x in content])
    bbox_temp = np.array([[float(info) for info in x[4:8]] for x in content]).reshape(-1, 4)
    bbox = np.zeros(bbox_temp.shape)
    if not is_synthetic:
        annotations['bbox'] = np.array([[float(info) for info in x[4:8]] for x in content]).reshape(-1, 4)
        if dimension_order == 'hwl':
            annotations['dimensions'] = np.array([[float(info) for info in x[8:11]]
                                                  for x in content]).reshape(-1, 3)[:, [2, 0, 1]]  # kitti-format hwl format to standard lhw(camera)
        else:
            annotations['dimensions'] = np.array([[float(info) for info in x[8:11]]
                                                  for x in content]).reshape(-1, 3)
    else:
        bbox[:, 0] = bbox_temp[:, 1]  # xmin
        bbox[:, 1] = bbox_temp[:, 3]  # ymin
        bbox[:, 2] = bbox_temp[:, 0]  # xmax
        bbox[:, 3] = bbox_temp[:, 2]  # ymax
        annotations['bbox'] = bbox
        annotations['dimensions'] = np.array([[float(info) for info in x[8:11]]
                                              for x in content]).reshape(-1, 3)
    annotations['location'] = np.array([[float(info) for info in x[11:14]]
                                        for x in content]).reshape(-1, 3)
    if not is_synthetic:
        if len(content[0]) == 18:
            annotations['rotation'] = np.array([[float(info) for info in x[14:17]]
                                                for x in content]).reshape(-1, 3)
        else:
            # zero-pad rotation
            annotations['rotation'] = np.array([[0.0, float(x[14]), 0.0]
                                                for x in content]).reshape(-1, 3)
    else:
        if len(content[0]) == 18:  # TAO3DSynthetic Data
            annotations['rotation'] = np.array([[math.radians(float(info)) for info in x[14:17]]
                                                for x in content]).reshape(-1, 3)
        else:
            raise NotImplementedError('Convert is not supported for this format. You need to provide three rotation angles in degrees for is_synthetic=True')

    if len(content) != 0 and len(content[0]) == 18:  # have score
        annotations['score'] = np.array([float(x[17]) for x in content])
    else:
        annotations['score'] = np.zeros((annotations['bbox'].shape[0], ))
    index = list(range(num_objects)) + [-1] * (num_gt - num_objects)
    annotations['index'] = np.array(index, dtype=np.int32)
    annotations['group_ids'] = np.arange(num_gt, dtype=np.int32)

    return annotations, is_empty


def add_difficulty_to_annos(info):
    """add difficituly to annotation. For TAO3D, it is all set to default -1"""
    # noqa pylint: disable=W0612
    annos = info['annos']
    dims = annos['dimensions']
    diff = []
    for i in range(len(dims)):
        diff.append(-1)

    annos['difficulty'] = np.array(diff, np.int32)


def get_tao3d_info_path(prefix,
                        file_name,
                        mode='training',
                        info_type='images',
                        seq_name='Warehouse_Normal_1',
                        relative_path=True,
                        exist_check=True):
    """get TAO3D info path """
    # noqa pylint: disable=R1705
    prefix = Path(prefix)
    file_path = Path(mode) / info_type / seq_name / file_name

    if exist_check and not (prefix / file_path).exists():
        raise ValueError('file not exist: {}'.format((prefix / file_path)))

    if relative_path:
        return str(file_path)

    return str(prefix / file_path)


def generate_per_sequence_pkl(seq_name,
                              input_root,
                              mode='training',
                              label_info=True,
                              lidar=False,
                              calib=False,
                              lidar_ext='.npy',
                              is_synthetic=True,
                              dimension_order='hwl',
                              relative_path=True,
                              with_imageshape=True):
    # noqa pylint: disable=W0612
    image_infos = []
    annotations = None
    if mode == 'validation':
        mode = 'training'

    image_list = glob.glob(os.path.join(input_root, mode, 'images', seq_name, '*.png'))

    for _, image_name in enumerate(image_list):
        info = {}
        pc_info = {'num_features': 4}
        image_info = {}
        calib_info = {}
        base_name = os.path.splitext(os.path.basename(image_name))[0]

        if label_info:
            label_path = get_tao3d_info_path(input_root, base_name + '.txt', mode=mode, info_type='labels', seq_name=seq_name, relative_path=relative_path)
            if relative_path:
                label_path = os.path.join(input_root, label_path)
            annotations, is_empty = get_label_anno(label_path, is_synthetic, dimension_order)
            if is_empty and mode == 'training':
                print('skipping this image due to empty GT : ' + image_name)
                continue
        if lidar:
            pc_info['lidar_path'] = get_tao3d_info_path(input_root, base_name + lidar_ext, mode=mode, info_type='lidar', seq_name=seq_name, relative_path=relative_path)

        image_info['image_path'] = get_tao3d_info_path(input_root, base_name + '.png', mode=mode, info_type='images', seq_name=seq_name, relative_path=relative_path)
        if with_imageshape:
            img_path = image_info['image_path']
            if relative_path:
                img_path = os.path.join(input_root, img_path)
            image_info['image_shape'] = np.array(io.imread(img_path).shape[:2], dtype=np.int32)

        if calib:
            calib_path = get_tao3d_info_path(input_root, base_name + '.txt', mode=mode, info_type='calib', seq_name=seq_name)
            if relative_path:
                calib_path = os.path.join(input_root, calib_path)
            with open(calib_path, 'r') as f:
                lines = f.readlines()
            lidar2cam = np.array([float(info) for info in lines[0].split(' ')[1:17]]).reshape([4, 4])

            yz_flip_array = np.eye(4).astype(np.float32)

            if is_synthetic:  # For Synthetic data only
                intrinsic_info = lines[4].split(' ')
                yz_flip_array[1, 1] = -1
                yz_flip_array[2, 2] = -1
            else:
                lidar2cam = np.transpose(lidar2cam)  # transpose for issac-ros
                intrinsic_info = lines[1].split(' ')
            cam2img = np.eye(4).astype(np.float32)
            cam2img[:3, :3] = np.array([[intrinsic_info[1], 0, intrinsic_info[3]], [0, intrinsic_info[2], intrinsic_info[4]], [0, 0, 1]])  # (R-H)

            cam2img = cam2img @ yz_flip_array
            cam2img = np.transpose(cam2img)

            calib_info['lidar2cam'] = lidar2cam
            calib_info['cam2img'] = cam2img

        if annotations is not None:
            info['annos'] = annotations
            add_difficulty_to_annos(info)

        info['image'] = image_info
        info['point_cloud'] = pc_info
        info['calib'] = calib_info
        image_infos.append(info)

    return image_infos


def remove_outside_points(points, projection_matrix, image_shape):
    """Remove points which are outside of image.

    Note:
        This function is for TAO3DSynthetic only.

    Args:
        points (np.ndarray, shape=[N, 3+dims]): Total points.
        projection_matrix (np.ndarray, shape=[4, 4]): Matrix to project points in
            specific camera coordinate CAM0. Left-hand
        image_shape (list[int]): Shape of image.

    Returns:
        np.ndarray, shape=[N, 3+dims]: Filtered points.
    """
    projected_points = tao_structures.project_cam2img(points[:, :3], projection_matrix, with_depth=False)

    xmin = 0
    ymin = 0
    xmax = image_shape[1]
    ymax = image_shape[0]
    fov_inds = ((projected_points[:, 0] < xmax) & (projected_points[:, 0] >= xmin) & (projected_points[:, 1] < ymax) & (projected_points[:, 1] >= ymin))
    reduced_points = points[fov_inds, :]
    if reduced_points.shape == 0:
        raise ValueError('no lidar points exist on image')

    return reduced_points


def _create_reduced_point_cloud(data_path,
                                info_path,
                                save_path=None,
                                num_features=4,
                                front_camera_id=2):
    """Create reduced point clouds for given info.

    Args:
        data_path (str): Path of original data.
        info_path (str): Path of data info.
        save_path (str, optional): Path to save reduced point cloud
            data. Default: None.
        back (bool, optional): Whether to flip the points to back.
            Default: False.
        num_features (int, optional): Number of point features. Default: 4.
        front_camera_id (int, optional): The referenced/front camera ID.
            Default: 2.
    """
    tao3d_infos = mmengine.load(info_path)

    for info in mmengine.track_iter_progress(tao3d_infos):
        pc_info = info['point_cloud']
        image_info = info['image']
        calib = info['calib']

        v_path = pc_info['lidar_path']
        v_path = Path(data_path) / v_path

        points_v = np.load(str(v_path)).reshape([-1, num_features])
        # original calib  - multiply vector from left side
        cam2img = calib['cam2img'].astype(np.float32)
        lidar2cam = calib['lidar2cam'].astype(np.float32)
        lidar2img = lidar2cam @ cam2img

        points_v = remove_outside_points(points_v, lidar2img, image_info['image_shape'])
        if save_path is None:
            save_dir = v_path.parent.parent.parent / (v_path.parent.parent.stem + '_reduced') / v_path.parent.stem
            if not save_dir.exists():
                save_dir.mkdir(parents=True, exist_ok=True)
            save_filename = save_dir / v_path.name
        else:
            save_dir = Path(save_path) / (v_path.parent.parent.stem + '_reduced')
            if not save_dir.exists():
                save_dir.mkdir(parents=True, exist_ok=True)
            save_filename = str(save_dir / v_path.name)

        np.save(save_filename, points_v, allow_pickle=False)


def create_reduced_point_cloud_tao3d(data_path,
                                     pkl_prefix='tao3d',
                                     info_path=None,
                                     mode='train',
                                     save_path=None):
    """Create reduced point clouds for training/validation/testing.

    Args:
        data_path (str): Path of original data.
        pkl_prefix (str): Prefix of info files.
        info_path (str, optional): Path of training set info.
            Default: None.
        save_path (str, optional): Path to save reduced point cloud data.
            Default: None.
        with_back (bool, optional): Whether to flip the points to back.
            Default: False.
    """
    if info_path is None:
        info_path = os.path.join(data_path, f'{pkl_prefix}_{mode}.pkl')
    print('create reduced point cloud')
    if mode == 'validation':
        reduced_output_dir = Path(os.path.join(save_path, 'training'))
    else:
        reduced_output_dir = Path(os.path.join(save_path, mode))
    if not reduced_output_dir.exists():
        reduced_output_dir.mkdir()
    _create_reduced_point_cloud(data_path, info_path, reduced_output_dir)


def merge_pkls(seq_list, prefix, pkl_root, output_pkl):
    """merge pkl from the sequence list """
    # check output pkl exist
    if os.path.exists(output_pkl):
        print(' Warning! output file exist - appending data to existing pkl file ' + output_pkl)
        output_infos = mmengine.load(output_pkl)
        idx_offset = len(output_infos['data_list'])
    else:
        print('creating new pkl in merge')
        output_infos = {'metainfo': {}, 'data_list': []}
        idx_offset = 0

    with open(seq_list) as f:
        lines = f.read().splitlines()
        for seq_name in lines:
            print('Merging ' + seq_name)
            input_pkl_seq = os.path.join(pkl_root, '{}_{}.pkl'.format(prefix, seq_name))
            infos = mmengine.load(input_pkl_seq)
            data_list = infos['data_list']
            for data in data_list:
                data['sample_idx'] = idx_offset + data['sample_idx']
                output_infos['data_list'].append(data)
            idx_offset = len(data_list)
            meta_info = infos['metainfo']
            output_infos['metainfo'] = meta_info
    mmengine.dump(output_infos, output_pkl, 'pkl')
