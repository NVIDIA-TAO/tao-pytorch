# SPDX-FileCopyrightText: Copyright (c) 2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

""" BEVFusion kitti-person converter functions. """

from pathlib import Path
import numpy as np
import os

import mmengine
from mmdet3d.structures.ops import box_np_ops

from .kitti_data_utils import get_kitti_image_info

kitti_categories = ('Pedestrian', 'Cyclist', 'Car')


def _read_imageset_file(path):
    """ Read imageset file"""
    with open(path, 'r') as f:
        lines = f.readlines()
    return [int(line) for line in lines]


class _NumPointsInGTCalculater:
    """Calculate the number of points inside the ground truth box. This is the
    parallel version. For the serialized version, please refer to
    `_calculate_num_points_in_gt`.
    """

    def __init__(self,
                 data_path,
                 relative_path,
                 remove_outside=True,
                 num_features=4,
                 num_worker=8) -> None:
        """
        Args:
            data_path (str): Path of the data.
            relative_path (bool): Whether to use relative path.
            remove_outside (bool, optional): Whether to remove points which are
                outside of image. Default: True.
            num_features (int, optional): Number of features per point.
                Default: False.
            num_worker (int, optional): the number of parallel workers to use.
                Default: 8.
        """
        self.data_path = data_path
        self.relative_path = relative_path
        self.remove_outside = remove_outside
        self.num_features = num_features
        self.num_worker = num_worker

    def calculate_single(self, info):
        """ process single frame"""
        pc_info = info['point_cloud']
        image_info = info['image']
        calib = info['calib']
        if self.relative_path:
            v_path = str(Path(self.data_path) / pc_info['velodyne_path'])
        else:
            v_path = pc_info['velodyne_path']
        points_v = np.fromfile(
            v_path, dtype=np.float32,
            count=-1).reshape([-1, self.num_features])
        rect = calib['R0_rect']
        Trv2c = calib['Tr_velo_to_cam']
        P2 = calib['P2']
        if self.remove_outside:
            points_v = box_np_ops.remove_outside_points(
                points_v, rect, Trv2c, P2, image_info['image_shape'])
        annos = info['annos']
        num_obj = len([n for n in annos['name'] if n != 'DontCare'])
        dims = annos['dimensions'][:num_obj]
        loc = annos['location'][:num_obj]
        rots = annos['rotation_y'][:num_obj]
        gt_boxes_camera = np.concatenate([loc, dims, rots[..., np.newaxis]],
                                         axis=1)
        gt_boxes_lidar = box_np_ops.box_camera_to_lidar(
            gt_boxes_camera, rect, Trv2c)
        indices = box_np_ops.points_in_rbbox(points_v[:, :3], gt_boxes_lidar)
        num_points_in_gt = indices.sum(0)
        num_ignored = len(annos['dimensions']) - num_obj
        num_points_in_gt = np.concatenate(
            [num_points_in_gt, -np.ones([num_ignored])])
        annos['num_points_in_gt'] = num_points_in_gt.astype(np.int32)
        return info

    def calculate(self, infos):
        ret_infos = mmengine.track_parallel_progress(self.calculate_single,
                                                     infos, self.num_worker)
        for i, ret_info in enumerate(ret_infos):
            infos[i] = ret_info


def _calculate_num_points_in_gt(data_path,
                                infos,
                                relative_path,
                                remove_outside=True,
                                num_features=4):
    """ calculate number of points in ground truth """
    for info in mmengine.track_iter_progress(infos):
        pc_info = info['point_cloud']
        image_info = info['image']
        calib = info['calib']
        if relative_path:
            v_path = str(Path(data_path) / pc_info['velodyne_path'])
        else:
            v_path = pc_info['velodyne_path']
        points_v = np.fromfile(
            v_path, dtype=np.float32, count=-1).reshape([-1, num_features])
        rect = calib['R0_rect']
        Trv2c = calib['Tr_velo_to_cam']
        P2 = calib['P2']
        if remove_outside:
            points_v = box_np_ops.remove_outside_points(
                points_v, rect, Trv2c, P2, image_info['image_shape'])

        # points_v = points_v[points_v[:, 0] > 0]
        annos = info['annos']
        num_obj = len([n for n in annos['name'] if n != 'DontCare'])
        # annos = kitti.filter_kitti_anno(annos, ['DontCare'])
        dims = annos['dimensions'][:num_obj]
        loc = annos['location'][:num_obj]
        rots = annos['rotation_y'][:num_obj]
        gt_boxes_camera = np.concatenate([loc, dims, rots[..., np.newaxis]],
                                         axis=1)
        gt_boxes_lidar = box_np_ops.box_camera_to_lidar(
            gt_boxes_camera, rect, Trv2c)
        indices = box_np_ops.points_in_rbbox(points_v[:, :3], gt_boxes_lidar)
        num_points_in_gt = indices.sum(0)
        num_ignored = len(annos['dimensions']) - num_obj
        num_points_in_gt = np.concatenate(
            [num_points_in_gt, -np.ones([num_ignored])])
        annos['num_points_in_gt'] = num_points_in_gt.astype(np.int32)


def create_kitti_info_file(data_path,
                           pkl_prefix='kitti',
                           with_plane=False,
                           save_path=None,
                           relative_path=True,
                           mode='training'):
    """Create info file of KITTI dataset.

    Given the raw data, generate its related info file in pkl format.

    Args:
        data_path (str): Path of the data root.
        pkl_prefix (str, optional): Prefix of the info file to be generated.
            Default: 'kitti'.
        with_plane (bool, optional): Whether to use plane information.
            Default: False.
        save_path (str, optional): Path to save the info file.
            Default: None.
        relative_path (bool, optional): Whether to use relative path.
            Default: True.
    """
    imageset_folder = Path(data_path) / 'ImageSets'
    train_img_ids = _read_imageset_file(str(imageset_folder / 'train.txt'))
    val_img_ids = _read_imageset_file(str(imageset_folder / 'val.txt'))
    test_img_ids = _read_imageset_file(str(imageset_folder / 'test.txt'))

    print('Generate info. this may take several minutes.')
    if save_path is None:
        save_path = Path(data_path)
    else:
        save_path = Path(save_path)
    if mode == 'training':
        kitti_infos_train = get_kitti_image_info(
            data_path,
            training=True,
            velodyne=True,
            calib=True,
            with_plane=with_plane,
            image_ids=train_img_ids,
            relative_path=relative_path)
        _calculate_num_points_in_gt(data_path, kitti_infos_train, relative_path)
        filename = save_path / f'{pkl_prefix}_infos_train.pkl'
        print(f'Kitti info train file is saved to {filename}')
        mmengine.dump(kitti_infos_train, filename)
        kitti_infos_val = get_kitti_image_info(
            data_path,
            training=True,
            velodyne=True,
            calib=True,
            with_plane=with_plane,
            image_ids=val_img_ids,
            relative_path=relative_path)
        _calculate_num_points_in_gt(data_path, kitti_infos_val, relative_path)
        filename = save_path / f'{pkl_prefix}_infos_val.pkl'
        print(f'Kitti info val file is saved to {filename}')
        mmengine.dump(kitti_infos_val, filename)
        filename = save_path / f'{pkl_prefix}_infos_trainval.pkl'
        print(f'Kitti info trainval file is saved to {filename}')
        mmengine.dump(kitti_infos_train + kitti_infos_val, filename)
    elif mode == 'validation':
        kitti_infos_val = get_kitti_image_info(
            data_path,
            training=True,
            velodyne=True,
            calib=True,
            with_plane=with_plane,
            image_ids=val_img_ids,
            relative_path=relative_path)
        _calculate_num_points_in_gt(data_path, kitti_infos_val, relative_path)
        filename = save_path / f'{pkl_prefix}_infos_val.pkl'
        print(f'Kitti info val file is saved to {filename}')
        mmengine.dump(kitti_infos_val, filename)
    elif mode == 'testing':
        kitti_infos_test = get_kitti_image_info(
            data_path,
            training=False,
            label_info=False,
            velodyne=True,
            calib=True,
            with_plane=False,
            image_ids=test_img_ids,
            relative_path=relative_path)
        filename = save_path / f'{pkl_prefix}_infos_test.pkl'
        print(f'Kitti info test file is saved to {filename}')
        mmengine.dump(kitti_infos_test, filename)
    else:
        raise NotImplementedError(f'Don\'t support convert {mode}.')


def _create_reduced_point_cloud(data_path,
                                info_path,
                                save_path=None,
                                back=False,
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
    kitti_infos = mmengine.load(info_path)

    for info in mmengine.track_iter_progress(kitti_infos):
        pc_info = info['point_cloud']
        image_info = info['image']
        calib = info['calib']

        v_path = pc_info['velodyne_path']
        v_path = Path(data_path) / v_path
        points_v = np.fromfile(
            str(v_path), dtype=np.float32,
            count=-1).reshape([-1, num_features])
        rect = calib['R0_rect']
        if front_camera_id == 2:
            P2 = calib['P2']
        else:
            P2 = calib[f'P{str(front_camera_id)}']
        Trv2c = calib['Tr_velo_to_cam']
        # first remove z < 0 points
        # keep = points_v[:, -1] > 0
        # points_v = points_v[keep]
        # then remove outside.
        if back:
            points_v[:, 0] = -points_v[:, 0]
        points_v = box_np_ops.remove_outside_points(points_v, rect, Trv2c, P2,
                                                    image_info['image_shape'])
        if save_path is None:
            save_dir = v_path.parent.parent / (v_path.parent.stem + '_reduced')
            if not save_dir.exists():
                save_dir.mkdir()
            save_filename = save_dir / v_path.name
            # save_filename = str(v_path) + '_reduced'
            if back:
                save_filename += '_back'
        else:
            save_dir = Path(save_path) / (v_path.parent.stem + '_reduced')
            if not save_dir.exists():
                save_dir.mkdir(parents=True, exist_ok=True)
            save_filename = str(save_dir / v_path.name)
            if back:
                save_filename += '_back'
        with open(save_filename, 'w') as f:
            points_v.tofile(f)


def create_reduced_point_cloud(data_path,
                               pkl_prefix,
                               train_info_path=None,
                               val_info_path=None,
                               test_info_path=None,
                               save_path=None,
                               with_back=False,
                               mode='training'):
    """Create reduced point clouds for training/validation/testing.

    Args:
        data_path (str): Path of original data.
        pkl_prefix (str): Prefix of info files.
        train_info_path (str, optional): Path of training set info.
            Default: None.
        val_info_path (str, optional): Path of validation set info.
            Default: None.
        test_info_path (str, optional): Path of test set info.
            Default: None.
        save_path (str, optional): Path to save reduced point cloud data.
            Default: None.
        with_back (bool, optional): Whether to flip the points to back.
            Default: False.
    """
    if train_info_path is None:
        train_info_path = Path(save_path) / f'{pkl_prefix}_infos_train.pkl'
    if val_info_path is None:
        val_info_path = Path(save_path) / f'{pkl_prefix}_infos_val.pkl'
    if test_info_path is None:
        test_info_path = Path(save_path) / f'{pkl_prefix}_infos_test.pkl'

    if mode == 'training':
        print('create reduced point cloud for training set')
        reduced_output_dir = Path(os.path.join(save_path, 'training'))
        if not reduced_output_dir.exists():
            reduced_output_dir.mkdir()
        _create_reduced_point_cloud(data_path, train_info_path, reduced_output_dir)
        print('create reduced point cloud for validation set')
        _create_reduced_point_cloud(data_path, val_info_path, reduced_output_dir)

        if with_back:
            _create_reduced_point_cloud(
                data_path, train_info_path, reduced_output_dir, back=True)
            _create_reduced_point_cloud(
                data_path, val_info_path, reduced_output_dir, back=True)

    elif mode == 'validation':
        print('create reduced point cloud for validation set')
        reduced_output_dir = Path(os.path.join(save_path, 'training'))
        if not reduced_output_dir.exists():
            reduced_output_dir.mkdir()
        _create_reduced_point_cloud(data_path, val_info_path, reduced_output_dir)
        if with_back:
            _create_reduced_point_cloud(
                data_path, val_info_path, reduced_output_dir, back=True)
    elif mode == 'testing':
        print('create reduced point cloud for testing set')
        reduced_output_dir = Path(os.path.join(save_path, 'testing'))
        if not reduced_output_dir.exists():
            reduced_output_dir.mkdir()
        _create_reduced_point_cloud(data_path, test_info_path, reduced_output_dir)
        if with_back:
            _create_reduced_point_cloud(
                data_path, test_info_path, reduced_output_dir, back=True)
    else:
        raise NotImplementedError(f'Don\'t support convert {mode}.')
