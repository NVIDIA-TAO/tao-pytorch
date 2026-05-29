# SPDX-FileCopyrightText: Copyright (c) 2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

""" BEVFusion data preparation functions """

from os import path as osp
import mmengine

from .update_infos_to_v2 import update_pkl_infos
from .kitti_converter import create_kitti_info_file, create_reduced_point_cloud
from .tao3d_converter import generate_per_sequence_pkl, create_reduced_point_cloud_tao3d, merge_pkls
from nvidia_tao_pytorch.core.utilities import check_and_create


def kitti_data_prep(root_path,
                    mode,
                    info_prefix,
                    out_dir,
                    with_plane=False):
    """Prepare data related to Kitti dataset.

    Related data consists of '.pkl' files recording basic infos,
    2D annotations and groundtruth database.

    Args:
        root_path (str): Path of dataset root.
        info_prefix (str): The prefix of info filenames.
        out_dir (str): Output directory of the groundtruth database info.
        with_plane (bool, optional): Whether to use plane information.
            Default: False.
    """
    create_kitti_info_file(root_path, info_prefix, with_plane, out_dir, mode=mode)
    create_reduced_point_cloud(root_path, info_prefix, mode=mode, save_path=out_dir)
    if mode == 'training':
        info_train_path = osp.join(out_dir, f'{info_prefix}_infos_train.pkl')
        info_val_path = osp.join(out_dir, f'{info_prefix}_infos_val.pkl')
        info_trainval_path = osp.join(out_dir, f'{info_prefix}_infos_trainval.pkl')
        update_pkl_infos('kitti', out_dir=out_dir, pkl_path=info_train_path)
        update_pkl_infos('kitti', out_dir=out_dir, pkl_path=info_val_path)
        update_pkl_infos('kitti', out_dir=out_dir, pkl_path=info_trainval_path)
    elif mode == 'validation':
        info_val_path = osp.join(out_dir, f'{info_prefix}_infos_val.pkl')
        update_pkl_infos('kitti', out_dir=out_dir, pkl_path=info_val_path)
    elif mode == 'testing':
        info_test_path = osp.join(out_dir, f'{info_prefix}_infos_test.pkl')
        update_pkl_infos('kitti', out_dir=out_dir, pkl_path=info_test_path)
    else:
        raise NotImplementedError(f'Don\'t support convert {mode}.')


def tao3d_data_prep(root_path,
                    mode,
                    seq_list,
                    is_synthetic,
                    dimension_order,
                    info_prefix,
                    out_dir,
                    merge_only=False):
    """Prepare data related to TAO3D dataset.

    Args:
        root_path (str): Path of dataset root.
        mode (str) : Select from training, validation or testing
        seq_list (str) : Path to the txt file containing sequence list to process
        info_prefix (str): The prefix of info filenames.
        out_dir (str): Output directory of the groundtruth database info.
        merge_only (str) : Defaults to False. Set it to True when you already have per sequence pkl file processed.
    """
    if not merge_only:
        check_and_create(osp.join(out_dir, 'pkls'))
        # For inference only, label doesn't need to be provided
        label_flag = True
        with open(seq_list) as f:
            lines = f.read().splitlines()
            for seq_name in lines:
                print('Processing ' + seq_name)
                if mode == 'inference':
                    label_flag = False
                image_infos = generate_per_sequence_pkl(seq_name, root_path, mode,
                                                        label_info=label_flag,
                                                        lidar=True,
                                                        calib=True,
                                                        lidar_ext='.npy',
                                                        is_synthetic=is_synthetic,
                                                        dimension_order=dimension_order)
                pkl_name = osp.join(out_dir, 'pkls', '{}_{}.pkl'.format(info_prefix, seq_name))
                print(f'Sequence {seq_name} info file is saved to {pkl_name}')
                mmengine.dump(image_infos, pkl_name, 'pkl')
                create_reduced_point_cloud_tao3d(root_path, info_path=pkl_name, save_path=out_dir)
                update_pkl_infos('tao3d', out_dir=osp.join(out_dir, 'pkls'), pkl_path=pkl_name)
        output_pkl = osp.join(out_dir, '{}_{}.pkl'.format(info_prefix, mode))
        merge_pkls(seq_list, info_prefix, osp.join(out_dir, 'pkls'), output_pkl)
    else:
        output_pkl = osp.join(out_dir, '{}_{}.pkl'.format(info_prefix, mode))
        merge_pkls(seq_list, info_prefix, osp.join(out_dir, 'pkls'), output_pkl)
