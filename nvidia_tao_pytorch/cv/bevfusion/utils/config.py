# SPDX-FileCopyrightText: Copyright (c) 2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""BEVFusion configuration utility Function"""

import os
import copy
from abc import abstractmethod
from omegaconf import OmegaConf

from mmengine.dist import get_dist_info, init_dist
from .misc import prepare_origin_per_dataset


class BEVFusionConfig(object):
    """BEVFusion Config Class to convert Hydra config to MMEngine config"""

    def __init__(self,
                 config,
                 phase='train'):
        """Init Function."""
        self.config = OmegaConf.to_container(config, resolve=True)
        self.input_modality = self.config['input_modality']

        self.updated_config = {}
        self.phase = phase
        self.update_config(phase)

    def update_config(self, phase):
        """ Function to update hydra config to mmdet3d based config"""
        self.update_env()
        self.update_dataset_config()
        self.update_model_config()
        if phase == 'train':
            self.update_train_params_config()
        elif phase in ('evaluate', 'inference'):
            origin, yaw_dim, is_synthetic = prepare_origin_per_dataset(self.config)
            self.updated_config['test_evaluator'] = {'type': 'TAO3DMetric', 'metric': 'bbox', 'gt_box_type': self.config['dataset']['gt_box_type'],
                                                     'origin': origin, 'yaw_dim': yaw_dim, 'is_synthetic': is_synthetic,
                                                     'pcd_limit_range': self.config['model']['point_cloud_range'],
                                                     'ann_file': self.config['dataset']['test_dataset']['ann_file'],
                                                     'default_cam_key': self.config['dataset']['default_cam_key'],
                                                     'backend_args': None}
            self.updated_config['infer_data_config'] = {'dataset_root': self.config['dataset']['root_dir'], 'show': self.config['inference']['show'],
                                                        'conf_threshold': self.config['inference']['conf_threshold'],
                                                        'default_cam_key': self.config['dataset']['default_cam_key'],
                                                        'img_file': self.config['dataset']['img_file'],
                                                        'pc_file': self.config['dataset']['pc_file'],
                                                        'cam2img': self.config['dataset']['cam2img'],
                                                        'lidar2cam': self.config['dataset']['lidar2cam'],
                                                        'per_sequence': self.config['dataset']['per_sequence']
                                                        }
        else:
            raise NotImplementedError(f'Phase {phase} is not supported yet')

    def update_env(self):
        """Function to update env variables"""
        if self.config['manual_seed'] is not None:
            self.updated_config['randomness'] = {'seed': self.config['manual_seed'], 'deterministic': True}
        self.updated_config['default_scope'] = self.config['default_scope']
        self.updated_config['default_hooks'] = self.config['default_hooks']
        self.updated_config['visualizer'] = {'type': 'TAO3DLocalVisualizer',
                                             'vis_backends': {'type': 'LocalVisBackend'}, 'name': 'visualizer'}
        self.updated_config['work_dir'] = self.config['results_dir']
        self.updated_config['env_cfg'] = {}
        # The env_cfg has params for the distributed environment
        if 'CUDA_VISIBLE_DEVICES' in os.environ.keys() and len(os.environ['CUDA_VISIBLE_DEVICES'].split(',')) > 1:
            self.updated_config['env_cfg'] = {'cudnn_benchmark': True,
                                              'mp_cfg': {'mp_start_method': 'fork', 'opencv_num_threads': 0},
                                              'dist_cfg': {'backend': 'nccl'}}
            self.updated_config['launcher'] = 'pytorch'
            rank, world_size = get_dist_info()
            # If distributed these env variables are set by torchrun
            if 'LOCAL_RANK' not in os.environ:
                os.environ['LOCAL_RANK'] = str(0)
            if "RANK" not in os.environ:
                os.environ['RANK'] = str(rank)
            if "WORLD_SIZE" not in os.environ:
                os.environ['WORLD_SIZE'] = str(world_size)
            if "MASTER_PORT" not in os.environ:
                os.environ['MASTER_PORT'] = str(631)
            if "MASTER_ADDR" not in os.environ:
                os.environ['MASTER_ADDR'] = "127.0.0.1"

            init_dist("pytorch", backend="nccl")

    @abstractmethod
    def update_dataset_config(self):
        """Update the dataset config"""
        dataset_config = self.config['dataset']
        origin, yaw_dim, is_synthetic = prepare_origin_per_dataset(self.config)

        # set up input pipeline
        if self.phase == 'train':
            train_dataloader = {}
            train_dataset_config = dataset_config['train_dataset']
            val_dataset_config = dataset_config['val_dataset']

            if not self.input_modality.get('use_lidar'):
                raise ValueError('input_modality - use_lidar must be set to True for this model')

            if not self.input_modality.get('use_camera'):
                # lidar-only
                train_pipeline = [
                    {
                        'type': 'TAOLoadPointsFromFile',
                        'coord_type': 'LIDAR',
                        'load_dim': self.config['dataset']['point_cloud_dim'],
                        'use_dim': self.config['dataset']['point_cloud_dim'],
                        'backend_args': None
                    },
                    {
                        'type': 'LoadAnnotations3D',
                        'with_bbox_3d': True,
                        'with_label_3d': True,
                        'with_attr_label': False
                    },
                    {'type': 'PointsRangeFilter', 'point_cloud_range': self.config['model']['point_cloud_range']},
                    {'type': 'MyObjectRangeFilter', 'point_cloud_range': self.config['model']['point_cloud_range']},
                    {'type': 'ObjectNameFilter', 'classes': [dataset_config['classes']]},
                    {'type': 'PointShuffle'},
                    {
                        'type': 'Pack3DDetInputs', 'keys': ['points', 'gt_bboxes_3d', 'gt_labels_3d', 'gt_bboxes', 'gt_labels'],
                        'meta_keys': ['cam2img', 'ori_cam2img', 'lidar2cam', 'lidar2img', 'cam2lidar',
                                      'ori_lidar2img', 'box_type_3d', 'sample_idx', 'lidar_path', 'transformation_3d_flow', 'pcd_rotation',
                                      'pcd_scale_factor', 'pcd_trans', 'lidar_aug_matrix', 'num_pts_feats', 'num_views']
                    }
                ]
            else:
                # image-lidar fusion
                train_pipeline = [
                    {
                        'type': 'BEVFusionLoadMultiViewImageFromFiles',
                        'set_default_scale': True,
                        'to_float32': True,
                        'num_views': self.config['dataset']['num_views'],
                        'color_type': 'color',
                        'backend_args': None
                    },
                    {
                        'type': 'TAOLoadPointsFromFile',
                        'coord_type': 'LIDAR',
                        'load_dim': self.config['dataset']['point_cloud_dim'],
                        'use_dim': self.config['dataset']['point_cloud_dim'],
                        'backend_args': None
                    },
                    {
                        'type': 'LoadAnnotations3D',
                        'with_bbox_3d': True,
                        'with_label_3d': True,
                        'with_attr_label': False},
                    {
                        'type': 'ImageAug3D',
                        'final_dim': [256, 704],
                        'resize_lim': [0.38, 0.55],
                        'bot_pct_lim': [0.0, 0.0],
                        'rot_lim': [-5.4, 5.4],
                        'rand_flip': True,
                        'is_train': True
                    },
                    {'type': 'PointsRangeFilter', 'point_cloud_range': self.config['model']['point_cloud_range']},
                    {'type': 'MyObjectRangeFilter', 'point_cloud_range': self.config['model']['point_cloud_range']},
                    {'type': 'ObjectNameFilter', 'classes': [dataset_config['classes']]},
                    {'type': 'PointShuffle'},
                    {
                        'type': 'Pack3DDetInputs', 'keys': ['img', 'points', 'gt_bboxes_3d', 'gt_labels_3d', 'gt_bboxes', 'gt_labels'],
                        'meta_keys': ['cam2img', 'ori_cam2img', 'lidar2cam', 'lidar2img', 'cam2lidar',
                                      'ori_lidar2img', 'img_aug_matrix', 'box_type_3d', 'sample_idx',
                                      'lidar_path', 'img_path', 'transformation_3d_flow', 'pcd_rotation',
                                      'pcd_scale_factor', 'pcd_trans',
                                      'lidar_aug_matrix', 'num_pts_feats', 'num_views']
                    }
                ]

            # training dataloader
            if train_dataset_config['ann_file'] is None or val_dataset_config['ann_file'] is None:
                raise ValueError('ann_file for train_dataset and val_dataset must be provided')

            train_dataloader['batch_size'] = train_dataset_config['batch_size']
            train_dataloader['num_workers'] = train_dataset_config['num_workers']
            train_dataloader['drop_last'] = True
            train_dataloader['persistent_workers'] = True
            train_dataloader['sampler'] = {'type': train_dataset_config['sampler'], 'shuffle': True}

            if train_dataset_config['repeat_time'] is not None and train_dataset_config['repeat_time'] > 1:
                train_dataloader['dataset'] = {'type': 'RepeatDataset', 'times': train_dataset_config['repeat_time'],
                                               'dataset':
                                               {'type': dataset_config['type'],
                                                'ann_file': train_dataset_config['ann_file'], 'pipeline': train_pipeline,
                                                'metainfo': {'classes': dataset_config['classes']},
                                                'modality': self.config['input_modality'], 'test_mode': False,
                                                'data_prefix': train_dataset_config['data_prefix'],
                                                'box_type_3d': dataset_config['box_type_3d'], 'origin': origin,
                                                'default_cam_key': self.config['dataset']['default_cam_key']}}
            else:
                train_dataloader['dataset'] = {'type': dataset_config['type'],
                                               'ann_file': train_dataset_config['ann_file'], 'pipeline': train_pipeline,
                                               'metainfo': {'classes': dataset_config['classes']},
                                               'modality': self.config['input_modality'], 'test_mode': False,
                                               'data_prefix': train_dataset_config['data_prefix'],
                                               'box_type_3d': dataset_config['box_type_3d'], 'origin': origin,
                                               'default_cam_key': self.config['dataset']['default_cam_key']}
            # validation dataloader
            temp = train_pipeline
            remove_pipe = ['GlobalRotScaleTrans', 'BEVFusionGlobalRotScaleTrans', 'BEVFusionRandomFlip3D', 'MyObjectRangeFilter', 'ObjectNameFilter', 'PointShuffle']
            val_pipeline = copy.deepcopy(list(filter(lambda i: i['type'] not in remove_pipe, temp)))

            for sub in val_pipeline:
                if sub['type'] == 'ImageAug3D':
                    sub['is_train'] = False
                    sub['rand_flip'] = False
                    sub['resize_lim'] = [0.48, 0.48]
                    sub['rot_lim'] = [0.0, 0.0]

            val_dataloader = {}
            val_dataloader['batch_size'] = val_dataset_config['batch_size']
            val_dataloader['num_workers'] = val_dataset_config['num_workers']
            val_dataloader['drop_last'] = False
            val_dataloader['persistent_workers'] = True
            val_dataloader['sampler'] = {'type': val_dataset_config['sampler'], 'shuffle': False}
            val_dataloader['dataset'] = {'type': dataset_config['type'],
                                         'ann_file': val_dataset_config['ann_file'], 'pipeline': val_pipeline,
                                         'metainfo': {'classes': dataset_config['classes']},
                                         'modality': self.config['input_modality'], 'test_mode': False,
                                         'data_prefix': val_dataset_config['data_prefix'],
                                         'box_type_3d': dataset_config['box_type_3d'], 'origin': origin,
                                         'default_cam_key': self.config['dataset']['default_cam_key']}

            self.updated_config['train_dataloader'] = train_dataloader
            self.updated_config['val_dataloader'] = val_dataloader
            self.updated_config['test_dataloader'] = None

            # validation evaluator
            self.updated_config['val_evaluator'] = {'type': 'TAO3DMetric', 'metric': 'bbox', 'gt_box_type': self.config['dataset']['gt_box_type'],
                                                    'origin': origin, 'yaw_dim': yaw_dim, 'is_synthetic': is_synthetic,
                                                    'pcd_limit_range': self.config['model']['point_cloud_range'],
                                                    'ann_file': val_dataset_config['ann_file'],
                                                    'default_cam_key': self.config['dataset']['default_cam_key'],
                                                    'backend_args': None}
        elif self.phase in ('evaluate', 'inference'):
            test_dataset_config = dataset_config['test_dataset']
            test_dataloader = {}
            test_dataloader['batch_size'] = test_dataset_config['batch_size']
            test_dataloader['num_workers'] = test_dataset_config['num_workers']
            test_dataloader['drop_last'] = False
            test_dataloader['persistent_workers'] = True
            test_dataloader['sampler'] = {'type': test_dataset_config['sampler'], 'shuffle': False}

            if not self.input_modality.get('use_lidar'):
                raise ValueError('input_modality - use_lidar must be set to True for this model')

            if not self.input_modality.get('use_camera'):
                # lidar-only
                test_pipeline = [
                    {
                        'type': 'TAOLoadPointsFromFile',
                        'coord_type': 'LIDAR',
                        'load_dim': self.config['dataset']['point_cloud_dim'],
                        'use_dim': self.config['dataset']['point_cloud_dim'],
                        'backend_args': None
                    },
                    {'type': 'PointsRangeFilter', 'point_cloud_range': self.config['model']['point_cloud_range']},
                    {
                        'type': 'Pack3DDetInputs', 'keys': ['points', 'gt_bboxes_3d', 'gt_labels_3d', 'gt_bboxes', 'gt_labels'],
                        'meta_keys': ['cam2img', 'ori_cam2img', 'lidar2cam', 'lidar2img', 'cam2lidar',
                                      'ori_lidar2img', 'box_type_3d', 'sample_idx',
                                      'lidar_path', 'transformation_3d_flow', 'pcd_rotation',
                                      'pcd_scale_factor', 'pcd_trans',
                                      'lidar_aug_matrix', 'num_pts_feats', 'num_views']
                    }
                ]
            else:
                # image-lidar fusion
                test_pipeline = [
                    {
                        'type': 'BEVFusionLoadMultiViewImageFromFiles',
                        'set_default_scale': True,
                        'to_float32': True,
                        'num_views': self.config['dataset']['num_views'],
                        'color_type': 'color',
                        'backend_args': None
                    },
                    {
                        'type': 'TAOLoadPointsFromFile',
                        'coord_type': 'LIDAR',
                        'load_dim': self.config['dataset']['point_cloud_dim'],
                        'use_dim': self.config['dataset']['point_cloud_dim'],
                        'backend_args': None
                    },
                    {
                        'type': 'ImageAug3D',
                        'final_dim': [256, 704],
                        'resize_lim': [0.48, 0.48],
                        'bot_pct_lim':  [0.0, 0.0],
                        'rot_lim': [0.0, 0.0],
                        'rand_flip': False,
                        'is_train': False
                    },
                    {'type': 'PointsRangeFilter', 'point_cloud_range': self.config['model']['point_cloud_range']},
                    {
                        'type': 'Pack3DDetInputs', 'keys': ['img', 'points', 'gt_bboxes_3d', 'gt_labels_3d', 'gt_bboxes', 'gt_labels'],
                        'meta_keys': ['cam2img', 'ori_cam2img', 'lidar2cam', 'lidar2img', 'cam2lidar',
                                      'ori_lidar2img', 'img_aug_matrix', 'box_type_3d', 'sample_idx',
                                      'lidar_path', 'img_path', 'transformation_3d_flow', 'pcd_rotation',
                                      'pcd_scale_factor', 'pcd_trans',
                                      'lidar_aug_matrix', 'num_pts_feats', 'num_views']
                    }
                ]

            if test_dataset_config['ann_file'] is None and self.phase == 'evaluate':
                raise ValueError('ann_file for test_dataset must be provided for evaluation')

            test_dataloader['dataset'] = {'type': dataset_config['type'],
                                          'ann_file': test_dataset_config['ann_file'], 'pipeline': test_pipeline,
                                          'metainfo': {'classes': dataset_config['classes']},
                                          'modality': self.config['input_modality'], 'test_mode': True,
                                          'load_eval_anns': True, 'data_prefix': test_dataset_config['data_prefix'],
                                          'box_type_3d': dataset_config['box_type_3d'], 'origin': origin,
                                          'default_cam_key': self.config['dataset']['default_cam_key']}
            self.updated_config['train_dataloader'] = None
            self.updated_config['val_dataloader'] = None
            self.updated_config['test_dataloader'] = test_dataloader
            self.updated_config['test_cfg'] = {}
        else:
            raise NotImplementedError(f'Phase {self.phase} is not supported yet')

    @abstractmethod
    def update_model_config(self):
        """Update the model config"""
        model_config = self.config['model']

        self.updated_config['model'] = {}
        self.updated_config['model']['type'] = model_config['type']

        # update data preprocessor
        point_cloud_range = model_config['point_cloud_range']
        self.updated_config['model']['data_preprocessor'] = model_config['data_preprocessor']
        self.updated_config['model']['data_preprocessor']['voxelize_cfg']['point_cloud_range'] = point_cloud_range
        self.updated_config['model']['data_preprocessor']['voxelize_cfg']['voxel_size'] = model_config['voxel_size']

        # update model architecture
        if not self.input_modality.get('use_lidar'):
            raise ValueError('input_modality - use_lidar must be set to True for this model.')

        if not self.input_modality.get('use_camera'):
            # lidar-only set model
            self.updated_config['model']['img_backbone'] = None
            self.updated_config['model']['img_neck'] = None
            self.updated_config['model']['view_transform'] = None
            self.updated_config['model']['fusion_layer'] = None
        else:
            self.updated_config['model']['img_backbone'] = model_config['img_backbone']
            self.updated_config['model']['img_neck'] = model_config['img_neck']
            self.updated_config['model']['view_transform'] = model_config['view_transform']
            self.updated_config['model']['fusion_layer'] = model_config['fusion_layer']

        self.updated_config['model']['pts_backbone'] = model_config['pts_backbone']
        self.updated_config['model']['pts_voxel_encoder'] = model_config['pts_voxel_encoder']
        self.updated_config['model']['pts_middle_encoder'] = model_config['pts_middle_encoder']
        self.updated_config['model']['pts_neck'] = model_config['pts_neck']

        grid_size = model_config.pop('grid_size')
        out_size_factor = model_config['bbox_head'].pop('out_size_factor')
        nms_type = model_config['bbox_head'].pop('nms_type')
        code_weights = model_config['bbox_head'].pop('code_weights')
        assigner = model_config['bbox_head'].pop('assigner')

        self.updated_config['model']['bbox_head'] = model_config['bbox_head']
        self.updated_config['model']['bbox_head']['bbox_coder']['pc_range'] = point_cloud_range
        self.updated_config['model']['bbox_head']['bbox_coder']['voxel_size'] = model_config['voxel_size']
        self.updated_config['model']['bbox_head']['bbox_coder']['out_size_factor'] = out_size_factor
        self.updated_config['model']['bbox_head']['bbox_coder']['post_center_range'] = model_config['post_center_range']

        self.updated_config['model']['bbox_head']['train_cfg'] = {'dataset': self.config['dataset']['type'],
                                                                  'grid_size': grid_size, 'out_size_factor': out_size_factor,
                                                                  'voxel_size': model_config['voxel_size'],
                                                                  'point_cloud_range': point_cloud_range,
                                                                  'nms_type': nms_type,
                                                                  'gaussian_overlap': 0.1,
                                                                  'min_radius': 2,
                                                                  'pos_weight': -1,
                                                                  'code_weights': code_weights,
                                                                  'assigner': assigner}

        self.updated_config['model']['bbox_head']['test_cfg'] = {'dataset': self.config['dataset']['type'],
                                                                 'grid_size': grid_size, 'out_size_factor': out_size_factor,
                                                                 'voxel_size': model_config['voxel_size'],
                                                                 'point_cloud_range': point_cloud_range,
                                                                 'nms_type': nms_type}

        self.updated_config['model']['test_cfg'] = {}

    def update_train_params_config(self):
        """Update train parameters"""
        train_param_config = self.config['train']

        self.updated_config['train_cfg'] = {'by_epoch': train_param_config['by_epoch'], 'max_epochs': train_param_config['num_epochs'],
                                            'val_interval': train_param_config['validation_interval']}
        self.updated_config['val_cfg'] = {}

        self.updated_config['default_hooks'] = self.config['default_hooks']
        self.updated_config["default_hooks"]["logger"]["type"] = self.config['logger_hook']
        self.updated_config['default_hooks']['checkpoint']['interval'] = train_param_config['checkpoint_interval']
        self.updated_config['default_hooks']['logger']['interval'] = train_param_config['logging_interval']
        optimizer_dict = train_param_config['optimizer']
        wrapper_type = optimizer_dict.pop('wrapper_type')
        clip_grad = optimizer_dict.pop('clip_grad')
        self.updated_config['optim_wrapper'] = {'type': wrapper_type, 'optimizer': optimizer_dict, 'clip_grad': clip_grad}

        # This is here just to satisfy the current way of setting the learning rate
        self.updated_config['param_scheduler'] = train_param_config['lr_scheduler']
        self.updated_config['resume'] = train_param_config['resume']
        if train_param_config.get("pretrained_checkpoint", None) == "":
            self.updated_config['load_from'] = None
        else:
            self.updated_config['load_from'] = train_param_config['pretrained_checkpoint']
