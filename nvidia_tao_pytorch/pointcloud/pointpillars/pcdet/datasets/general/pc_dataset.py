# SPDX-FileCopyrightText: Copyright (c) 2023 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Generic PintPillars data loader."""
import copy
import os
import pickle

import numpy as np

from ...ops.roiaware_pool3d import roiaware_pool3d_utils
from ...utils import object3d_general
from ..dataset import DatasetTemplate


class GeneralPCDataset(DatasetTemplate):
    """Generic data loader."""

    def __init__(self, dataset_cfg, class_names, training=True, root_path=None, info_path=None, logger=None):
        """
        Args:
            root_path:
            dataset_cfg:
            class_names:
            training:
            logger:
        """
        super().__init__(
            dataset_cfg=dataset_cfg, class_names=class_names, training=training,
            root_path=root_path, info_path=info_path, logger=logger
        )
        self.num_point_features = self.dataset_cfg.data_augmentor.aug_config_list[0].num_point_features
        self.split = self.dataset_cfg.data_split[self.mode]
        self.root_split_path = self.root_path / self.split
        lidar_path = self.root_split_path / "lidar"
        sample_id_list = os.listdir(lidar_path)
        assert len(sample_id_list), "lidar directory is empty"
        # strip .bin suffix
        self.sample_id_list = [x[:-4] for x in sample_id_list]
        for sid in self.sample_id_list:
            if len(self.get_label(sid)) == 0:
                raise IOError(
                    f"Got empty label for sample {sid} in {self.split} split"
                    ", please check the dataset"
                )
        self.infos = []
        self.include_data(self.mode)
        if self.training and self.dataset_cfg.get('balanced_resampling', False):
            self.infos = self.balanced_infos_resampling(self.infos)

    def include_data(self, mode):
        """Inlcude data files."""
        if self.logger is not None:
            self.logger.info('Loading point cloud dataset')
        pc_infos = []
        for info_path in self.dataset_cfg.info_path[mode]:
            info_path = self.info_path / info_path
            if not info_path.exists():
                continue
            with open(info_path, 'rb') as f:
                infos = pickle.load(f)
                pc_infos.extend(infos)
        self.infos.extend(pc_infos)
        if self.logger is not None:
            self.logger.info('Total samples for point cloud dataset: %d' % (len(pc_infos)))

    def balanced_infos_resampling(self, infos):
        """
        Class-balanced sampling of nuScenes dataset from https://arxiv.org/abs/1908.09492
        """
        if self.class_names is None:
            return infos
        cls_infos = {name: [] for name in self.class_names}
        for info in infos:
            for name in set(info["annos"]['name']):
                if name in self.class_names:
                    cls_infos[name].append(info)
        duplicated_samples = sum([len(v) for _, v in cls_infos.items()])
        cls_dist = {k: len(v) / duplicated_samples for k, v in cls_infos.items()}
        sampled_infos = []
        frac = 1.0 / len(self.class_names)
        ratios = [frac / v for v in cls_dist.values()]
        for cur_cls_infos, ratio in zip(list(cls_infos.values()), ratios):
            sampled_infos += np.random.choice(
                cur_cls_infos, int(len(cur_cls_infos) * ratio)
            ).tolist()
        if self.logger is not None:
            self.logger.info('Total samples after balanced resampling: %s' % (len(sampled_infos)))
        cls_infos_new = {name: [] for name in self.class_names}
        for info in sampled_infos:
            for name in set(info["annos"]['name']):
                if name in self.class_names:
                    cls_infos_new[name].append(info)
        return sampled_infos

    def set_split(self, split):
        """Setup train/val split."""
        super().__init__(
            dataset_cfg=self.dataset_cfg, class_names=self.class_names, training=self.training,
            root_path=self.root_path, info_path=self.info_path, logger=self.logger
        )
        self.split = split
        self.root_split_path = self.root_path / self.split
        lidar_path = self.root_split_path / "lidar"
        sample_id_list = []

        if os.path.isdir(lidar_path):
            sample_id_list = os.listdir(lidar_path)
        else:
            raise NotADirectoryError(f"{lidar_path} is not a directory")

        assert len(sample_id_list), "lidar directory is empty"
        # strip .bin suffix
        self.sample_id_list = [x[:-4] for x in sample_id_list]
        for sid in self.sample_id_list:
            if len(self.get_label(sid)) == 0:
                raise IOError(
                    f"Got empty label for sample {sid} in {split} split"
                    ", please check the dataset"
                )

    def get_lidar(self, idx):
        """Get LIDAR points."""
        lidar_file = self.root_split_path / 'lidar' / ('%s.bin' % idx)
        assert lidar_file.exists()
        points = np.fromfile(str(lidar_file), dtype=np.float32).reshape(-1, self.num_point_features)
        return points

    def get_label(self, idx):
        """Get KITTI labels."""
        label_file = self.root_split_path / 'label' / ('%s.txt' % idx)
        assert label_file.exists()
        return object3d_general.get_objects_from_label(label_file)

    def get_infos(self, num_workers=4, has_label=True, count_inside_pts=True, sample_id_list=None):
        """Get statistics info."""
        import concurrent.futures as futures

        def process_single_scene(sample_idx):
            print(f'{self.split} sample_idx: f{sample_idx}')
            info = {}
            pc_info = {'num_features': self.num_point_features, 'lidar_idx': sample_idx}
            info['point_cloud'] = pc_info
            if has_label:
                obj_list = self.get_label(sample_idx)
                annotations = {}
                annotations['name'] = np.array([obj.cls_type for obj in obj_list])
                annotations['truncated'] = np.array([obj.truncation for obj in obj_list])
                annotations['occluded'] = np.array([obj.occlusion for obj in obj_list])
                annotations['alpha'] = np.array([obj.alpha for obj in obj_list])
                annotations['bbox'] = np.concatenate([obj.box2d.reshape(1, 4) for obj in obj_list], axis=0)
                annotations['dimensions'] = np.array([[obj.l, obj.h, obj.w] for obj in obj_list])  # lhw(camera) format
                annotations['location'] = np.concatenate([obj.loc.reshape(1, 3) for obj in obj_list], axis=0)
                annotations['rotation_y'] = np.array([obj.ry for obj in obj_list])
                annotations['score'] = np.array([obj.score for obj in obj_list])
                num_objects = len(obj_list)
                index = list(range(num_objects))
                annotations['index'] = np.array(index, dtype=np.int32)
                loc = np.copy(annotations['location'])
                dims = annotations['dimensions']
                rots = annotations['rotation_y']
                l, h, w = dims[:, 0:1], dims[:, 1:2], dims[:, 2:3]
                gt_boxes_lidar = np.concatenate([loc, l, w, h, rots[..., np.newaxis]], axis=1)
                annotations['gt_boxes_lidar'] = gt_boxes_lidar
                info['annos'] = annotations
            return info

        sample_id_list = sample_id_list if sample_id_list is not None else self.sample_id_list
        with futures.ThreadPoolExecutor(num_workers) as executor:
            infos = executor.map(process_single_scene, sample_id_list)
        return list(infos)

    def create_groundtruth_database(self, info_path=None, used_classes=None, split='train'):
        """Create groundtruth database for augmentation."""
        import torch
        from pathlib import Path
        database_save_path = Path(self.info_path) / ('gt_database' if split == 'train' else ('gt_database_%s' % split))
        db_info_save_path = Path(self.info_path) / ('dbinfos_%s.pkl' % split)

        database_save_path.mkdir(parents=True, exist_ok=True)
        all_db_infos = {}

        infos = np.load(info_path, allow_pickle=True)

        for k in range(len(infos)):
            print('gt_database sample: %d/%d' % (k + 1, len(infos)))
            info = infos[k]
            sample_idx = info['point_cloud']['lidar_idx']
            points = self.get_lidar(sample_idx)
            annos = info['annos']
            names = annos['name']
            gt_boxes = annos['gt_boxes_lidar']

            num_obj = gt_boxes.shape[0]
            point_indices = roiaware_pool3d_utils.points_in_boxes_cpu(
                torch.from_numpy(points[:, 0:3]), torch.from_numpy(gt_boxes)
            ).numpy()  # (nboxes, npoints)

            for i in range(num_obj):
                filename = '%s_%s_%d.bin' % (sample_idx, names[i], i)
                filepath = database_save_path / filename
                gt_points = points[point_indices[i] > 0]

                gt_points[:, :3] -= gt_boxes[i, :3]
                with open(filepath, 'w') as f:
                    gt_points.tofile(f)

                if (used_classes is None) or names[i] in used_classes:
                    db_path = str(filepath.relative_to(self.info_path))  # gt_database/xxxxx.bin
                    db_info = {'name': names[i], 'path': db_path, 'image_idx': sample_idx, 'gt_idx': i,
                               'box3d_lidar': gt_boxes[i], 'num_points_in_gt': gt_points.shape[0],
                               }
                    if names[i] in all_db_infos:
                        all_db_infos[names[i]].append(db_info)
                    else:
                        all_db_infos[names[i]] = [db_info]
        for k, v in all_db_infos.items():
            print('Database %s: %d' % (k, len(v)))

        with open(db_info_save_path, 'wb') as f:
            pickle.dump(all_db_infos, f)

    @staticmethod
    def generate_prediction_dicts(batch_dict, pred_dicts, class_names, output_path=None):
        """
        Args:
            batch_dict:
                frame_id:
            pred_dicts: list of pred_dicts
                pred_boxes: (N, 7), Tensor
                pred_scores: (N), Tensor
                pred_labels: (N), Tensor
            class_names:
            output_path:

        Returns:

        """
        def get_template_prediction(num_samples):
            """Get template for prediction result."""
            ret_dict = {
                'name': np.zeros(num_samples), 'truncated': np.zeros(num_samples),
                'occluded': np.zeros(num_samples), 'alpha': np.zeros(num_samples),
                'bbox': np.zeros([num_samples, 4]), 'dimensions': np.zeros([num_samples, 3]),
                'location': np.zeros([num_samples, 3]), 'rotation_y': np.zeros(num_samples),
                'score': np.zeros(num_samples), 'boxes_lidar': np.zeros([num_samples, 7])
            }
            return ret_dict

        def generate_single_sample_dict(batch_index, box_dict):
            """Get single prediction result."""
            pred_scores = box_dict['pred_scores'].cpu().numpy()
            pred_boxes = box_dict['pred_boxes'].cpu().numpy()
            pred_labels = box_dict['pred_labels'].cpu().numpy()
            pred_dict = get_template_prediction(pred_scores.shape[0])
            if pred_scores.shape[0] == 0:
                return pred_dict
            pred_dict['name'] = np.array(class_names)[pred_labels - 1]
            pred_dict['score'] = pred_scores
            pred_dict['boxes_lidar'] = pred_boxes
            return pred_dict

        annos = []
        for index, box_dict in enumerate(pred_dicts):
            frame_id = batch_dict['frame_id'][index]

            single_pred_dict = generate_single_sample_dict(index, box_dict)
            single_pred_dict['frame_id'] = frame_id
            annos.append(single_pred_dict)

            if output_path:
                cur_det_file = output_path / ('%s.txt' % frame_id)
                with open(cur_det_file, 'w') as f:
                    box_lidar = single_pred_dict['boxes_lidar']
                    for idx in range(len(box_lidar)):
                        x, y, z, l, w, h, rt = box_lidar[idx]
                        print('%s -1 -1 %.4f %.4f %.4f %.4f %.4f %.4f %.4f %.4f %.4f %.4f %.4f %.4f %.4f'
                              % (single_pred_dict['name'][idx], 0,
                                 0, 0, 0, 0,
                                 h, w, l, x, y, z, rt,
                                 single_pred_dict['score'][idx]), file=f)

        return annos

    def evaluation(self, det_annos, class_names, **kwargs):
        """Evaluation of prediction results."""
        if 'annos' not in self.infos[0].keys():
            return None, {}

        from ..kitti.kitti_object_eval_python import eval as kitti_eval

        eval_det_annos = copy.deepcopy(det_annos)
        eval_gt_annos = [copy.deepcopy(info['annos']) for info in self.infos]
        ap_result_str, ap_dict = kitti_eval.get_kitti_eval_result(eval_gt_annos, eval_det_annos, class_names)

        return ap_result_str, ap_dict

    def __len__(self):
        """Length."""
        if self._merge_all_iters_to_one_epoch:
            return len(self.infos) * self.total_epochs

        return len(self.infos)

    def __getitem__(self, index):
        """Get item."""
        # index = 4
        if self._merge_all_iters_to_one_epoch:
            index = index % len(self.infos)

        info = copy.deepcopy(self.infos[index])
        sample_idx = info['point_cloud']['lidar_idx']
        points = self.get_lidar(sample_idx)
        input_dict = {
            'points': points,
            'frame_id': sample_idx,
        }
        if 'annos' in info:
            annos = info['annos']
            gt_names = annos['name']
            gt_boxes_lidar = annos["gt_boxes_lidar"]
            input_dict.update({
                'gt_names': gt_names,
                'gt_boxes': gt_boxes_lidar
            })
        data_dict = self.prepare_data(data_dict=input_dict)
        return data_dict


def create_pc_infos(dataset_cfg, class_names, data_path, save_path, status_logging, workers=4):
    """Create point cloud statistics for data augmentations."""
    dataset = GeneralPCDataset(
        dataset_cfg=dataset_cfg,
        class_names=class_names,
        root_path=data_path,
        info_path=dataset_cfg.data_info_path,
        training=False
    )
    train_split = dataset_cfg.data_split['train']
    val_split = dataset_cfg.data_split['test']
    train_filename = save_path / (f"infos_{train_split}.pkl")
    val_filename = save_path / (f"infos_{val_split}.pkl")
    trainval_filename = save_path / (f"infos_{train_split}_{val_split}.pkl")
    print('---------------Start to generate data infos---------------')
    status_logging.get_status_logger().write(
        message="---------------Start to generate data infos---------------",
        status_level=status_logging.Status.STARTED
    )
    dataset.set_split(train_split)
    infos_train = dataset.get_infos(num_workers=workers, has_label=True, count_inside_pts=False)
    with open(train_filename, 'wb') as f:
        pickle.dump(infos_train, f)
    print(f'Info train file is saved to {train_filename}')
    status_logging.get_status_logger().write(
        message='Info train file is saved to %s' % train_filename,
        status_level=status_logging.Status.RUNNING
    )
    dataset.set_split(val_split)
    infos_val = dataset.get_infos(num_workers=workers, has_label=True, count_inside_pts=False)
    with open(val_filename, 'wb') as f:
        pickle.dump(infos_val, f)
    print(f'Info val file is saved to {val_filename}')
    status_logging.get_status_logger().write(
        message='Info val file is saved to %s' % val_filename,
        status_level=status_logging.Status.RUNNING
    )
    with open(trainval_filename, 'wb') as f:
        pickle.dump(infos_train + infos_val, f)
    print(f'Info trainval file is saved to {trainval_filename}')
    status_logging.get_status_logger().write(
        message='Info trainval file is saved to %s' % trainval_filename,
        status_level=status_logging.Status.RUNNING
    )
    print('---------------Start create groundtruth database for data augmentation---------------')
    status_logging.get_status_logger().write(
        message='---------------Start create groundtruth database for data augmentation---------------',
        status_level=status_logging.Status.RUNNING
    )
    dataset.set_split(train_split)
    dataset.create_groundtruth_database(train_filename, split=train_split)
    print('---------------Data preparation Done---------------')
    status_logging.get_status_logger().write(
        message='---------------Data preparation Done---------------',
        status_level=status_logging.Status.RUNNING
    )
