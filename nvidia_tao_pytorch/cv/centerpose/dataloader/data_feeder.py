# SPDX-FileCopyrightText: Copyright (c) 2023 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Data feeder for loading CenterPose sequences."""

import numpy as np
import glob
import cv2
import copy
import math
import json
import logging
from torch.utils.data import Dataset
import os
from os.path import exists
from scipy.spatial.transform import Rotation as R

from nvidia_tao_pytorch.cv.centerpose.dataloader.augmentation import draw_dense_reg, gaussian_radius, bounding_box_rotation, rotation_y_matrix, get_affine_transform, affine_transform, draw_umich_gaussian, color_aug


# List of valid image extensions
VALID_IMAGE_EXTENSIONS = ["jpg", "jpeg", "png", "bmp", "JPEG", "JPG", "PNG"]
logger = logging.getLogger(__name__)


class ObjectPoseDataset(Dataset):
    """Dataload for CenterPose model"""

    def __init__(self, data_path, configs, split):
        """Initialize the image path and ground truth"""
        super(ObjectPoseDataset, self).__init__()

        self.split = split
        self.opt = configs

        self.num_classes = self.opt.num_classes
        self.num_joints = self.opt.num_joints

        self.mean = np.array(self.opt.mean, dtype=np.float32).reshape(1, 1, 3)
        self.std = np.array(self.opt.std, dtype=np.float32).reshape(1, 1, 3)

        self.img_dir = data_path

        self.max_objs = self.opt.max_objs

        # Eigenvalues and eigenvector for color data augmentation from the CenterNet
        self._data_rng = np.random.RandomState(123)
        self._eig_val = np.array(self.opt._eig_val, dtype=np.float32)
        self._eig_vec = np.array(self.opt._eig_vec, dtype=np.float32)

        logger.info('Initializing {} {} data.'.format(split, self.opt.category))
        self.images = []
        self.images += self._load_data(self.img_dir, extensions=VALID_IMAGE_EXTENSIONS)
        self.num_samples = len(self.images)
        if self.num_samples == 0:
            raise FileNotFoundError(f"No valid image with extensions {VALID_IMAGE_EXTENSIONS} found in following directories {self.img_dir}.")
        logger.info('Loaded {} {} samples'.format(split, self.num_samples))

    def __len__(self):
        """Return the number of images"""
        return self.num_samples

    def _loadimages(self, root, extensions=VALID_IMAGE_EXTENSIONS):
        """Load the images and its related json annotation files"""
        imgs = []

        def add_json_files(path, ):
            for ext in extensions:
                for imgpath in glob.glob(path + "/*.{}".format(ext.replace('.', ''))):
                    if exists(imgpath) and exists(imgpath.replace(ext, "json")):
                        imgs.append((imgpath, imgpath.replace(ext, "json")))

        def explore(path):
            if not os.path.isdir(path):
                return
            folders = [os.path.join(path, o) for o in os.listdir(path) if os.path.isdir(os.path.join(path, o))]
            if len(folders) > 0:
                for path_entry in folders:
                    explore(path_entry)
            else:
                add_json_files(path)

        explore(root)
        return imgs

    def _load_data(self, path, extensions):
        """Load the images and its related json annotation files"""
        imgs = self._loadimages(path, extensions=extensions)
        return imgs

    def _get_border(self, border, size):
        """Return the border info for cropping the image"""
        i = 1
        while size - border // i <= border // i:
            i *= 2
        return border // i

    # Add noise
    def _get_aug_param(self, c_ori, s, width, height, disturb=False):
        """Return the cropping and rotation aumentation parameters"""
        c = c_ori.copy()
        if (not self.opt.not_rand_crop) and not disturb:
            # Training for current frame
            aug_s = np.random.choice(np.arange(0.6, 1.4, 0.1))
            w_border = self._get_border(128, width)
            h_border = self._get_border(128, height)
            c[0] = np.random.randint(low=w_border, high=width - w_border)
            c[1] = np.random.randint(low=h_border, high=height - h_border)
        else:
            # Training for previous frame
            sf = self.opt.scale
            cf = self.opt.shift

            c[0] += s * np.clip(np.random.randn() * cf, -2 * cf, 2 * cf)
            c[1] += s * np.clip(np.random.randn() * cf, -2 * cf, 2 * cf)
            aug_s = np.clip(np.random.randn() * sf + 1, 1 - sf, 1 + sf)

        if np.random.random() < self.opt.aug_rot:
            rf = self.opt.rotate  # 0 - 180
            rot = 2 * (np.random.rand() - 0.5) * rf
        else:
            rot = 0

        return c, aug_s, rot

    def _transform_bbox(self, bbox, trans, width, height):
        """Transform bounding boxes according to image crop."""
        bbox = np.array(copy.deepcopy(bbox), dtype=np.float32)
        bbox[:2] = affine_transform(bbox[:2], trans)
        bbox[2:] = affine_transform(bbox[2:], trans)
        bbox[[0, 2]] = np.clip(bbox[[0, 2]], 0, width - 1)
        bbox[[1, 3]] = np.clip(bbox[[1, 3]], 0, height - 1)
        return bbox

    def _get_input(self, img, trans_input):
        """Augmentation of the image"""
        inp = cv2.warpAffine(img, trans_input,
                             (self.opt.input_res, self.opt.input_res),
                             flags=cv2.INTER_LINEAR)

        inp = (inp.astype(np.float32) / 255.)
        if self.split == 'train' and not self.opt.no_color_aug:
            color_aug(self._data_rng, inp, self._eig_val, self._eig_vec)

        inp = (inp - self.mean) / self.std
        inp = inp.transpose(2, 0, 1)
        return inp

    def __getitem__(self, index):
        """Return the image and label in the dataloader"""
        img_path, path_json = self.images[index]

        with open(path_json) as f:
            anns = json.load(f)
        num_objs = min(len(anns['objects']), self.max_objs)

        img = cv2.imread(img_path)
        height, width = img.shape[0], img.shape[1]

        c_ori = np.array([img.shape[1] / 2., img.shape[0] / 2.], dtype=np.float32)
        s_ori = max(img.shape[0], img.shape[1]) * 1.0
        rot = 0

        flipped = False
        if self.split == 'train':

            c, aug_s, rot = self._get_aug_param(c_ori, s_ori, width, height, disturb=False)
            s = s_ori * aug_s

            if np.random.random() < self.opt.flip:
                flipped = True
                img = img[:, ::-1, :]

                c[0] = width - c[0] - 1
        else:
            c = c_ori
            s = s_ori

        trans_input = get_affine_transform(
            c, s, rot, [self.opt.input_res, self.opt.input_res])

        inp = self._get_input(img, trans_input)

        output_res = self.opt.output_res
        num_joints = self.num_joints
        trans_output_rot = get_affine_transform(c, s, rot, [output_res, output_res])

        # Parameter initialization
        # Set the rotational symmetry, will be varied according to the category
        # Set these two categories' rotational symmetry for the Objectron Dataset
        if self.opt.category == 'chair':
            num_symmetry = 4
            theta = 2 * np.pi / num_symmetry
        elif self.opt.category == 'bottle':
            num_symmetry = 12
            theta = 2 * np.pi / num_symmetry
        else:
            # Set the customized rotational symmetry
            num_symmetry = self.opt.num_symmetry
            theta = 2 * np.pi / num_symmetry

        # All the gt info:
        hm = np.zeros((num_symmetry, self.num_classes, output_res, output_res), dtype=np.float32)
        hm_hp = np.zeros((num_symmetry, num_joints, output_res, output_res), dtype=np.float32)
        dense_kps = np.zeros((num_symmetry, num_joints, 2, output_res, output_res),
                             dtype=np.float32)
        dense_kps_mask = np.zeros((num_symmetry, num_joints, output_res, output_res),
                                  dtype=np.float32)
        wh = np.zeros((num_symmetry, self.max_objs, 2), dtype=np.float32)
        scale = np.zeros((num_symmetry, self.max_objs, 3), dtype=np.float32)

        kps = np.zeros((num_symmetry, self.max_objs, num_joints * 2), dtype=np.float32)
        kps_displacement_std = np.zeros((num_symmetry, self.max_objs, num_joints * 2), dtype=np.float32)

        reg = np.zeros((num_symmetry, self.max_objs, 2), dtype=np.float32)
        ind = np.zeros((num_symmetry, self.max_objs), dtype=np.int64)
        reg_mask = np.zeros((num_symmetry, self.max_objs), dtype=np.uint8)
        kps_mask = np.zeros((num_symmetry, self.max_objs, num_joints * 2), dtype=np.uint8)
        hp_offset = np.zeros((num_symmetry, self.max_objs * num_joints, 2), dtype=np.float32)
        hp_ind = np.zeros((num_symmetry, self.max_objs * num_joints), dtype=np.int64)
        hp_mask = np.zeros((num_symmetry, self.max_objs * num_joints), dtype=np.int64)

        draw_gaussian = draw_umich_gaussian

        # <editor-fold desc="Step2: Work on the current frame">
        cam_projection_matrix = anns['camera_data']['camera_projection_matrix']

        # add the camera intrinsic matrix
        intrinsic = np.identity(3)
        intrinsic[0, 0] = anns['camera_data']['intrinsics']['fx']
        intrinsic[0, 2] = anns['camera_data']['intrinsics']['cx']
        intrinsic[1, 1] = anns['camera_data']['intrinsics']['fy']
        intrinsic[1, 2] = anns['camera_data']['intrinsics']['cy']

        for k in range(num_objs):
            ann = anns['objects'][k]

            # Only for Objectron "chair" category
            # Because some of the chairs are symmetry in 4 directions.
            if 'symmetric' in ann:
                if ann['symmetric'] == 'True':
                    num_symmetry = 4
                else:
                    num_symmetry = 1

            # Fixed as the class id = 0 because the CenterPose is category-based method
            cls_id = 0
            pts_ori = np.array(ann['projected_cuboid'])

            # Only apply rotation on gt annotation when symmetry exists
            for id_symmetry in range(num_symmetry):

                if num_symmetry != 1:

                    object_rotations = ann['quaternion_xyzw']
                    object_translations = ann['location']
                    keypoints_3d = np.array(ann['keypoints_3d'])

                    M_o2c = np.identity(4)
                    M_o2c[:3, :3] = R.from_quat(object_rotations).as_matrix()
                    M_o2c[:3, 3] = object_translations

                    M_c2o = np.linalg.inv(M_o2c)

                    M_R = rotation_y_matrix(theta * id_symmetry)

                    # Project the rotated 3D keypoint to the image plane
                    M_trans = cam_projection_matrix @ M_o2c @ M_R @ M_c2o

                    new_keypoints_2d = []
                    for i in range(9):
                        projected_point_ori = M_trans @ np.vstack((keypoints_3d[i].reshape(3, -1), 1))
                        projected_point_ori = (projected_point_ori / projected_point_ori[3])[:3]
                        viewport_point = (projected_point_ori + 1.0) / 2.0 * np.array([height, width, 1.0]).reshape(3,
                                                                                                                    1)
                        new_keypoints_2d.append([int(viewport_point[1]), int(viewport_point[0])])

                    pts_ori = new_keypoints_2d

                ct_ori = pts_ori[0]  # center
                pts_ori = pts_ori[1:]  # 8 corners

                # Change visibility, following the protocol of COCO
                pts = np.zeros((len(pts_ori), 3), dtype='int64')
                for idx, p in enumerate(pts_ori):
                    if p[0] >= width or p[0] < 0 or p[1] < 0 or p[1] >= height:
                        pts[idx] = [p[0], p[1], 1]  # labeled but not visible
                    else:
                        pts[idx] = [p[0], p[1], 2]  # labeled and visible

                # Horizontal flip
                if flipped:
                    pts[:, 0] = width - pts[:, 0] - 1
                    for e in self.opt.flip_idx:
                        temp_1 = e[1] - 1
                        temp_0 = e[0] - 1
                        pts[temp_0], pts[temp_1] = pts[temp_1].copy(), pts[temp_0].copy()

                bbox = np.array(bounding_box_rotation(pts, trans_output_rot))

                bbox = np.clip(bbox, 0, output_res - 1)
                h, w = bbox[3] - bbox[1], bbox[2] - bbox[0]

                # Filter out GT if most of the keypoints are not visible (more than 4)
                visible_flag = True
                if ct_ori[0] >= width or ct_ori[0] < 0 or ct_ori[1] < 0 or ct_ori[1] >= height:
                    if pts[:, 2].sum() <= 12:
                        visible_flag = False

                if ((h > 0 and w > 0) or (rot != 0)) and visible_flag:
                    radius = gaussian_radius((math.ceil(h), math.ceil(w)))
                    radius = self.opt.hm_gauss if self.opt.mse_loss else max(0, int(radius))

                    if self.opt.center_3D is False:
                        # Need modification, bbox is not accurate enough as we do not have gt info from Objectron Dataset
                        ct = np.array(
                            [(bbox[0] + bbox[2]) / 2, (bbox[1] + bbox[3]) / 2], dtype=np.float32)
                        ct_int = ct.astype(np.int32)
                    else:
                        # Right now, do not consider objects whose center is out of the image
                        if flipped:
                            ct_ori[0] = width - ct_ori[0] - 1
                        ct = affine_transform(ct_ori, trans_output_rot)
                        ct_int = ct.astype(np.int32)
                        if ct_int[0] >= output_res or ct_int[1] >= output_res or ct_int[0] < 0 or ct_int[1] < 0:
                            continue

                    # Currently, normalized by y axis (up)
                    if self.opt.obj_scale:
                        if self.opt.use_absolute_scale:
                            scale[id_symmetry, k] = np.abs(ann['scale'])
                        else:
                            # normalized by z-axis (up)
                            scale[id_symmetry, k] = np.abs(ann['scale']) / ann['scale'][1]

                    wh[id_symmetry, k] = 1. * w, 1. * h
                    ind[id_symmetry, k] = ct_int[1] * output_res + ct_int[0]
                    reg[id_symmetry, k] = ct - ct_int
                    reg_mask[id_symmetry, k] = 1

                    # From CenterNet, not used in our case
                    num_kpts = pts[:, 2].sum()
                    if num_kpts == 0:
                        hm[id_symmetry, cls_id, ct_int[1], ct_int[0]] = 0.9999
                        reg_mask[id_symmetry, k] = 0

                    # Todo: Currently, hp_radius follows the same way as radius
                    hp_radius = radius
                    for j in range(num_joints):
                        # Every point no matter if it is visible or not will be converted first
                        pts[j, :2] = affine_transform(pts[j, :2], trans_output_rot)
                        if pts[j, 2] > 1:  # Check visibility
                            if pts[j, 0] >= 0 and pts[j, 0] < output_res and \
                                    pts[j, 1] >= 0 and pts[j, 1] < output_res:
                                kps[id_symmetry, k, j * 2: j * 2 + 2] = pts[j, :2] - ct_int
                                # Todo: Use the same std as what is used in heatmap, not used yet
                                kps_displacement_std[id_symmetry, k, j * 2: j * 2 + 2] = hp_radius

                                kps_mask[id_symmetry, k, j * 2: j * 2 + 2] = 1
                                pt_int = pts[j, :2].astype(np.int32)
                                hp_offset[id_symmetry, k * num_joints + j] = pts[j, :2] - pt_int
                                hp_ind[id_symmetry, k * num_joints + j] = pt_int[1] * output_res + pt_int[0]
                                hp_mask[id_symmetry, k * num_joints + j] = 1

                                if self.opt.dense_hp:
                                    # Must be before draw center hm gaussian
                                    draw_dense_reg(dense_kps[id_symmetry, j], hm[id_symmetry, cls_id], ct_int,
                                                   pts[j, :2] - ct_int, radius, is_offset=True)
                                    draw_gaussian(dense_kps_mask[id_symmetry, j], ct_int, radius)
                                draw_gaussian(hm_hp[id_symmetry, j], pt_int, hp_radius)

                    # For center point
                    draw_gaussian(hm[id_symmetry, cls_id], ct_int, radius)

        # <editor-fold desc="Update data record">
        ret = {'input': inp, 'hm': hm, 'reg_mask': reg_mask, 'ind': ind,
               'hps': kps, 'hps_mask': kps_mask}
        if self.opt.hps_uncertainty:
            ret.update({'hps_uncertainty': kps_displacement_std})

        if self.opt.obj_scale:
            ret.update({'scale': scale})

        if self.opt.reg_bbox:
            ret.update({'wh': wh})
        if self.opt.dense_hp:
            dense_kps = dense_kps.reshape(num_symmetry, num_joints * 2, output_res, output_res)
            dense_kps_mask = dense_kps_mask.reshape(num_symmetry,
                                                    num_joints, 1, output_res, output_res)
            dense_kps_mask = np.concatenate([dense_kps_mask, dense_kps_mask], axis=2)
            dense_kps_mask = dense_kps_mask.reshape(num_symmetry,
                                                    num_joints * 2, output_res, output_res)
            ret.update({'dense_hps': dense_kps, 'dense_hps_mask': dense_kps_mask})
            del ret['hps'], ret['hps_mask']
        if self.opt.reg_offset:
            ret.update({'reg': reg})
        if self.opt.hm_hp:
            ret.update({'hm_hp': hm_hp})
        if self.opt.reg_hp_offset:
            ret.update({'hp_offset': hp_offset, 'hp_ind': hp_ind, 'hp_mask': hp_mask})
        if self.split == 'test':
            ret.update({'principle_points': c, 'max_axis': s, 'path_json': path_json, 'intrinsic_matrix': intrinsic})
        return ret


class CPPredictDataset(Dataset):
    """Base CenterPose Predict Dataset Class."""

    def __init__(self, dataset_config):
        """Initialize the CenterPose Dataset Class for inference.

        Unlike ObjectPoseDataset, this class does not require JSON file.

        Args:
            dataset_list (list): list of dataset directory.
            transforms: augmentations to apply.

        Raises:
            FileNotFoundErorr: If provided sequence or image extension does not exist.
        """
        self.inference_data = dataset_config.inference_data
        self.opt = dataset_config

        self.mean = np.array(self.opt.mean, dtype=np.float32).reshape(1, 1, 3)
        self.std = np.array(self.opt.std, dtype=np.float32).reshape(1, 1, 3)

        self.ids = []
        self.images = []
        self.images += self._load_data(self.inference_data)
        if len(self.images) == 0:
            raise FileNotFoundError(f"No valid image with extensions {VALID_IMAGE_EXTENSIONS} found in following directories {self.inference_data}.")
        logger.info('Initializing {} inference images.'.format(len(self.images)))

    def __len__(self):
        """Return the number of images"""
        return len(self.images)

    def _load_inference_images(self, root, extensions=VALID_IMAGE_EXTENSIONS):
        """Load the inference files inside the folder"""
        imgs = []

        def add_img_files(path, ):
            for ext in extensions:
                for imgpath in glob.glob(path + "/*.{}".format(ext.replace('.', ''))):
                    if exists(imgpath):
                        imgs.append(imgpath)

        def explore(path):
            if not os.path.isdir(path):
                return
            folders = [os.path.join(path, o) for o in os.listdir(path) if os.path.isdir(os.path.join(path, o))]
            if len(folders) > 0:
                for path_entry in folders:
                    explore(path_entry)
            else:
                add_img_files(path)

        explore(root)
        return imgs

    def _load_data(self, path):
        """Load the inference images according to the extensions"""
        imgs = self._load_inference_images(path)
        return imgs

    def _preprocess_img(self, img, trans_input):
        """Processing the inference image"""
        inp = cv2.warpAffine(img, trans_input,
                             (self.opt.input_res, self.opt.input_res),
                             flags=cv2.INTER_LINEAR)
        inp = (inp.astype(np.float32) / 255.)
        inp = (inp - self.mean) / self.std
        inp = inp.transpose(2, 0, 1)
        return inp

    def __getitem__(self, index):
        """Get image_path given index.

        Args:
            index (int): index of the image id to load.

        Returns:
            image: pre-processed image for the model.
        """
        img_path = self.images[index]

        img = cv2.imread(img_path)

        c = np.array([img.shape[1] / 2., img.shape[0] / 2.], dtype=np.float32)
        s = max(img.shape[0], img.shape[1]) * 1.0
        # Set to 0 since it is the inference process
        rot = 0

        trans_input = get_affine_transform(
            c, s, rot, [self.opt.input_res, self.opt.input_res])

        inp = self._preprocess_img(img, trans_input)
        ret = {'input': inp, 'img_path': img_path, 'principle_points': c, 'max_axis': s}
        return ret
