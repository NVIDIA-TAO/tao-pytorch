# SPDX-FileCopyrightText: Copyright (c) 2023 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

""" CenterPose Evaluator. """


import os
import json
import math
import numpy as np
import scipy
from scipy.spatial.transform import Rotation as rotation_util
from tabulate import tabulate
from nvidia_tao_pytorch.cv.centerpose.dataloader.augmentation import rotation_y_matrix, safe_divide
import nvidia_tao_pytorch.cv.centerpose.utils.iou3d as Box


_MAX_PIXEL_ERROR = 0.1
_MAX_AZIMUTH_ERROR = 30.
_MAX_POLAR_ERROR = 20.
_MAX_SCALE_ERROR = 2.
_MAX_DISTANCE = 1.0
_NUM_BINS = 21
METRIC_UPDATED = True
UNIT_BOX = np.asarray([
    [0., 0., 0.],
    [-0.5, -0.5, -0.5],
    [-0.5, -0.5, 0.5],
    [-0.5, 0.5, -0.5],
    [-0.5, 0.5, 0.5],
    [0.5, -0.5, -0.5],
    [0.5, -0.5, 0.5],
    [0.5, 0.5, -0.5],
    [0.5, 0.5, 0.5],
])


class HitMiss(object):
    """Class for recording hits and misses of detection results."""

    def __init__(self, thresholds):
        """Initialize the recording hits and misses"""
        self.thresholds = thresholds
        self.size = thresholds.shape[0]
        self.hit = []
        self.miss = []
        for _ in range(self.size):
            self.hit.append([])
            self.miss.append([])

    def reset(self):
        """Reset the recording hits and misses"""
        self.hit = []
        self.miss = []
        for _ in range(self.size):
            self.hit.append([])
            self.miss.append([])

    def record_hit_miss(self, metric, greater=True):
        """Records the hit or miss for the object based on the metric threshold."""
        for i in range(self.size):
            threshold = self.thresholds[i]
            hit_flag = (greater and metric[0] >= threshold) or ((not greater) and metric[0] <= threshold)
            if hit_flag:
                self.hit[i].append([1, metric[1]])
                self.miss[i].append([0, metric[1]])
            else:
                self.hit[i].append([0, metric[1]])
                self.miss[i].append([1, metric[1]])


class AveragePrecision(object):
    """Class for computing average precision."""

    def __init__(self, size):
        """Initialize the average precision parameters"""
        self.size = size
        self.aps = np.zeros(size)
        self.true_positive = []
        self.false_positive = []
        for _ in range(size):
            self.true_positive.append([])
            self.false_positive.append([])
        self._total_instances = 0.

    def append(self, hit_miss, num_instances):
        """Appending the ture positive and false positive results"""
        for i in range(self.size):
            self.true_positive[i].append(hit_miss.hit[i])
            self.false_positive[i].append(hit_miss.miss[i])
        self._total_instances += num_instances

    def compute_ap(self, recall, precision):
        """Calculates the AP given the recall and precision array.

        The reference implementation is from Pascal VOC 2012 eval script. First we
        filter the precision recall rate so precision would be monotonically
        decreasing. Next, we compute the average precision by numerically
        integrating the precision-recall curve.

        Args:
          recall: Recall list
          precision: Precision list

        Returns:
          Average precision.
        """
        recall = np.insert(recall, 0, [0.])
        recall = np.append(recall, [1.])
        precision = np.insert(precision, 0, [0.])
        precision = np.append(precision, [0.])
        monotonic_precision = precision.copy()

        # Make the precision monotonically decreasing.
        for i in range(len(monotonic_precision) - 2, -1, -1):
            monotonic_precision[i] = max(monotonic_precision[i],
                                         monotonic_precision[i + 1])

        recall_changes = []
        for i in range(1, len(recall)):
            if recall[i] != recall[i - 1]:
                recall_changes.append(i)

        # Compute the average precision by integrating the recall curve.
        ap = 0.0
        for step in recall_changes:
            delta_recall = recall[step] - recall[step - 1]
            ap += delta_recall * monotonic_precision[step]
        return ap

    def compute_ap_curve(self):
        """Computes the precision/recall curve."""
        for i in range(self.size):
            tp, fp = self.true_positive[i], self.false_positive[i]

            tp_decoded = []
            for j in tp:
                for k in j:
                    tp_decoded.append(k)

            fp_decoded = []
            for j in fp:
                for k in j:
                    fp_decoded.append(k)

            if len(fp_decoded) != 0 and len(tp_decoded) != 0:
                combined = np.concatenate((tp_decoded, fp_decoded), axis=1).astype('float32')
                # Sort
                combined = combined[np.argsort(-combined[:, 1])]
                tp = combined[:, 0]
                fp = combined[:, 2]

                tp = np.cumsum(tp)
                fp = np.cumsum(fp)
                tp_fp = tp + fp
                recall = tp / (self._total_instances + 1e-15)

                precision = np.divide(tp, tp_fp, out=np.zeros_like(tp), where=tp_fp != 0)

                self.aps[i] = self.compute_ap(recall, precision)
            else:
                self.aps[i] = 0


class Accuracy(object):
    """Class for accuracy metric."""

    def __init__(self):
        """Initialize the accuracy metric"""
        self._errors = []
        self.acc = []

    def add_error(self, error):
        """Adds an error."""
        self._errors.append(error)

    def compute_accuracy(self, thresh=0.1):
        """Computes accuracy for a given threshold."""
        if not self._errors:
            return 0
        return len(np.where(np.array(self._errors) <= thresh)[0]) * 100. / (
            len(self._errors))


class Evaluator(object):
    """Class for evaluating a deep pursuit model."""

    def __init__(self, experiment_spec):
        """Initialize all the accuracy parameters"""
        self.opt = experiment_spec.dataset
        self.eval_config = experiment_spec.evaluate
        self._vis_thresh = 0.1
        self._error_scale = 0.
        self._error_2d = 0.
        self._matched = 0
        self._iou_3d = 0.
        self._azimuth_error = 0.
        self._polar_error = 0.

        self._scale_thresholds = np.linspace(0.0, 1., num=_NUM_BINS)
        self._iou_thresholds = np.linspace(0.0, 1., num=_NUM_BINS)
        self._pixel_thresholds = np.linspace(0.0, _MAX_PIXEL_ERROR, num=_NUM_BINS)
        self._azimuth_thresholds = np.linspace(0.0, _MAX_AZIMUTH_ERROR, num=_NUM_BINS)
        self._polar_thresholds = np.linspace(0.0, _MAX_POLAR_ERROR, num=_NUM_BINS)
        self._add_thresholds = np.linspace(0.0, _MAX_DISTANCE, num=_NUM_BINS)
        self._adds_thresholds = np.linspace(0.0, _MAX_DISTANCE, num=_NUM_BINS)

        self._scale_ap = AveragePrecision(_NUM_BINS)
        self._iou_ap = AveragePrecision(_NUM_BINS)
        self._pixel_ap = AveragePrecision(_NUM_BINS)
        self._azimuth_ap = AveragePrecision(_NUM_BINS)
        self._polar_ap = AveragePrecision(_NUM_BINS)
        self._add_ap = AveragePrecision(_NUM_BINS)
        self._adds_ap = AveragePrecision(_NUM_BINS)

    def evaluate(self, results, gts):
        """Evaluates a batch of final outputs."""
        labels, projs, cam_intrinsics, planes, views = [], [], [], [], []

        json_paths = gts['path_json']

        # Built the ground truth from the json files.
        for json_path in json_paths:
            anns = json.load(open(json_path))

            label = {'2d_instance': [], '3d_instance': [], 'scale_instance': [], 'Mo2c_instance': [], 'visibilities': []}

            for k in range(len(anns['objects'])):
                ann = anns['objects'][k]
                points_2d = np.array(ann['projected_cuboid'], dtype=np.float32)
                points_2d[:, 0] = points_2d[:, 0] / anns['camera_data']['width']
                points_2d[:, 1] = points_2d[:, 1] / anns['camera_data']['height']
                visibility = ann['visibility']

                label['2d_instance'].append(points_2d)
                label['3d_instance'].append(ann['keypoints_3d'])
                label['scale_instance'].append(ann['scale'])
                label['visibilities'].append(visibility)

                ori = rotation_util.from_quat(np.array(ann['quaternion_xyzw'])).as_matrix()
                trans = np.array(ann['location'])
                transformation = np.identity(4)
                transformation[:3, :3] = ori
                transformation[:3, 3] = trans

                label['Mo2c_instance'].append(transformation)

            label['2d_instance'] = np.array(label['2d_instance'])
            label['3d_instance'] = np.array(label['3d_instance'])
            label['scale_instance'] = np.array(label['scale_instance'])
            label['Mo2c_instance'] = np.array(label['Mo2c_instance'])
            label['visibilities'] = np.array(label['visibilities'])

            labels.append(label)

            proj = np.array(anns['camera_data']['camera_projection_matrix'])
            view = np.array(anns['camera_data']['camera_view_matrix'])
            cx = anns['camera_data']['intrinsics']['cx']
            cy = anns['camera_data']['intrinsics']['cy']
            fx = anns['camera_data']['intrinsics']['fx']
            fy = anns['camera_data']['intrinsics']['fy']
            cam_intrinsic = np.array([[fx, 0, cx], [0, fy, cy], [0, 0, 1]])

            projs.append(proj)
            views.append(view)
            cam_intrinsics.append(cam_intrinsic)

            center = np.asarray(anns['AR_data']['plane_center'])
            normal = np.asarray(anns['AR_data']['plane_normal'])
            plane = (center, normal)
            planes.append(plane)

        for boxes, label, plane, cam_intrinsic, projection_matrix, view in zip(results, labels, planes, cam_intrinsics, projs, views):

            # Extract the ground truth info.
            instances_scale = label['scale_instance']
            instances_2d = label['2d_instance']
            instances_3d = label['3d_instance']
            instances_Mo2c = label['Mo2c_instance']
            visibilities = label['visibilities']

            num_instances = 0
            for instance, instance_3d, visibility in zip(instances_2d, instances_3d, visibilities):
                if (visibility > self._vis_thresh and self._is_visible(instance[0]) and instance_3d[0, 2] < 0):
                    num_instances += 1

            # Skip negative examples in evaluation.
            if num_instances == 0:
                continue

            scale_hit_miss = HitMiss(self._scale_thresholds)
            iou_hit_miss = HitMiss(self._iou_thresholds)
            azimuth_hit_miss = HitMiss(self._azimuth_thresholds)
            polar_hit_miss = HitMiss(self._polar_thresholds)
            pixel_hit_miss = HitMiss(self._pixel_thresholds)
            add_hit_miss = HitMiss(self._add_thresholds)
            adds_hit_miss = HitMiss(self._adds_thresholds)

            num_matched = 0
            for idx_box in range(len(boxes['projected_points'])):

                # Correspond to one prediction in one image
                box_point_2d = boxes['projected_points'][idx_box]

                box_point_2d[:, 0] = box_point_2d[:, 0] / anns['camera_data']['width']
                box_point_2d[:, 1] = box_point_2d[:, 1] / anns['camera_data']['height']

                box_point_3d = boxes['point_3d_cam'][idx_box]
                relative_scale = boxes['obj_scale'][idx_box]
                score = boxes['score'][idx_box]

                index = self.match_box(box_point_2d, instances_2d, visibilities)

                if index >= 0:
                    num_matched += 1

                    # If you only compute the 3D bounding boxes from RGB images,
                    # your 3D keypoints may be upto scale. However the ground truth
                    # is at metric scale. There is a hack to re-scale your box using
                    # the ground planes (assuming your box is sitting on the ground).
                    # However many models learn to predict depths and scale correctly.
                    if not self.opt.use_absolute_scale:
                        scale = self.compute_scale(box_point_3d, plane)
                        box_point_3d = box_point_3d * scale

                    pixel_error = self.evaluate_2d(box_point_2d, instances_3d[index], instances_Mo2c[index], projection_matrix)
                    azimuth_error, polar_error, iou, add, adds = self.evaluate_3d(box_point_3d, instances_3d[index])
                    scale_error = self.evaluate_scale(relative_scale, instances_scale[index])
                    conf = score

                else:
                    conf = 0
                    pixel_error = _MAX_PIXEL_ERROR
                    azimuth_error = _MAX_AZIMUTH_ERROR
                    polar_error = _MAX_POLAR_ERROR
                    iou = 0.
                    add = _MAX_DISTANCE
                    adds = _MAX_DISTANCE
                    scale_error = _MAX_SCALE_ERROR

                scale_hit_miss.record_hit_miss([scale_error, conf], greater=False)
                iou_hit_miss.record_hit_miss([iou, conf])
                add_hit_miss.record_hit_miss([add, conf], greater=False)
                adds_hit_miss.record_hit_miss([adds, conf], greater=False)
                pixel_hit_miss.record_hit_miss([pixel_error, conf], greater=False)
                azimuth_hit_miss.record_hit_miss([azimuth_error, conf], greater=False)
                polar_hit_miss.record_hit_miss([polar_error, conf], greater=False)

            self._scale_ap.append(scale_hit_miss, len(instances_2d))
            self._iou_ap.append(iou_hit_miss, len(instances_2d))
            self._pixel_ap.append(pixel_hit_miss, len(instances_2d))
            self._azimuth_ap.append(azimuth_hit_miss, len(instances_2d))
            self._polar_ap.append(polar_hit_miss, len(instances_2d))
            self._add_ap.append(add_hit_miss, len(instances_2d))
            self._adds_ap.append(adds_hit_miss, len(instances_2d))
            self._matched += num_matched

    def evaluate_scale(self, relative_scale, instance):
        """Evalate the scale errors"""
        relative_scale_normalized = relative_scale / relative_scale[1]
        instance_normalized = instance / instance[1]

        error = np.sum(np.absolute(relative_scale_normalized - instance_normalized) / instance_normalized)
        self._error_scale += error
        return error

    def evaluate_2d(self, box, instance_3d, Mo2c, proj):
        """Evaluates a pair of 2D projections of 3D boxes.

        It computes the mean normalized distances of eight vertices of a box.

        Args:
          box: A 9*2 array of a predicted box.
          instance_2d: A 9*2 array of an annotated box.
          instance_3d: A 9*3 array of an annotated box.
          Mo2c: A gt transformation matrix from object frame to camera frame
          proj: Projection matrix

        Returns:
          Pixel error
        """
        Mc2o = np.linalg.inv(Mo2c)
        error_best = np.inf
        for id_symmetry in range(self.eval_config.eval_num_symmetry):
            theta = 2 * np.pi / self.eval_config.eval_num_symmetry
            M_R = rotation_y_matrix(theta * id_symmetry)
            M_trans = proj @ Mo2c @ M_R @ Mc2o
            instance_new = M_trans @ np.hstack((instance_3d, np.ones((instance_3d.shape[0], 1)))).T

            pp2 = (instance_new / instance_new[3])[:2]
            viewport_point = (pp2 + 1.0) / 2.0
            viewport_point[[0, 1]] = viewport_point[[1, 0]]
            instance_new = viewport_point.T
            error = np.mean(np.linalg.norm(box[1:] - instance_new[1:], axis=1))
            if error_best > error:
                error_best = error

        self._error_2d += error_best

        return error_best

    def _get_rotated_box(self, box_point_3d, angle):
        """Rotate a box along its vertical axis.
        Args:
          box: Input box.
          angle: Rotation angle in rad.
        Returns:
          A rotated box
        """
        CENTER = 0
        BACK_TOP_LEFT = 3
        BACK_BOTTOM_LEFT = 1
        up_vector = box_point_3d[BACK_TOP_LEFT] - box_point_3d[BACK_BOTTOM_LEFT]
        rot_vec = angle * up_vector / np.linalg.norm(up_vector)
        rotation = rotation_util.from_rotvec(rot_vec).as_matrix()
        box_center = box_point_3d[CENTER]
        box_point_3d_rotated = np.matmul((box_point_3d - box_center), rotation) + box_center
        return box_point_3d_rotated

    def evaluate_3d(self, box_point_3d, instance_3d):
        """Evaluates a box in 3D.

        It computes metrics of view angle and 3D IoU.

        Args:
          box: A predicted box.
          instance_3d: A 9*3 array of an annotated box, in metric level.
        Returns:
          The 3D IoU (float)
        """
        azimuth_error, polar_error = self.evaluate_viewpoint(box_point_3d, instance_3d)
        avg_distance, avg_sym_distance = self.compute_average_distance(box_point_3d, instance_3d)
        iou_best = 0
        avg_distance_best = _MAX_DISTANCE
        avg_sym_distance_best = _MAX_DISTANCE

        # Adapted from the official one: rotate the estimated one
        for theta in np.linspace(0, np.pi * 2, self.eval_config.eval_num_symmetry):

            box_point_3d_rotated = self._get_rotated_box(box_point_3d, theta)
            iou = self.evaluate_iou(box_point_3d_rotated, instance_3d)

            if iou > iou_best:
                azimuth_error, polar_error = self.evaluate_viewpoint(box_point_3d_rotated,
                                                                     instance_3d)
                avg_distance, avg_sym_distance = self.compute_average_distance(box_point_3d_rotated,
                                                                               instance_3d)
                iou_best = iou
                avg_distance_best = avg_distance
                avg_sym_distance_best = avg_sym_distance

        self._iou_3d += iou_best
        self._azimuth_error += azimuth_error
        self._polar_error += polar_error

        return azimuth_error, polar_error, iou_best, avg_distance_best, avg_sym_distance_best

    def compute_scale(self, box, plane):
        """Computes scale of the given box sitting on the plane."""
        center, normal = plane
        vertex_dots = [np.dot(vertex, normal) for vertex in box[1:]]
        vertex_dots = np.sort(vertex_dots)
        center_dot = np.dot(center, normal)
        scales = center_dot / vertex_dots[:4]

        return np.mean(scales)

    def compute_ray(self, box):
        """Computes a ray from camera to box centroid in box frame.

        For vertex in camera frame V^c, and object unit frame V^o, we have
          R * Vc + T = S * Vo,
        where S is a 3*3 diagonal matrix, which scales the unit box to its real size.

        In fact, the camera coordinates we get have scale ambiguity. That is, we have
          Vc' = 1/beta * Vc, and S' = 1/beta * S
        where beta is unknown. Since all box vertices should have negative Z values,
        we can assume beta is always positive.

        To update the equation,
          R * beta * Vc' + T = beta * S' * Vo.

        To simplify,
          R * Vc' + T' = S' * Vo,
        where Vc', S', and Vo are known. The problem is to compute
          T' = 1/beta * T,
        which is a point with scale ambiguity. It forms a ray from camera to the
        centroid of the box.

        By using homogeneous coordinates, we have
          M * Vc'_h = (S' * Vo)_h,
        where M = [R|T'] is a 4*4 transformation matrix.

        To solve M, we have
          M = ((S' * Vo)_h * Vc'_h^T) * (Vc'_h * Vc'_h^T)_inv.
        And T' = M[:3, 3:].

        Args:
          box: A 9*3 array of a 3D bounding box.

        Returns:
          A ray represented as [x, y, z].
        """
        size_x = np.linalg.norm(box[5] - box[1])
        size_y = np.linalg.norm(box[3] - box[1])
        size_z = np.linalg.norm(box[2] - box[1])
        size = np.asarray([size_x, size_y, size_z])
        box_o = Box.UNIT_BOX * size
        box_oh = np.ones((4, 9))
        box_oh[:3] = np.transpose(box_o)

        box_ch = np.ones((4, 9))
        box_ch[:3] = np.transpose(box)
        box_cht = np.transpose(box_ch)

        box_oct = np.matmul(box_oh, box_cht)
        try:
            box_cct_inv = np.linalg.inv(np.matmul(box_ch, box_cht))
        except:  # noqa: E722
            box_cct_inv = np.linalg.pinv(np.matmul(box_ch, box_cht))

        transform = np.matmul(box_oct, box_cct_inv)
        return transform[:3, 3:].reshape((3))

    def compute_average_distance(self, box, instance):
        """Computes Average Distance (ADD) metric."""
        add_distance = 0.
        for i in range(Box.NUM_KEYPOINTS):
            delta = np.linalg.norm(box[i, :] - instance[i, :])
            add_distance += delta
        add_distance /= Box.NUM_KEYPOINTS

        # Computes the symmetric version of the average distance metric.
        # From PoseCNN https://arxiv.org/abs/1711.00199
        # For each keypoint in predicttion, search for the point in ground truth
        # that minimizes the distance between the two.
        add_sym_distance = 0.
        for i in range(Box.NUM_KEYPOINTS):
            # Find nearest vertex in instance
            distance = np.linalg.norm(box[i, :] - instance[0, :])
            for j in range(Box.NUM_KEYPOINTS):
                d = np.linalg.norm(box[i, :] - instance[j, :])
                if d < distance:
                    distance = d
            add_sym_distance += distance
        add_sym_distance /= Box.NUM_KEYPOINTS

        return add_distance, add_sym_distance

    def compute_viewpoint(self, box):
        """Computes viewpoint of a 3D bounding box.

        We use the definition of polar angles in spherical coordinates
        (http://mathworld.wolfram.com/PolarAngle.html), expect that the
        frame is rotated such that Y-axis is up, and Z-axis is out of screen.

        Args:
          box: A 9*3 array of a 3D bounding box.

        Returns:
          Two polar angles (azimuth and elevation) in degrees. The range is between
          -180 and 180.
        """
        x, y, z = self.compute_ray(box)
        theta = math.degrees(math.atan2(z, x))
        phi = math.degrees(math.atan2(y, math.hypot(x, z)))
        return theta, phi

    def evaluate_viewpoint(self, box, instance):
        """Evaluates a 3D box by viewpoint.

        Args:
          box: A 9*3 array of a predicted box.
          instance: A 9*3 array of an annotated box, in metric level.
        Returns:
          Two viewpoint angle errors.
        """
        predicted_azimuth, predicted_polar = self.compute_viewpoint(box)
        gt_azimuth, gt_polar = self.compute_viewpoint(instance)

        polar_error = abs(predicted_polar - gt_polar)
        # Azimuth is from (-180,180) and a spherical angle so angles -180 and 180
        # are equal. E.g. the azimuth error for -179 and 180 degrees is 1'.
        # azimuth_error = abs(predicted_azimuth - gt_azimuth)

        azimuth_error = abs(predicted_azimuth - gt_azimuth) % (360 / self.eval_config.eval_num_symmetry)

        if azimuth_error > 180:
            azimuth_error = 360 - azimuth_error

        return azimuth_error, polar_error

    def evaluate_rotation(self, box, instance):
        """Evaluates rotation of a 3D box.

        1. The L2 norm of rotation angles
        2. The rotation angle computed from rotation matrices
              trace(R_1^T R_2) = 1 + 2 cos(theta)
              theta = arccos((trace(R_1^T R_2) - 1) / 2)

        3. The rotation angle computed from quaternions. Similar to the above,
           except instead of computing the trace, we compute the dot product of two
           quaternion.
             theta = 2 * arccos(| p.q |)
           Note the distance between quaternions is not the same as distance between
           rotations.

        4. Rotation distance from "3D Bounding box estimation using deep learning
           and geometry""
               d(R1, R2) = || log(R_1^T R_2) ||_F / sqrt(2)

        Args:
          box: A 9*3 array of a predicted box.
          instance: A 9*3 array of an annotated box, in metric level.

        Returns:
          Magnitude of the rotation angle difference between the box and instance.
        """
        prediction = Box.Box(box)
        annotation = Box.Box(instance)
        gt_rotation_inverse = np.linalg.inv(annotation.rotation)
        rotation_error = np.matmul(prediction.rotation, gt_rotation_inverse)

        error_angles = np.array(
            rotation_util.from_dcm(rotation_error).as_euler('zxy'))
        abs_error_angles = np.absolute(error_angles)
        abs_error_angles = np.minimum(
            abs_error_angles, np.absolute(math.pi * np.ones(3) - abs_error_angles))
        error = np.linalg.norm(abs_error_angles)

        # Compute the error as the angle between the two rotation
        rotation_error_trace = abs(np.matrix.trace(rotation_error))
        angular_distance = math.acos((rotation_error_trace - 1.) / 2.)

        # angle = 2 * acos(|q1.q2|)
        box_quat = np.array(rotation_util.from_dcm(prediction.rotation).as_quat())
        gt_quat = np.array(rotation_util.from_dcm(annotation.rotation).as_quat())
        quat_distance = 2 * math.acos(np.dot(box_quat, gt_quat))

        # The rotation measure from "3D Bounding box estimation using deep learning and geometry"
        rotation_error_log = scipy.linalg.logm(rotation_error)
        rotation_error_frob_norm = np.linalg.norm(rotation_error_log, ord='fro')
        rotation_distance = rotation_error_frob_norm / 1.4142

        return (error, quat_distance, angular_distance, rotation_distance)

    def evaluate_iou(self, box, instance):
        """Evaluates a 3D box by 3D IoU.

        It computes 3D IoU of predicted and annotated boxes.

        Args:
          box: A 9*3 array of a predicted box.
          instance: A 9*3 array of an annotated box, in metric level.

        Returns:
          3D Intersection over Union (float)
        """
        # Computes 3D IoU of the two boxes.
        prediction = Box.Box(box)
        annotation = Box.Box(instance)
        iou = Box.IoU(prediction, annotation)
        try:
            iou_result = iou.iou()
        except:  # noqa: E722
            iou_result = 0
        return iou_result

    def match_box(self, box, instances, visibilities):
        """Matches a detected box with annotated instances.

        For a predicted box, finds the nearest annotation in instances. This means
        we always assume a match for a prediction. If the nearest annotation is
        below the visibility threshold, the match can be skipped.

        Args:
          box: A 9*2 array of a predicted box.
          instances: A ?*9*2 array of annotated instances. Each instance is a 9*2
            array.
          visibilities: An array of the visibilities of the instances.

        Returns:
          Index of the matched instance; otherwise -1.
        """
        norms = np.linalg.norm(instances[:, 1:, :] - box[1:, :], axis=(1, 2))
        i_min = np.argmin(norms)
        if visibilities[i_min] < self._vis_thresh:
            return -1
        return i_min

    def write_report(self, report_file=None):
        """Writes a report of the evaluation."""
        def report_array(label, array):
            """Write the result array to the report"""
            built_array = [label]
            for val in array:
                built_array.append('{:.4f}'.format(val))
            return built_array

        if report_file is None:
            report_file = os.path.join(self.eval_config.results_dir, 'report.txt')

        # Write the table
        table = [["Mean Error Scale:", safe_divide(self._error_scale, self._matched)],
                 ["Mean Error 2D:", safe_divide(self._error_2d, self._matched)],
                 ["Mean 3D IoU:", safe_divide(self._iou_3d, self._matched)],
                 ["Mean Azimuth Error:", safe_divide(self._azimuth_error, self._matched)],
                 ["Mean Polar Error:", safe_divide(self._polar_error, self._matched)],
                 []
                 ]

        # Scale average precision
        thresh = ["Scale Thresh:"]
        for threshold in self._scale_thresholds:
            thresh.append("{:.4f}".format(threshold))
        table.append(thresh)
        table.append(report_array('AP @Scale:', self._scale_ap.aps))
        table.append([])

        # 3D IoU average precision
        thresh = ["IoU Thresholds:"]
        for threshold in self._iou_thresholds:
            thresh.append("{:.4f}".format(threshold))
        table.append(thresh)
        table.append(report_array('AP @3D IoU:', self._iou_ap.aps))
        table.append([])

        # 2D pixel average precision
        thresh = ["2D Thresholds:"]
        for threshold in self._pixel_thresholds:
            thresh.append("{:.4f}".format(threshold))
        table.append(thresh)
        table.append(report_array('AP @2D Pixel:', self._pixel_ap.aps))
        table.append([])

        # Azimuth average precision
        thresh = ["Azimuth Thresh:"]
        for threshold in self._azimuth_thresholds:
            thresh.append("{:.4f}".format(threshold * 0.1))
        table.append(thresh)
        table.append(report_array('AP @Azimuth:', self._azimuth_ap.aps))
        table.append([])

        # Polar average precision
        thresh = ["Polar Thresh:"]
        for threshold in self._polar_thresholds:
            thresh.append("{:.4f}".format(threshold * 0.1))
        table.append(thresh)
        table.append(report_array('AP @Polar:', self._polar_ap.aps))
        table.append([])

        # ADD average precision
        thresh = ["ADD Thresh:"]
        for threshold in self._add_thresholds:
            thresh.append("{:.4f}".format(threshold))
        table.append(thresh)
        table.append(report_array('AP @ADD:', self._add_ap.aps))
        table.append([])

        # ADDS average precision
        thresh = ["ADDS Thresh:"]
        for threshold in self._adds_thresholds:
            thresh.append("{:.4f}".format(threshold))
        table.append(thresh)
        table.append(report_array('AP @ADDS:', self._adds_ap.aps))
        table.append([])

        # Write the report
        open(report_file, 'w', encoding='utf-8').write(tabulate(table, tablefmt="psql"))

    def get_accuracy(self):
        """Return 3D IoU accuracy and 2D MPE (mean pixel error)"""
        return self._iou_ap.aps[10], safe_divide(self._error_2d, self._matched)

    def reset(self):
        """Reset all the precision related parameters."""
        # Reset AP stuffs
        self._scale_ap = AveragePrecision(_NUM_BINS)
        self._iou_ap = AveragePrecision(_NUM_BINS)
        self._pixel_ap = AveragePrecision(_NUM_BINS)
        self._azimuth_ap = AveragePrecision(_NUM_BINS)
        self._polar_ap = AveragePrecision(_NUM_BINS)
        self._add_ap = AveragePrecision(_NUM_BINS)
        self._adds_ap = AveragePrecision(_NUM_BINS)

        # Reset mean related
        self._error_scale = 0.
        self._error_2d = 0.
        self._matched = 0
        self._iou_3d = 0.
        self._azimuth_error = 0.
        self._polar_error = 0.

    def finalize(self):
        """Computes average precision curves."""
        self._scale_ap.compute_ap_curve()
        self._iou_ap.compute_ap_curve()
        self._pixel_ap.compute_ap_curve()
        self._azimuth_ap.compute_ap_curve()
        self._polar_ap.compute_ap_curve()
        self._add_ap.compute_ap_curve()
        self._adds_ap.compute_ap_curve()

    def _is_visible(self, point):
        """Determines if a 2D point is visible."""
        return point[0] > 0 and point[0] < 1 and point[1] > 0 and point[1] < 1
