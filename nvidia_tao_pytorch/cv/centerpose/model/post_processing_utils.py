# SPDX-FileCopyrightText: Copyright (c) 2023 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

""" CenterPose Post processing utils for inference. """

import os
import torch
import numpy as np
import cv2
import logging
import json
from enum import IntEnum
from pyrr import Quaternion
from scipy.spatial.transform import Rotation as R

logger = logging.getLogger(__name__)


class CuboidVertexType(IntEnum):
    """This class contains a 3D cuboid vertex type"""

    FrontTopRight = 0
    FrontTopLeft = 1
    FrontBottomLeft = 2
    FrontBottomRight = 3
    RearTopRight = 4
    RearTopLeft = 5
    RearBottomLeft = 6
    RearBottomRight = 7
    Center = 8
    TotalCornerVertexCount = 8
    TotalVertexCount = 9


class Cuboid3d():
    """This class initialize a 3D cuboid according to the scale."""

    def __init__(self, size3d=[1.0, 1.0, 1.0],
                 coord_system=None, parent_object=None):
        """This local coordinate system is similar to the intrinsic transform matrix of a 3d object.
        Create a box with a certain size.
        """
        self.center_location = [0, 0, 0]
        self.coord_system = coord_system
        self.size3d = size3d
        self._vertices = [0, 0, 0] * CuboidVertexType.TotalCornerVertexCount
        self.generate_vertexes()

    def get_vertex(self, vertex_type):
        """Returns the location of a vertex.

        Args:
            vertex_type: enum of type CuboidVertexType

        Returns:
            Numpy array(3) - Location of the vertex type in the cuboid
        """
        return self._vertices[vertex_type]

    def get_vertices(self):
        """Return the 3D cuboid vertices"""
        return self._vertices

    def generate_vertexes(self):
        """Generate the 3D cuboid vertices"""
        width, height, depth = self.size3d

        # By default use the normal OpenCV coordinate system
        if (self.coord_system is None):
            cx, cy, cz = self.center_location
            # X axis point to the right
            right = cx + width / 2.0
            left = cx - width / 2.0
            # Y axis point upward
            top = cy + height / 2.0
            bottom = cy - height / 2.0
            # Z axis point forward
            front = cz + depth / 2.0
            rear = cz - depth / 2.0

            # List of 8 vertices of the box
            self._vertices = [
                # self.center_location,   # Center
                [left, bottom, rear],  # Rear Bottom Left
                [left, bottom, front],  # Front Bottom Left
                [left, top, rear],  # Rear Top Left
                [left, top, front],  # Front Top Left

                [right, bottom, rear],  # Rear Bottom Right
                [right, bottom, front],  # Front Bottom Right
                [right, top, rear],  # Rear Top Right
                [right, top, front],  # Front Top Right

            ]


class CuboidPNPSolver(object):
    """
    This class is used to find the 6-DoF pose of a cuboid given its projected vertices.

    Runs perspective-n-point (PNP) algorithm.
    """

    # Class variables
    cv2version = cv2.__version__.split('.')
    cv2majorversion = int(cv2version[0])

    def __init__(self, scaling_factor=1,
                 camera_intrinsic_matrix=None,
                 cuboid3d=None,
                 dist_coeffs=np.zeros((4, 1)),
                 min_required_points=4
                 ):
        """Initialize the 3D cuboid and camera parameters"""
        self.min_required_points = max(4, min_required_points)
        self.scaling_factor = scaling_factor

        if camera_intrinsic_matrix is not None:
            self._camera_intrinsic_matrix = camera_intrinsic_matrix
        else:
            self._camera_intrinsic_matrix = np.array([
                [0, 0, 0],
                [0, 0, 0],
                [0, 0, 0]
            ])
        self._cuboid3d = cuboid3d

        self._dist_coeffs = dist_coeffs

    def set_camera_intrinsic_matrix(self, new_intrinsic_matrix):
        """Sets the camera intrinsic matrix"""
        self._camera_intrinsic_matrix = new_intrinsic_matrix

    def set_dist_coeffs(self, dist_coeffs):
        """Sets the camera intrinsic matrix"""
        self._dist_coeffs = dist_coeffs

    def solve_pnp(self, cuboid2d_points, pnp_algorithm=cv2.SOLVEPNP_ITERATIVE, opencv_return=True):
        """
        Detects the rotation and traslation
        of a cuboid object from its vertexes'
        2D location in the image

        Inputs:
        - cuboid2d_points:  list of XY tuples
        - pnp_algorithm: algorithm of the Perspective-n-Point (PnP) pose computation
        - opencv_return: if ture, return the OpenCV coordinate; else, return the OpenGL coordinate.
        OpenCV coordiate is used to demo the visualization results and the OpenGL coordinate is used to calculate the 3D IoU.

        Outputs:
        - location in 3D
        - pose in 3D (as quaternion)
        - projected points:  np.ndarray of np.ndarrays

        """
        location = None
        quaternion = None
        location_new = None
        quaternion_new = None
        loc = None
        quat = None
        reprojectionError = None
        projected_points = cuboid2d_points
        cuboid3d_points = np.array(self._cuboid3d.get_vertices())

        obj_2d_points = []
        obj_3d_points = []

        # 8*n points
        for i in range(len(cuboid2d_points)):
            check_point_2d = cuboid2d_points[i]
            # Ignore invalid points
            if (check_point_2d is None or check_point_2d[0] < -5000 or check_point_2d[1] < -5000):
                continue
            obj_2d_points.append(check_point_2d)
            obj_3d_points.append(cuboid3d_points[int(i // (len(cuboid2d_points) / CuboidVertexType.TotalCornerVertexCount))])

        obj_2d_points = np.array(obj_2d_points, dtype=float)
        obj_3d_points = np.array(obj_3d_points, dtype=float)
        valid_point_count = len(obj_2d_points)

        # Can only do PNP if we have more than 3 valid points
        is_points_valid = valid_point_count >= self.min_required_points

        if is_points_valid:

            # Heatmap representation may have less than 6 points, in which case we have to use another pnp algorithm
            if valid_point_count < 6:
                pnp_algorithm = cv2.SOLVEPNP_EPNP

            # Usually, we use this one
            ret, rvec, tvec, reprojectionError = cv2.solvePnPGeneric(
                obj_3d_points,
                obj_2d_points,
                self._camera_intrinsic_matrix,
                self._dist_coeffs,
                flags=pnp_algorithm
            )

            if ret:

                rvec = np.array(rvec[0])
                tvec = np.array(tvec[0])

                reprojectionError = reprojectionError.flatten()[0]

                # Convert OpenCV coordinate system to OpenGL coordinate system
                transformation = np.identity(4)
                r = R.from_rotvec(rvec.reshape(1, 3))
                transformation[:3, :3] = r.as_matrix()
                transformation[:3, 3] = tvec.reshape(1, 3)
                M = np.zeros((4, 4))
                M[0, 1] = 1
                M[1, 0] = 1
                M[3, 3] = 1
                M[2, 2] = -1
                transformation = np.matmul(M, transformation)

                rvec_new = R.from_matrix(transformation[:3, :3]).as_rotvec()
                tvec_new = transformation[:3, 3]

                # OpenGL result, to be compared against GT
                location_new = list(x for x in tvec_new)
                quaternion_new = list(self.convert_rvec_to_quaternion(rvec_new))

                # OpenCV result
                location = list(x[0] for x in tvec)
                quaternion = list(self.convert_rvec_to_quaternion(rvec))

                # Still use OpenCV way to project 3D points
                projected_points, _ = cv2.projectPoints(cuboid3d_points, rvec, tvec, self._camera_intrinsic_matrix,
                                                        self._dist_coeffs)

                projected_points = np.squeeze(projected_points)

                # Currently, we assume pnp fails if z<0
                _, _, z = location
                if z < 0:
                    location = None
                    quaternion = None
                    location_new = None
                    quaternion_new = None

                    logger.debug("PNP solution is behind the camera (Z < 0) => Fail")
                else:
                    logger.debug("solvePNP found good results - location: {} - rotation: {} !!!".format(location, quaternion))
            else:
                logger.debug('solvePnP return false')
        else:
            logger.debug("Need at least 4 valid points in order to run PNP. Currently: {}".format(valid_point_count))

        if opencv_return:
            # Return OpenCV result for demo
            loc = location
            quat = quaternion
        else:
            # Return OpenGL result for eval
            loc = location_new
            quat = quaternion_new
        return loc, quat, projected_points, reprojectionError

    def convert_rvec_to_quaternion(self, rvec):
        """Convert rvec (which is log quaternion) to quaternion"""
        theta = np.sqrt(rvec[0] * rvec[0] + rvec[1] * rvec[1] + rvec[2] * rvec[2])  # in radians
        raxis = [rvec[0] / theta, rvec[1] / theta, rvec[2] / theta]

        # pyrr's Quaternion (order is XYZW), https://pyrr.readthedocs.io/en/latest/oo_api_quaternion.html
        return Quaternion.from_axis_rotation(raxis, theta)

        # Alternatively: pyquaternion
        # return Quaternion(axis=raxis, radians=theta)  # uses OpenCV's Quaternion (order is WXYZ)


def _gather_feat(feat, ind, mask=None):
    """Gather the feature maps according to the index"""
    if len(ind.size()) > 2:

        num_symmetry = ind.size(1)
        dim = feat.size(2)
        ind = ind.unsqueeze(3).expand(ind.size(0), ind.size(1), ind.size(2),
                                      dim)  # batch x num_symmetry x max_object x (num_joint x 2)

        ind = ind.view(ind.size(0), -1, ind.size(3))  # batch x (num_symmetry x max_object) x (num_joint x 2)

        feat = feat.gather(1, ind)
        feat = feat.view(ind.size(0), num_symmetry, -1,
                         ind.size(2))  # batch x num_symmetry x max_object x (num_joint x 2)
        if mask is not None:
            mask = mask.unsqueeze(3).expand_as(feat)
            feat = feat[mask]
            feat = feat.view(-1, dim)
    else:
        dim = feat.size(2)
        ind = ind.unsqueeze(2).expand(ind.size(0), ind.size(1), dim)
        feat = feat.gather(1, ind)

        if mask is not None:
            mask = mask.unsqueeze(2).expand_as(feat)
            feat = feat[mask]
            feat = feat.view(-1, dim)
    return feat


def _transpose_and_gather_feat(feat, ind):
    """Transpose and gather the features"""
    feat = feat.permute(0, 2, 3, 1).contiguous()
    feat = feat.view(feat.size(0), -1, feat.size(3))
    feat = _gather_feat(feat, ind)
    return feat


def _nms(heat, kernel=3):
    """Non-maximum Suppression for the heatmap using max pooling layer"""
    pad = (kernel - 1) // 2
    hmax = torch.nn.functional.max_pool2d(
        heat, (kernel, kernel), stride=1, padding=pad)
    keep = (hmax == heat).float()
    return heat * keep


def _topk(scores, K=40):
    """Select the top-k locations in the heatmap"""
    batch, cat, height, width = scores.size()

    topk_scores, topk_inds = torch.topk(scores.view(batch, cat, -1), K)

    topk_inds = topk_inds % (height * width)
    topk_ys = (topk_inds / width).int().float()
    topk_xs = (topk_inds % width).int().float()

    topk_score, topk_ind = torch.topk(topk_scores.view(batch, -1), K)
    topk_clses = (topk_ind / K).int()
    topk_inds = _gather_feat(
        topk_inds.view(batch, -1, 1), topk_ind).view(batch, K)
    topk_ys = _gather_feat(topk_ys.view(batch, -1, 1), topk_ind).view(batch, K)
    topk_xs = _gather_feat(topk_xs.view(batch, -1, 1), topk_ind).view(batch, K)

    return topk_score, topk_inds, topk_clses, topk_ys, topk_xs


def _topk_channel(scores, K=40):
    """Select the top K accourding the scores"""
    batch, cat, height, width = scores.size()

    topk_scores, topk_inds = torch.topk(scores.view(batch, cat, -1), K)

    topk_inds = topk_inds % (height * width)
    topk_ys = (topk_inds / width).int().float()
    topk_xs = (topk_inds % width).int().float()

    return topk_scores, topk_inds, topk_ys, topk_xs


def get_3rd_point(a, b):
    """Get the 3rd points according to the two input points"""
    direct = a - b
    return b + np.array([-direct[1], direct[0]], dtype=np.float32)


def get_dir(src_point, rot_rad):
    """Get the direction according to the rotation parameter"""
    sn, cs = np.sin(rot_rad), np.cos(rot_rad)
    src_result = [0, 0]
    src_result[0] = src_point[0] * cs - src_point[1] * sn
    src_result[1] = src_point[0] * sn + src_point[1] * cs
    return src_result


def get_affine_transform(center,
                         scale,
                         rot,
                         output_size,
                         shift=np.array([0, 0], dtype=np.float32),
                         inv=0):
    """Fetch the transform matrix from target image and original image"""
    if not isinstance(scale, np.ndarray) and not isinstance(scale, list):
        scale = np.array([scale, scale], dtype=np.float32)

    scale_tmp = scale
    src_w = scale_tmp[0]
    dst_w = output_size[0]
    dst_h = output_size[1]

    rot_rad = np.pi * rot / 180
    src_dir = get_dir([0, src_w * -0.5], rot_rad)
    dst_dir = np.array([0, dst_w * -0.5], np.float32)

    src = np.zeros((3, 2), dtype=np.float32)
    dst = np.zeros((3, 2), dtype=np.float32)
    src[0, :] = center + scale_tmp * shift
    src[1, :] = center + src_dir + scale_tmp * shift
    dst[0, :] = [dst_w * 0.5, dst_h * 0.5]
    dst[1, :] = np.array([dst_w * 0.5, dst_h * 0.5], np.float32) + dst_dir

    src[2:, :] = get_3rd_point(src[0, :], src[1, :])
    dst[2:, :] = get_3rd_point(dst[0, :], dst[1, :])

    if inv:
        trans = cv2.getAffineTransform(np.float32(dst), np.float32(src))
    else:
        trans = cv2.getAffineTransform(np.float32(src), np.float32(dst))

    return trans


def affine_transform(pt, t):
    """Affine transformation"""
    new_pt = np.array([pt[0], pt[1], 1.], dtype=np.float32).T
    new_pt = np.dot(t, new_pt)
    return new_pt[:2]


def transform_preds(coords, center, scale, output_size):
    """Affine transformation through the transform matrix"""
    target_coords = np.zeros(coords.shape)
    trans = get_affine_transform(center, scale, 0, output_size, inv=1)
    for p in range(coords.shape[0]):
        if coords[p, 0] == -10000 and coords[p, 1] == -10000:
            # Still give it a very small number
            target_coords[p, 0:2] = [-10000, -10000]
        else:
            target_coords[p, 0:2] = affine_transform(coords[p, 0:2], trans)
    return target_coords


def soft_nms(src_boxes, sigma=0.5, Nt=0.3, threshold=0.001, method=0):
    """Soft non-maximum Suppression for bounding boxes"""
    N = src_boxes.shape[0]
    pos = 0
    maxscore = 0
    maxpos = 0

    for i in range(N):
        maxscore = src_boxes[i]['score']
        maxpos = i

        tx1 = src_boxes[i]['bbox'][0]
        ty1 = src_boxes[i]['bbox'][1]
        tx2 = src_boxes[i]['bbox'][2]
        ty2 = src_boxes[i]['bbox'][3]
        ts = src_boxes[i]['score']

        pos = i + 1
        # get max box
        while pos < N:
            if maxscore < src_boxes[pos]['score']:
                maxscore = src_boxes[pos]['score']
                maxpos = pos
            pos = pos + 1

        # add max box as a detection
        src_boxes[i]['bbox'] = src_boxes[maxpos]['bbox']
        src_boxes[i]['score'] = src_boxes[maxpos]['score']

        # swap ith box with position of max box
        src_boxes[maxpos]['bbox'] = [tx1, ty1, tx2, ty2]
        src_boxes[maxpos]['score'] = ts

        for key in src_boxes[0]:
            if key not in ('bbox', 'score'):
                tmp = src_boxes[i][key]
                src_boxes[i][key] = src_boxes[maxpos][key]
                src_boxes[maxpos][key] = tmp

        tx1 = src_boxes[i]['bbox'][0]
        ty1 = src_boxes[i]['bbox'][1]
        tx2 = src_boxes[i]['bbox'][2]
        ty2 = src_boxes[i]['bbox'][3]
        ts = src_boxes[i]['score']

        pos = i + 1
        # NMS iterations, note that N changes if detection boxes fall below threshold
        while pos < N:

            x1 = src_boxes[pos]['bbox'][0]
            y1 = src_boxes[pos]['bbox'][1]
            x2 = src_boxes[pos]['bbox'][2]
            y2 = src_boxes[pos]['bbox'][3]

            area = (x2 - x1 + 1) * (y2 - y1 + 1)
            iw = (min(tx2, x2) - max(tx1, x1) + 1)
            if iw > 0:
                ih = (min(ty2, y2) - max(ty1, y1) + 1)
                if ih > 0:
                    ua = float((tx2 - tx1 + 1) * (ty2 - ty1 + 1) + area - iw * ih)
                    ov = iw * ih / ua  # iou between max box and detection box

                    if method == 1:  # linear
                        if ov > Nt:
                            weight = 1 - ov
                        else:
                            weight = 1
                    elif method == 2:  # gaussian
                        weight = np.exp(-(ov * ov) / sigma)
                    else:  # original NMS
                        if ov > Nt:
                            weight = 0
                        else:
                            weight = 1

                    src_boxes[pos]['score'] = weight * src_boxes[pos]['score']

                    # if box score falls below threshold, discard the box by swapping with last box
                    # update N
                    if src_boxes[pos]['score'] < threshold:

                        src_boxes[pos]['bbox'] = src_boxes[N - 1]['bbox']
                        src_boxes[pos]['score'] = src_boxes[N - 1]['score']

                        for key in src_boxes[0]:
                            if key not in ('bbox', 'score'):
                                tmp = src_boxes[pos][key]
                                src_boxes[pos][key] = src_boxes[N - 1][key]
                                src_boxes[N - 1][key] = tmp

                        N = N - 1
                        pos = pos - 1

            pos = pos + 1

    keep = [i for i in range(N)]
    return keep


def pnp_shell(points_filtered, scale, cam_intrinsic, opencv_return=True):
    """Initialize a 3D cuboid and process the PnP calcualation to get the 2D/3D keypoints"""
    # Initial a 3d cuboid
    cuboid3d = Cuboid3d(1 * np.array(scale))

    pnp_solver = \
        CuboidPNPSolver(
            cuboid3d=cuboid3d
        )
    pnp_solver.set_camera_intrinsic_matrix(cam_intrinsic)

    # Process the 3D cuboid, 2D keypoints and intrinsic matrix to solve the pnp
    location, quaternion, projected_points, _ = pnp_solver.solve_pnp(
        points_filtered, opencv_return=opencv_return)

    # Calculate the actual 3D keypoints by using the location and quaternion from pnp solver
    if location is not None:

        ori = R.from_quat(quaternion).as_matrix()
        pose_pred = np.identity(4)
        pose_pred[:3, :3] = ori
        pose_pred[:3, 3] = location
        point_3d_obj = cuboid3d.get_vertices()

        point_3d_cam = pose_pred @ np.hstack(
            (np.array(point_3d_obj), np.ones((np.array(point_3d_obj).shape[0], 1)))).T

        point_3d_cam = point_3d_cam[:3, :].T

        # Add the centroid
        point_3d_cam = np.insert(point_3d_cam, 0, np.mean(point_3d_cam, axis=0), axis=0)

        # Add the center
        projected_points = np.insert(projected_points, 0, np.mean(projected_points, axis=0), axis=0)

        return projected_points, point_3d_cam, location, quaternion

    return None


def add_obj_order(img, keypoints2d):
    """Draw the 2D keypoints on the image"""
    bbox = np.array(keypoints2d, dtype=np.int32)
    font = cv2.FONT_HERSHEY_SIMPLEX

    for i in range(len(bbox)):
        txt = '{:d}'.format(i)
        cat_size = cv2.getTextSize(txt, font, 1, 2)[0]
        cv2.putText(img, txt, (bbox[i][0], bbox[i][1] + cat_size[1]),
                    font, 1, (0, 255, 0), thickness=2, lineType=cv2.LINE_AA)


def add_axes(img, box, cam_intrinsic):
    """Draw the 6-DoF on the image"""
    # box 9x3 array
    # OpenCV way
    N = 0.5
    # Centroid, top, front, right
    axes_point_list = [0, box[3] - box[1], box[2] - box[1], box[5] - box[1]]

    viewport_point_list = []
    for axes_point in axes_point_list:
        vector = axes_point
        vector = vector / np.linalg.norm(vector) * N if np.linalg.norm(vector) != 0 else 0
        vector = vector + box[0]
        vector = vector.flatten()

        k_3d = np.array([vector[0], vector[1], vector[2]])
        pp = np.matmul(cam_intrinsic, k_3d.reshape(3, 1))
        viewport_point = [pp[0] / pp[2], pp[1] / pp[2]]
        viewport_point_list.append((int(viewport_point[0]), int(viewport_point[1])))

    # BGR space
    cv2.line(img, viewport_point_list[0], viewport_point_list[1], (0, 255, 0), 5)  # y-> green
    cv2.line(img, viewport_point_list[0], viewport_point_list[2], (255, 0, 0), 5)  # z-> blue
    cv2.line(img, viewport_point_list[0], viewport_point_list[3], (0, 0, 255), 5)  # x-> red


def add_coco_hp(img, points):
    """Draw the 3D Bounding Box on the image"""
    # objectron
    edges = [[2, 4], [2, 6], [6, 8], [4, 8],
             [1, 2], [3, 4], [5, 6], [7, 8],
             [1, 3], [1, 5], [3, 7], [5, 7]]

    num_joints = 8
    points = np.array(points, dtype=np.int32).reshape(num_joints, 2)
    # Draw edges
    for e in edges:
        temp = [e[0] - 1, e[1] - 1]
        edge_color = (0, 0, 255)  # bgr
        if points[temp[1], 0] <= -10000 or points[temp[1], 1] <= -10000 or points[temp[0], 0] <= -10000 or \
                points[temp[0], 1] <= -10000:
            continue
        else:
            cv2.line(img, (points[temp[0], 0], points[temp[0], 1]),
                          (points[temp[1], 0], points[temp[1], 1]), edge_color, 2)


def add_obj_scale(img, bbox, scale):
    """Draw the relative dimension numbers in a small region"""
    bbox = np.array(bbox, dtype=np.int32)
    font = cv2.FONT_HERSHEY_SIMPLEX
    txt = '{:.3f}/{:.3f}/{:.3f}'.format(scale[0], scale[1], scale[2])
    cat_size = cv2.getTextSize(txt, font, 0.5, 2)[0]
    cv2.rectangle(img,
                  (bbox[0], bbox[1] + 2),
                  (bbox[0] + cat_size[0], bbox[1] + cat_size[1] + 6), (0, 0, 0), -1)
    cv2.putText(img, txt, (bbox[0], bbox[1] + cat_size[1]),
                font, 0.5, (255, 255, 255), thickness=1, lineType=cv2.LINE_AA)


def save_inference_prediction(outputs, output_dir, batch, infer_config):
    """Save the visualization results to the required folder"""
    # Camera Intrinsic matrix
    cx = infer_config['principle_point_x']
    cy = infer_config['principle_point_y']
    fx = infer_config['focal_length_x']
    fy = infer_config['focal_length_y']
    skew = infer_config['skew']
    cam_intrinsic = np.array([[fx, skew, cx], [0, fy, cy], [0, 0, 1]])

    save_json = infer_config['save_json']
    save_visualization = infer_config['save_visualization']

    for idx in range(len(batch['input'])):
        out = outputs[idx]
        img = cv2.imread(batch['img_path'][idx])
        _, tail = os.path.split(batch['img_path'][idx])

        output_image_name = os.path.join(output_dir, tail)
        output_json_name = os.path.join(output_dir, os.path.splitext(tail)[0] + '.json')

        dict_results = {'image_name': tail, "objects": []}
        for k in range(len(out['projected_points'])):

            projected_points = out['projected_points'][k]
            point_3d_cam = out['point_3d_cam'][k]
            bbox = out['bbox'][k]
            obj_scale = out['obj_scale'][k]
            obj_translations = out['location'][k]
            obj_rotations = out['quaternion'][k]
            keypoints_2d = out['keypoints_2d'][k]

            if save_json:
                if obj_translations is not None:
                    dict_obj = {
                        'id': f'object_{k}',
                        'location': obj_translations,
                        'quaternion_xyzw': obj_rotations,
                        'projected_keypoints_2d': projected_points.tolist(),
                        'keypoints_3d': point_3d_cam.tolist(),
                        'relative_scale': obj_scale.tolist(),
                        'keypoints_2d': keypoints_2d.tolist()}
                else:
                    dict_obj = {
                        'id': f'object_{k}',
                        'location': [],
                        'quaternion_xyzw': [],
                        'projected_keypoints_2d': [],
                        'keypoints_3d': [],
                        'relative_scale': obj_scale.tolist(),
                        'keypoints_2d': keypoints_2d.tolist()}
                dict_results['objects'].append(dict_obj)

            if save_visualization is True and obj_translations is not None:
                # visualize the points order
                add_obj_order(img, projected_points)
                # visualize the bounding box
                add_coco_hp(img, projected_points[1:])
                # visualize the 6-DoF
                add_axes(img, point_3d_cam, cam_intrinsic)
                # visualize the relative dimension of the object
                add_obj_scale(img, bbox, obj_scale)

        if save_visualization is True:
            cv2.imwrite(output_image_name, img)

        if save_json is True:
            with open(output_json_name, 'w+') as fp:
                json.dump(dict_results, fp, indent=4, sort_keys=True)
