# SPDX-FileCopyrightText: Copyright (c) 2023 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Evaluation utils for PointPillars."""
import pickle
import time

import numpy as np
import torch
import tqdm
import matplotlib.pyplot as plt

from nvidia_tao_pytorch.pointcloud.pointpillars.pcdet.models import load_data_to_gpu
from nvidia_tao_pytorch.pointcloud.pointpillars.pcdet.utils import common_utils


def statistics_info(cfg, ret_dict, metric, disp_dict):
    """Statistics infomation."""
    for cur_thresh in cfg.model.post_processing.recall_thresh_list:
        metric['recall_roi_%s' % str(cur_thresh)] += ret_dict.get('roi_%s' % str(cur_thresh), 0)
        metric['recall_rcnn_%s' % str(cur_thresh)] += ret_dict.get('rcnn_%s' % str(cur_thresh), 0)
    metric['gt_num'] += ret_dict.get('gt', 0)
    min_thresh = cfg.model.post_processing.recall_thresh_list[0]
    disp_dict['recall_%s' % str(min_thresh)] = \
        '(%d, %d) / %d' % (metric['recall_roi_%s' % str(min_thresh)], metric['recall_rcnn_%s' % str(min_thresh)], metric['gt_num'])


def eval_one_epoch(cfg, model, dataloader, epoch_id, logger, dist_test=False, save_to_file=False, result_dir=None):
    """Evaluate on one epoch."""
    result_dir.mkdir(parents=True, exist_ok=True)

    final_output_dir = result_dir / "detected_labels"
    if save_to_file:
        final_output_dir.mkdir(parents=True, exist_ok=True)

    metric = {
        'gt_num': 0,
    }
    for cur_thresh in cfg.model.post_processing.recall_thresh_list:
        metric['recall_roi_%s' % str(cur_thresh)] = 0
        metric['recall_rcnn_%s' % str(cur_thresh)] = 0

    dataset = dataloader.dataset
    class_names = dataset.class_names
    det_annos = []

    logger.info('*************** EPOCH %s EVALUATION *****************' % epoch_id)
    if dist_test:
        num_gpus = torch.cuda.device_count()
        local_rank = cfg.local_rank % num_gpus
        model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[local_rank], broadcast_buffers=False)
    model.eval()

    if cfg.local_rank == 0:
        progress_bar = tqdm.tqdm(total=len(dataloader), leave=True, desc='eval', dynamic_ncols=True)
    start_time = time.time()
    for _i, batch_dict in enumerate(dataloader):
        load_data_to_gpu(batch_dict)
        with torch.no_grad():
            pred_dicts, ret_dict = model(batch_dict)
        disp_dict = {}

        statistics_info(cfg, ret_dict, metric, disp_dict)
        annos = dataset.generate_prediction_dicts(
            batch_dict, pred_dicts, class_names,
            output_path=final_output_dir if save_to_file else None
        )
        det_annos += annos
        if cfg.local_rank == 0:
            progress_bar.set_postfix(disp_dict)
            progress_bar.update()

    if cfg.local_rank == 0:
        progress_bar.close()

    if dist_test:
        world_size = common_utils.get_dist_info()[1]
        det_annos = common_utils.merge_results_dist(det_annos, len(dataset))
        metric = common_utils.merge_results_dist([metric], world_size)

    logger.info('*************** Performance of EPOCH %s *****************' % epoch_id)
    sec_per_example = (time.time() - start_time) / len(dataloader.dataset)
    logger.info('Generate label finished(sec_per_example: %.4f second).' % sec_per_example)

    if cfg.local_rank != 0:
        return {}

    ret_dict = {}
    if dist_test:
        for key, _val in metric[0].items():
            for k in range(1, world_size):
                metric[0][key] += metric[k][key]
        metric = metric[0]

    gt_num_cnt = metric['gt_num']
    for cur_thresh in cfg.model.post_processing.recall_thresh_list:
        cur_roi_recall = metric['recall_roi_%s' % str(cur_thresh)] / max(gt_num_cnt, 1)
        cur_rcnn_recall = metric['recall_rcnn_%s' % str(cur_thresh)] / max(gt_num_cnt, 1)
        logger.info('recall_roi_%s: %f' % (cur_thresh, cur_roi_recall))
        logger.info('recall_rcnn_%s: %f' % (cur_thresh, cur_rcnn_recall))
        ret_dict['recall/roi_%s' % str(cur_thresh)] = cur_roi_recall
        ret_dict['recall/rcnn_%s' % str(cur_thresh)] = cur_rcnn_recall

    total_pred_objects = 0
    for anno in det_annos:
        total_pred_objects += anno['name'].__len__()
    logger.info('Average predicted number of objects(%d samples): %.3f'
                % (len(det_annos), total_pred_objects / max(1, len(det_annos))))

    with open(result_dir / 'result.pkl', 'wb') as f:
        pickle.dump(det_annos, f)

    result_str, result_dict = dataset.evaluation(
        det_annos, class_names,
        eval_metric=cfg.model.post_processing.eval_metric,
        output_path=final_output_dir
    )

    logger.info(result_str)
    ret_dict.update(result_dict)

    logger.info('Result is save to %s' % result_dir)
    logger.info('****************Evaluation done.*****************')
    return ret_dict


def view_points(points: np.ndarray, view: np.ndarray, normalize: bool) -> np.ndarray:
    """
    This is a helper class that maps 3d points to a 2d plane. It can be used to implement both perspective and
    orthographic projections. It first applies the dot product between the points and the view. By convention,
    the view should be such that the data is projected onto the first 2 axis. It then optionally applies a
    normalization along the third dimension.
    For a perspective projection the view should be a 3x3 camera matrix, and normalize=True
    For an orthographic projection with translation the view is a 3x4 matrix and normalize=False
    For an orthographic projection without translation the view is a 3x3 matrix (optionally 3x4 with last columns
     all zeros) and normalize=False
    :param points: <np.float32: 3, n> Matrix of points, where each point (x, y, z) is along each column.
    :param view: <np.float32: n, n>. Defines an arbitrary projection (n <= 4).
        The projection should be such that the corners are projected onto the first 2 axis.
    :param normalize: Whether to normalize the remaining coordinate (along the third axis).
    :return: <np.float32: 3, n>. Mapped point. If normalize=False, the third coordinate is the height.
    """
    assert view.shape[0] <= 4
    assert view.shape[1] <= 4
    assert points.shape[0] == 3
    viewpad = np.eye(4)
    viewpad[:view.shape[0], :view.shape[1]] = view
    nbr_points = points.shape[1]
    # Do operation in homogenous coordinates.
    points = np.concatenate((points, np.ones((1, nbr_points))))
    points = np.dot(viewpad, points)
    points = points[:3, :]
    if normalize:
        points = points / points[2:3, :].repeat(3, 0).reshape(3, nbr_points)
    return points


def check_numpy_to_torch(x):
    """Check and convert numpy array to torch tensor."""
    if isinstance(x, np.ndarray):
        return torch.from_numpy(x).float(), True
    return x, False


def rotate_points_along_z(points, angle):
    """
    Args:
        points: (B, N, 3 + C)
        angle: (B), angle along z-axis, angle increases x ==> y
    Returns:

    """
    points, is_numpy = check_numpy_to_torch(points)
    angle, _ = check_numpy_to_torch(angle)
    cosa = torch.cos(angle)
    sina = torch.sin(angle)
    zeros = angle.new_zeros(points.shape[0])
    ones = angle.new_ones(points.shape[0])
    rot_matrix = torch.stack((
        cosa,  sina, zeros,
        -sina, cosa, zeros,
        zeros, zeros, ones
    ), dim=1).view(-1, 3, 3).float()
    points_rot = torch.matmul(points[:, :, 0:3], rot_matrix)
    points_rot = torch.cat((points_rot, points[:, :, 3:]), dim=-1)
    return points_rot.numpy() if is_numpy else points_rot


def boxes_to_corners_3d(boxes3d):
    """
        7 -------- 4
       /|         /|
      6 -------- 5 .
      | |        | |
      . 3 -------- 0
      |/         |/
      2 -------- 1
    Args:
        boxes3d:  (N, 7) [x, y, z, dx, dy, dz, heading], (x, y, z) is the box center

    Returns:
    """
    boxes3d, is_numpy = check_numpy_to_torch(boxes3d)
    template = boxes3d.new_tensor((
        [1, 1, -1], [1, -1, -1], [-1, -1, -1], [-1, 1, -1],
        [1, 1, 1], [1, -1, 1], [-1, -1, 1], [-1, 1, 1],
    )) / 2
    corners3d = boxes3d[:, None, 3:6].repeat(1, 8, 1) * template[None, :, :]
    corners3d = rotate_points_along_z(corners3d.view(-1, 8, 3), boxes3d[:, 6]).view(-1, 8, 3)
    corners3d += boxes3d[:, None, 0:3]
    return corners3d.numpy() if is_numpy else corners3d


def draw_box(box, axis, view, colors, linewidth):
    """Draw box."""
    # box: (3, 4), append first point to form a loop
    x = np.concatenate((box[0, :], box[0, :1]), axis=-1)
    y = np.concatenate((box[1, :], box[1, :1]), axis=-1)
    axis.plot(
        x, y,
        color=colors[0],
        linewidth=linewidth
    )


def visual(points, gt_anno, det, det_scores, frame_id, eval_range=35, conf_th=0.1):
    """Visualization."""
    _, ax = plt.subplots(1, 1, figsize=(9, 9), dpi=200)
    # points
    points = view_points(points[:3, :], np.eye(4), normalize=False)
    dists = np.sqrt(np.sum(points[:2, :] ** 2, axis=0))
    colors = np.minimum(1, dists / eval_range)
    ax.scatter(points[0, :], points[1, :], c=colors, s=0.2)
    # (B, 8, 3)
    boxes_gt = boxes_to_corners_3d(gt_anno)
    # Show GT boxes.
    for box in boxes_gt:
        # (8, 3)
        bev = box[4:, :]
        bev = view_points(bev.transpose(), np.eye(4), normalize=False)
        draw_box(bev, ax, view=np.eye(4), colors=('r', 'r', 'r'), linewidth=2)
    # Show EST boxes.
    if len(det) == 0:
        plt.axis('off')
        plt.savefig(frame_id + ".png")
        plt.close()
        return
    boxes_est = boxes_to_corners_3d(det)
    for idx, box in enumerate(boxes_est):
        if det_scores[idx] < conf_th:
            continue
        bev = box[4:, :]
        bev = view_points(bev.transpose(), np.eye(4), normalize=False)
        draw_box(bev, ax, view=np.eye(4), colors=('g', 'g', 'g'), linewidth=1)
    axes_limit = eval_range + 3  # Slightly bigger to include boxes that extend beyond the range.
    ax.set_xlim(-axes_limit, axes_limit)
    ax.set_ylim(-axes_limit, axes_limit)
    plt.axis('off')
    plt.savefig(frame_id + ".png")
    plt.close()


def infer_one_epoch(
    cfg, model, dataloader,
    logger, save_to_file=False,
    result_dir=None
):
    """Do inference on one epoch."""
    result_dir.mkdir(parents=True, exist_ok=True)
    final_output_dir = result_dir / "detected_labels"
    if save_to_file:
        final_output_dir.mkdir(parents=True, exist_ok=True)
    image_output_dir = result_dir / "detected_boxes"
    image_output_dir.mkdir(parents=True, exist_ok=True)
    dataset = dataloader.dataset
    class_names = dataset.class_names
    det_annos = []
    model.eval()
    if cfg.local_rank == 0:
        progress_bar = tqdm.tqdm(total=len(dataloader), leave=True, desc='infer', dynamic_ncols=True)
    for _i, batch_dict in enumerate(dataloader):
        load_data_to_gpu(batch_dict)
        with torch.no_grad():
            pred_dicts, ret_dict = model(batch_dict)
        disp_dict = {}
        annos = dataset.generate_prediction_dicts(
            batch_dict, pred_dicts, class_names,
            output_path=final_output_dir if save_to_file else None
        )
        det_annos += annos
        visual(
            batch_dict['points'].cpu().numpy()[:, 1:].transpose(),
            batch_dict["gt_boxes"][0].cpu().numpy()[:, :7],
            pred_dicts[0]['pred_boxes'].cpu().numpy(),
            pred_dicts[0]['pred_scores'].cpu().numpy(),
            str(image_output_dir / batch_dict['frame_id'][0]),
            eval_range=100,
            conf_th=cfg.inference.viz_conf_thresh
        )
        if cfg.local_rank == 0:
            progress_bar.set_postfix(disp_dict)
            progress_bar.update()
    if cfg.local_rank == 0:
        progress_bar.close()
    ret_dict = {}
    with open(result_dir / 'result.pkl', 'wb') as f:
        pickle.dump(det_annos, f)
    logger.info('Result is save to %s' % result_dir)
    logger.info('****************Inference done.*****************')
    return ret_dict


def sparse_to_dense(points, batch_size):
    """Convert sparse points to dense format."""
    points = points.cpu().numpy()
    points_dense = []
    num_points_dense = []
    for b in range(batch_size):
        points_per_frame = np.copy(points[points[:, 0] == b][:, 1:])
        num_points_ = points_per_frame.shape[0]
        points_dense.append(points_per_frame)
        num_points_dense.append(num_points_)
    return points_dense, num_points_dense


if __name__ == '__main__':
    pass
