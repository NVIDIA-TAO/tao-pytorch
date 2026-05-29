# SPDX-FileCopyrightText: Copyright (c) 2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

""" BEVFusion visualizer functions. """

from typing import List, Optional, Sequence, Tuple, Union
import numpy as np
from matplotlib.collections import PatchCollection
from matplotlib.patches import PathPatch
from matplotlib.path import Path
import matplotlib.pyplot as plt
from torch import Tensor

import mmcv
from mmdet.visualization import DetLocalVisualizer, get_palette
from mmengine.dist import master_only
from mmengine.structures import InstanceData
from mmengine.visualization.utils import (check_type, color_val_matplotlib, tensor2ndarray)
from mmdet3d.registry import VISUALIZERS
from mmdet3d.structures import (BaseInstance3DBoxes, Det3DDataSample)

try:
    import open3d as o3d
    from open3d import geometry
    from open3d.visualization import Visualizer
except ImportError:
    o3d = geometry = Visualizer = None

from .vis_utils import proj_bbox3d_cam2img, proj_bbox3d_lidar2img
import nvidia_tao_pytorch.cv.bevfusion.structures as tao_structures


@VISUALIZERS.register_module()
class TAO3DLocalVisualizer(DetLocalVisualizer):
    """TAO3D Dataset Local Visualizer Class."""

    def __init__(
        self,
        name: str = 'visualizer',
        points: Optional[np.ndarray] = None,
        image: Optional[np.ndarray] = None,
        pcd_mode: int = 0,
        vis_backends: Optional[List[dict]] = None,
        save_dir: Optional[str] = None,
        bbox_color: Optional[Union[str, Tuple[int]]] = None,
        text_color: Union[str, Tuple[int]] = (200, 200, 200),
        mask_color: Optional[Union[str, Tuple[int]]] = None,
        line_width: Union[int, float] = 3,
        frame_cfg: dict = {"size": 1, "origin": [0, 0, 0]},
        alpha: Union[int, float] = 0.8,
        multi_imgs_col: int = 3,
        fig_show_cfg: dict = {"figsize": (18, 12)}
    ) -> None:
        """ Initialize TAO3DLocalVisualizer.

        Args:
        name (str): Name of the instance. Defaults to 'visualizer'.
        points (np.ndarray, optional): Points to visualize with shape (N, 3+C).
            Defaults to None.
        image (np.ndarray, optional): The origin image to draw. The format
            should be RGB. Defaults to None.
        pcd_mode (int): The point cloud mode (coordinates): 0 represents LiDAR,
            1 represents CAMERA, 2 represents Depth. Defaults to 0.
        vis_backends (List[dict], optional): Visual backend config list.
            Defaults to None.
        save_dir (str, optional): Save file dir for all storage backends.
            If it is None, the backend storage will not save any data.
            Defaults to None.
        bbox_color (str or Tuple[int], optional): Color of bbox lines.
            The tuple of color should be in BGR order. Defaults to None.
        text_color (str or Tuple[int]): Color of texts. The tuple of color
            should be in BGR order. Defaults to (200, 200, 200).
        mask_color (str or Tuple[int], optional): Color of masks. The tuple of
            color should be in BGR order. Defaults to None.
        line_width (int or float): The linewidth of lines. Defaults to 3.
        frame_cfg (dict): The coordinate frame config while Open3D
            visualization initialization.
            Defaults to dict(size=1, origin=[0, 0, 0]).
        alpha (int or float): The transparency of bboxes or mask.
            Defaults to 0.8.
        multi_imgs_col (int): The number of columns in arrangement when showing
            multi-view images.
        """
        super().__init__(
            name=name,
            image=image,
            vis_backends=vis_backends,
            save_dir=save_dir,
            bbox_color=bbox_color,
            text_color=text_color,
            mask_color=mask_color,
            line_width=line_width,
            alpha=alpha)
        if points is not None:
            self.set_points(points, pcd_mode=pcd_mode, frame_cfg=frame_cfg)
        self.multi_imgs_col = multi_imgs_col
        self.fig_show_cfg.update(fig_show_cfg)

    def _clear_o3d_vis(self) -> None:
        """Clear open3d vis."""
        if hasattr(self, 'o3d_vis'):
            del self.o3d_vis
            del self.pcd
            del self.points_colors

    def _initialize_o3d_vis(self, frame_cfg: dict) -> Visualizer:
        """Initialize open3d vis according to frame_cfg.

        Args:
            frame_cfg (dict): The config to create coordinate frame in open3d
                vis.

        Returns:
            :obj:`o3d.visualization.Visualizer`: Created open3d vis.
        """
        if o3d is None or geometry is None:
            raise ImportError(
                'Please run "pip install open3d" to install open3d first.')
        o3d_vis = o3d.visualization.Visualizer()
        o3d_vis.create_window()
        # create coordinate frame
        mesh_frame = geometry.TriangleMesh.create_coordinate_frame(**frame_cfg)
        o3d_vis.add_geometry(mesh_frame)
        return o3d_vis

    @master_only
    def set_points(self,
                   points: np.ndarray,
                   pcd_mode: int = 0,
                   vis_mode: str = 'replace',
                   frame_cfg: dict = {"size": 1, "origin": [0, 0, 0]},
                   points_color: Tuple[float] = (0.8, 0.8, 0.8),
                   points_size: int = 2,
                   mode: str = 'xyz') -> None:
        """Set the point cloud to draw.

        Args:
            points (np.ndarray): Points to visualize with shape (N, 3+C).
            pcd_mode (int): The point cloud mode (coordinates): 0 represents
                LiDAR, 1 represents CAMERA, 2 represents Depth. Defaults to 0.
            vis_mode (str): The visualization mode in Open3D:

                - 'replace': Replace the existing point cloud with input point
                  cloud.
                - 'add': Add input point cloud into existing point cloud.

                Defaults to 'replace'.
            frame_cfg (dict): The coordinate frame config for Open3D
                visualization initialization.
                Defaults to dict(size=1, origin=[0, 0, 0]).
            points_color (Tuple[float]): The color of points.
                Defaults to (1, 1, 1).
            points_size (int): The size of points to show on visualizer.
                Defaults to 2.
            mode (str): Indicate type of the input points, available mode
                ['xyz', 'xyzrgb']. Defaults to 'xyz'.
        """
        assert points is not None
        assert vis_mode in ('replace', 'add')
        check_type('points', points, np.ndarray)

        if not hasattr(self, 'o3d_vis'):
            self.o3d_vis = self._initialize_o3d_vis(frame_cfg)

        if hasattr(self, 'pcd') and vis_mode != 'add':
            self.o3d_vis.remove_geometry(self.pcd)

        # set points size in Open3D
        render_option = self.o3d_vis.get_render_option()
        if render_option is not None:
            render_option.point_size = points_size
            render_option.background_color = np.asarray([0, 0, 0])

        points = points.copy()
        pcd = geometry.PointCloud()
        if mode == 'xyz':
            pcd.points = o3d.utility.Vector3dVector(points[:, :3])
            points_colors = np.tile(
                np.array(points_color), (points.shape[0], 1))
        elif mode == 'xyzrgb':
            pcd.points = o3d.utility.Vector3dVector(points[:, :3])
            points_colors = points[:, 3:6]
            # normalize to [0, 1] for Open3D drawing
            if not ((points_colors >= 0.0) & (points_colors <= 1.0)).all():
                points_colors /= 255.0
        else:
            raise NotImplementedError

        pcd.colors = o3d.utility.Vector3dVector(points_colors)
        self.o3d_vis.add_geometry(pcd)
        self.pcd = pcd
        self.points_colors = points_colors

    @master_only
    def draw_points_on_image(self,
                             points: Union[np.ndarray, Tensor],
                             input_meta: dict,
                             sizes: Union[np.ndarray, int] = 3,
                             max_depth: Optional[float] = None) -> None:
        """Draw projected points on the image.

        Args:
            points (np.ndarray or Tensor): Points to draw.
            sizes (np.ndarray or int): The marker size. Defaults to 10.
            max_depth (float): The max depth in the color map. Defaults to
                None.
        """
        check_type('points', points, (np.ndarray, Tensor))
        points = tensor2ndarray(points)
        assert self._image is not None, 'Please set image using `set_image`'
        projected_points = tao_structures.project_cam2img(points, input_meta['lidar2img'], with_depth=True)
        depths = projected_points[:, 2]
        # Show depth adaptively consideing different scenes
        if max_depth is None:
            max_depth = depths.max()
        colors = (depths % max_depth) / max_depth
        # use colormap to obtain the render color
        color_map = plt.get_cmap('jet')
        self.ax_save.scatter(
            projected_points[:, 0],
            projected_points[:, 1],
            c=colors,
            cmap=color_map,
            s=sizes,
            alpha=0.7,
            edgecolors='none')

    # TODO: set bbox color according to palette
    @master_only
    def draw_proj_bboxes_3d(  # GT Bboxes in cam coordinate
            self,
            bboxes_3d: BaseInstance3DBoxes,
            input_meta: dict,
            edge_colors: Union[str, Tuple[int],
                               List[Union[str, Tuple[int]]]] = 'royalblue',
            line_styles: Union[str, List[str]] = '-',
            line_widths: Union[int, float, List[Union[int, float]]] = 0.5,
            face_colors: Union[str, Tuple[int],
                               List[Union[str, Tuple[int]]]] = 'royalblue',
            alpha: Union[int, float] = 0.4,
            img_size: Optional[Tuple] = None):
        """Draw projected 3D boxes on the image.

        Args:
            bboxes_3d (:obj:`BaseInstance3DBoxes`): 3D bbox
                (x, y, z, x_size, y_size, z_size, yaw, pitch, roll) to visualize.
            input_meta (dict): Input meta information.
            edge_colors (str or Tuple[int] or List[str or Tuple[int]]):
                The colors of bboxes. ``colors`` can have the same length with
                lines or just single value. If ``colors`` is single value, all
                the lines will have the same colors. Refer to `matplotlib.
                colors` for full list of formats that are accepted.
                Defaults to 'royalblue'.
            line_styles (str or List[str]): The linestyle of lines.
                ``line_styles`` can have the same length with texts or just
                single value. If ``line_styles`` is single value, all the lines
                will have the same linestyle.
                for more details. Defaults to '-'.
            line_widths (int or float or List[int or float]): The linewidth of
                lines. ``line_widths`` can have the same length with lines or
                just single value. If ``line_widths`` is single value, all the
                lines will have the same linewidth. Defaults to 2.
            face_colors (str or Tuple[int] or List[str or Tuple[int]]):
                The face colors. Defaults to 'royalblue'.
            alpha (int or float): The transparency of bboxes. Defaults to 0.4.
            img_size (tuple, optional): The size (w, h) of the image.
        """
        check_type('bboxes', bboxes_3d, BaseInstance3DBoxes)
        if isinstance(bboxes_3d, tao_structures.TAOCameraInstance3DBoxes):
            proj_bbox3d_to_img = proj_bbox3d_cam2img
        elif isinstance(bboxes_3d, tao_structures.TAOLiDARInstance3DBoxes):
            proj_bbox3d_to_img = proj_bbox3d_lidar2img
        else:
            raise NotImplementedError('unsupported box type!')

        edge_colors_norm = color_val_matplotlib(edge_colors)

        corners_2d = proj_bbox3d_to_img(bboxes_3d, input_meta)

        if img_size is not None:
            # Filter out the bbox where half of stuff is outside the image.
            # This is for the visualization of multi-view image.
            valid_point_idx = (corners_2d[..., 0] >= 0) & \
                (corners_2d[..., 0] <= img_size[0]) & \
                (corners_2d[..., 1] >= 0) & (corners_2d[..., 1] <= img_size[1])  # noqa: E501
            valid_bbox_idx = valid_point_idx.sum(axis=-1) >= 4
            corners_2d = corners_2d[valid_bbox_idx]

            filter_edge_colors = []
            filter_edge_colors_norm = []
            for i, color in enumerate(edge_colors):
                if valid_bbox_idx[i]:
                    filter_edge_colors.append(color)
                    filter_edge_colors_norm.append(edge_colors_norm[i])
            edge_colors = filter_edge_colors
            edge_colors_norm = filter_edge_colors_norm

        lines_verts_idx = [0, 1, 2, 3, 7, 6, 5, 4, 0, 3, 7, 4, 5, 1, 2, 6]
        lines_verts = corners_2d[:, lines_verts_idx, :]
        front_polys = corners_2d[:, 4:, :]

        codes = [Path.LINETO] * lines_verts.shape[1]
        codes[0] = Path.MOVETO
        pathpatches = []
        for i in range(len(corners_2d)):
            verts = lines_verts[i]
            pth = Path(verts, codes)
            pathpatches.append(PathPatch(pth))

        p = PatchCollection(
            pathpatches,
            facecolors='none',
            edgecolors=edge_colors_norm,
            linewidths=line_widths,
            linestyles=line_styles)

        self.ax_save.add_collection(p)

        # draw a mask on the front of project bboxes
        front_polys = list(front_polys)
        return self.draw_polygons(
            front_polys,
            alpha=alpha,
            edge_colors=edge_colors,
            line_styles=line_styles,
            line_widths=line_widths,
            face_colors=edge_colors)

    def _draw_instances_3d(self,
                           data_input: dict,
                           instances: InstanceData,
                           input_meta: dict,
                           vis_task: str,
                           palette: Optional[List[tuple]] = None) -> dict:
        """Draw 3D instances of GT or prediction.

        Args:
            data_input (dict): The input dict to draw.
            instances (:obj:`InstanceData`): Data structure for instance-level
                annotations or predictions.
            input_meta (dict): Meta information.
            vis_task (str): Visualization task, it includes: 'lidar_det',
                'multi-modality_det', 'mono_det'.
            palette (List[tuple], optional): Palette information corresponding
                to the category. Defaults to None.

        Returns:
            dict: The drawn point cloud and image whose channel is RGB.
        """
        # Only visualize when there is at least one instance
        # if not len(instances) > 0:
        #     return None

        bboxes_3d = instances.bboxes_3d  # BaseInstance3DBoxes
        labels_3d = instances.labels_3d
        data_3d = {}

        if vis_task in ['lidar_det', 'multi-modality_det']:
            assert 'points' in data_input, 'data_input must contain points'
            points = data_input['points']
            check_type('points', points, (np.ndarray, Tensor))
            points = tensor2ndarray(points)

            self.set_points(points, pcd_mode=0)

            assert 'img' in data_input, 'data_input must contain img'
            img = data_input['img']

            # show single-view image
            # TODO: Solve the problem: some line segments of 3d bboxes are
            # out of image by a large margin
            if isinstance(data_input['img'], Tensor):
                img = img.permute(1, 2, 0).numpy()
                img = img[..., [2, 1, 0]]  # bgr to rgb
            self.set_image(img)
            single_img_meta = {}
            for key, meta in input_meta.items():
                if key in ('lidar2img', 'cam2img', 'lidar2cam'):
                    if len(np.array(meta).shape) == 3:  # single-view flatten meta info, Use only first view, need to modify for multi-view
                        if isinstance(meta, np.ndarray):
                            single_img_meta[key] = np.transpose(meta[0])
                        elif isinstance(meta, Tensor):
                            single_img_meta[key] = meta[0].tranpose()
                        elif isinstance(meta, Sequence):
                            single_img_meta[key] = np.transpose(np.array(meta[0]))
                        else:
                            raise NotImplementedError
                    else:
                        if isinstance(meta, np.ndarray):
                            single_img_meta[key] = np.transpose(meta)
                        elif isinstance(meta, Tensor):
                            single_img_meta[key] = meta.tranpose()
                        elif isinstance(meta, Sequence):
                            single_img_meta[key] = np.transpose(np.array(meta))
                        else:
                            raise NotImplementedError
                else:
                    single_img_meta[key] = meta
            self.draw_points_on_image(points[:, 0:3], single_img_meta)
            drawn_img = self.get_image()
            data_3d['img'] = drawn_img
            max_label = int(max(labels_3d) if len(labels_3d) > 0 else 0)
            bbox_color = palette if self.bbox_color is None \
                else self.bbox_color
            bbox_palette = get_palette(bbox_color, max_label + 1)
            colors = [bbox_palette[label] for label in labels_3d]

            self.draw_proj_bboxes_3d(
                bboxes_3d, single_img_meta,
                img_size=img.shape[:2][::-1],
                edge_colors=colors)
            drawn_img = self.get_image()
            data_3d['img'] = drawn_img

            if hasattr(instances, 'centers_2d'):
                centers_2d = instances.centers_2d
                self.draw_points(centers_2d)

        return data_3d

    # TODO: Support Visualize the 3D results from image and point cloud
    # respectively
    @master_only
    def add_datasample(self,
                       name: str,
                       data_input: dict,
                       data_sample: Optional[Det3DDataSample] = None,
                       draw_gt: bool = True,
                       draw_pred: bool = True,
                       show: bool = False,
                       wait_time: float = 0,
                       out_file: Optional[str] = None,
                       o3d_save_path: Optional[str] = None,
                       vis_task: str = 'mono_det',
                       pred_score_thr: float = 0.3,
                       step: int = 0) -> None:
        """Draw datasample and save to all backends.

        - If GT and prediction are plotted at the same time, they are displayed
          in a stitched image where the left image is the ground truth and the
          right image is the prediction.
        - If ``show`` is True, all storage backends are ignored, and the images
          will be displayed in a local window.
        - If ``out_file`` is specified, the drawn image will be saved to
          ``out_file``. It is usually used when the display is not available.

        Args:
            name (str): The image identifier.
            data_input (dict): It should include the point clouds or image
                to draw.
            data_sample (:obj:`Det3DDataSample`, optional): Prediction
                Det3DDataSample. Defaults to None.
            draw_gt (bool): Whether to draw GT Det3DDataSample.
                Defaults to True.
            draw_pred (bool): Whether to draw Prediction Det3DDataSample.
                Defaults to True.
            show (bool): Whether to display the drawn point clouds and image.
                Defaults to False.
            wait_time (float): The interval of show (s). Defaults to 0.
            out_file (str, optional): Path to output file. Defaults to None.
            o3d_save_path (str, optional): Path to save open3d visualized
                results. Defaults to None.
            vis_task (str): Visualization task. Defaults to 'mono_det'.
            pred_score_thr (float): The threshold to visualize the bboxes
                and masks. Defaults to 0.3.
            step (int): Global step value to record. Defaults to 0.
        """
        assert vis_task in ('lidar_det', 'multi-modality_det'), f'got unexpected vis_task {vis_task}.'
        classes = self.dataset_meta.get('classes', None)
        # For object detection datasets, no palette is saved
        palette = self.dataset_meta.get('palette', None)

        gt_data_3d = None
        pred_data_3d = None
        gt_img_data = None
        pred_img_data = None
        if draw_gt and data_sample is not None:
            if 'gt_instances_3d' in data_sample:
                gt_palette = [(230, 0, 0),]
                gt_data_3d = self._draw_instances_3d(
                    data_input, data_sample.gt_instances_3d,
                    data_sample.metainfo, vis_task, gt_palette)  # GT color-Red

            if 'gt_instances' in data_sample:
                if len(data_sample.gt_instances) > 0:
                    assert 'img' in data_input, 'data_input must contain img'
                    gt_palette = [(230, 0, 0),]
                    img = data_input['img']
                    if isinstance(data_input['img'], Tensor):
                        img = data_input['img'].permute(1, 2, 0).numpy()
                        img = img[..., [2, 1, 0]]  # bgr to rgb
                    gt_img_data = self._draw_instances(
                        img, data_sample.gt_instances, classes, gt_palette)

        if draw_pred and data_sample is not None:
            if 'pred_instances_3d' in data_sample:
                pred_instances_3d = data_sample.pred_instances_3d
                # .cpu can not be used for BaseInstance3DBoxes
                # so we need to use .to('cpu')
                pred_instances_3d = pred_instances_3d[
                    pred_instances_3d.scores_3d > pred_score_thr].to('cpu')

                pred_data_3d = self._draw_instances_3d(data_input,
                                                       pred_instances_3d,
                                                       data_sample.metainfo,
                                                       vis_task, palette)
            if 'pred_instances' in data_sample:
                if 'img' in data_input and len(data_sample.pred_instances) > 0:
                    pred_instances = data_sample.pred_instances
                    pred_instances = pred_instances[
                        pred_instances.scores > pred_score_thr].cpu()
                    img = data_input['img']
                    if isinstance(data_input['img'], Tensor):
                        img = data_input['img'].permute(1, 2, 0).numpy()
                        img = img[..., [2, 1, 0]]  # bgr to rgb
                    pred_img_data = self._draw_instances(
                        img, pred_instances, classes, palette)

        # 3d object detection image
        if vis_task in ['lidar_det', 'multi-modality_det']:
            if gt_data_3d is not None and pred_data_3d is not None:
                drawn_img_3d = np.concatenate(
                    (gt_data_3d['img'], pred_data_3d['img']), axis=1)
            elif gt_data_3d is not None:
                drawn_img_3d = gt_data_3d['img']
            elif pred_data_3d is not None:
                drawn_img_3d = pred_data_3d['img']
            else:  # both instances of gt and pred are empty
                drawn_img_3d = None
        else:
            drawn_img_3d = None

        # 2d object detection image
        if gt_img_data is not None and pred_img_data is not None:
            drawn_img = np.concatenate((gt_img_data, pred_img_data), axis=1)
        elif gt_img_data is not None:
            drawn_img = gt_img_data
        elif pred_img_data is not None:
            drawn_img = pred_img_data
        else:
            drawn_img = None

        if out_file is not None:

            # check the suffix of the name of image file
            if not (out_file.endswith('.png') or out_file.endswith('.jpg')):
                out_file = f'{out_file}.png'
            if drawn_img_3d is not None:
                mmcv.imwrite(drawn_img_3d[..., ::-1], out_file)
            if drawn_img is not None:
                mmcv.imwrite(drawn_img[..., ::-1], out_file)
        else:
            self.add_image(name, drawn_img_3d, step)
