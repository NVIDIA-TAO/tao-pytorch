# SPDX-FileCopyrightText: Copyright (c) 2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""BEVFusion inferencer utility Function"""

from typing import Dict, List, Optional, Sequence, Union, Tuple, Iterable
from rich.progress import track
import os.path as osp
import numpy as np
import os
import copy

import mmengine
from mmengine.dataset import Compose
from mmengine.infer.infer import ModelType
from mmengine.structures import InstanceData

import mmcv
from mmcv.transforms.base import BaseTransform

from mmdet3d.registry import TRANSFORMS, INFERENCERS
from mmdet3d.utils import ConfigType
from mmdet3d.apis import Base3DInferencer
from mmdet3d.structures import Det3DDataSample

import nvidia_tao_pytorch.cv.bevfusion.structures as tao_structures

InstanceList = List[InstanceData]
InputType = Union[str, np.ndarray]
InputsType = Union[InputType, Sequence[InputType]]
PredType = Union[InstanceData, InstanceList]
ImgType = Union[np.ndarray, Sequence[np.ndarray]]
ResType = Union[Dict, List[Dict], InstanceData, List[InstanceData]]


def prepare_inferencer_args(infer_cfg, checkpoint, results_dir):
    """Prepare inferencer arguments"""
    init_args = {'model': infer_cfg, 'weights': checkpoint, 'device': 'cuda:0'}

    info = infer_cfg['test_dataloader']['dataset']['ann_file']

    data_prefix = infer_cfg['test_dataloader']['dataset']['data_prefix']

    if info is None:
        # provide path to image file and pcd file and raw lidar2cam and cam2img (row-major or col-major ??)
        # sanity check
        check_list = [infer_cfg['infer_data_config']['cam2img'], infer_cfg['infer_data_config']['lidar2cam'],
                      infer_cfg['infer_data_config']['img_file'], infer_cfg['infer_data_config']['pc_file']]
        if any(x is None for x in check_list):
            raise ValueError('All four values: lidar2cam, cam2img, img_file, pc_file must be \
            set in config file to run inference without pkl file')

        call_args = {'show': infer_cfg['infer_data_config']['show'],
                     'cam_type': infer_cfg['infer_data_config']['default_cam_key'],
                     'pred_score_thr': infer_cfg['infer_data_config']['conf_threshold'],
                     'out_dir': results_dir,
                     'batch_size': infer_cfg['test_dataloader']['batch_size'],
                     'inputs': {'img': infer_cfg['infer_data_config']['img_file'],
                                'points': infer_cfg['infer_data_config']['pc_file'],
                                'calib': {'cam2img': infer_cfg['infer_data_config']['cam2img'],
                                          'lidar2cam': infer_cfg['infer_data_config']['lidar2cam']}}}
    else:
        call_args = {'show': infer_cfg['infer_data_config']['show'],
                     'cam_type': infer_cfg['infer_data_config']['default_cam_key'],
                     'pred_score_thr': infer_cfg['infer_data_config']['conf_threshold'],
                     'out_dir': results_dir,
                     'batch_size': infer_cfg['test_dataloader']['batch_size'],
                     'inputs': {'img_root': data_prefix['img'],
                                'points_root': data_prefix['pts'], 'infos': info}}

    return init_args, call_args


@INFERENCERS.register_module()
class TAOMultiModalDet3DInferencer(Base3DInferencer):
    """The inferencer of multi-modality detection."""

    def __init__(self,
                 model: Union[ModelType, str, None] = None,
                 weights: Optional[str] = None,
                 device: Optional[str] = None,
                 scope: str = 'mmdet3d',
                 palette: str = 'none') -> None:
        """Inferencer Initialization

        Args:
            model (str, optional): Path to the config file or the model name
                defined in metafile. For example, it could be
                "pointpillars_kitti-3class" or
                "configs/pointpillars/pointpillars_hv_secfpn_8xb6-160e_kitti-3d-3class.py".
                If model is not specified, user must provide the
                `weights` saved by MMEngine which contains the config string.
                Defaults to None.
            weights (str, optional): Path to the checkpoint. If it is not specified
                and model is a model name of metafile, the weights will be loaded
                from metafile. Defaults to None.
            device (str, optional): Device to run inference. If None, the available
                device will be automatically used. Defaults to None.
            scope (str): The scope of registry. Defaults to 'mmdet3d'.
            palette (str): The palette of visualization. Defaults to 'none'.
        """
        # A global counter tracking the number of frames processed, for
        # naming of the output results
        self.num_visualized_frames = 0
        super().__init__(
            model=model,
            weights=weights,
            device=device,
            scope=scope,
            palette=palette)

    def _inputs_to_list(self,
                        inputs: Union[dict, list],
                        cam_type: str = 'CAM2',
                        **kwargs) -> list:
        """Preprocess the inputs to a list.

        Preprocess inputs to a list according to its type:

        - list or tuple: return inputs
        - dict: the value with key 'points' is
            - Directory path: return all files in the directory
            - other cases: return a list containing the string. The string
              could be a path to file, a url or other types of string according
              to the task.

        Args:
            inputs (Union[dict, list]): Inputs for the inferencer.

        Returns:
            list: List of input for the :meth:`preprocess`.
        """
        out_inputs = []
        if 'calib' in inputs:
            calib = inputs.pop('calib')
            cam2img = np.asarray(calib['cam2img'], dtype=np.float32)
            lidar2cam = np.asarray(calib['lidar2cam'], dtype=np.float32)

            if cam2img.shape != (4, 4):
                cam2img = np.concatenate([cam2img, np.array([[0., 0., 0., 1.]])], axis=0)

            if lidar2cam.shape != (4, 4):
                lidar2cam = np.concatenate([lidar2cam, np.array([[0., 0., 0., 1.]])], axis=0)

            lidar2img = cam2img @ lidar2cam
            out_inputs.append({'img': inputs['img'], 'points': inputs['points'], 'cam2img': cam2img, 'lidar2cam': lidar2cam, 'lidar2img': lidar2img})
        else:  # process with pkl
            if isinstance(inputs, dict):
                assert 'infos' in inputs, 'inputs must contain infos'
                infos = inputs.pop('infos')
                info_list = mmengine.load(infos)['data_list']
                img_root = inputs.pop('img_root')
                points_root = inputs.pop('points_root')

                for _, data_info in enumerate(info_list):
                    img_path = osp.join(img_root, data_info['images'][cam_type]['img_path'])
                    lidar_path = osp.join(points_root, data_info['lidar_points']['lidar_path'])

                    cam2img = np.asarray(data_info['images'][cam_type]['cam2img'], dtype=np.float32)
                    lidar2cam = np.asarray(data_info['images'][cam_type]['lidar2cam'], dtype=np.float32)
                    if 'lidar2img' in data_info['images'][cam_type]:
                        lidar2img = np.asarray(data_info['images'][cam_type]['lidar2img'], dtype=np.float32)
                    else:
                        lidar2img = cam2img @ lidar2cam

                    out_inputs.append({'img': img_path, 'points': lidar_path,
                                       'cam2img': cam2img, 'lidar2cam': lidar2cam, 'lidar2img': lidar2img})

        return out_inputs

    def _init_pipeline(self, cfg: ConfigType) -> Compose:
        """Initialize the test pipeline."""
        pipeline_cfg = cfg.test_dataloader.dataset.pipeline

        load_point_idx = self._get_transform_idx(pipeline_cfg, 'TAOLoadPointsFromFile')

        load_img_idx = self._get_transform_idx(pipeline_cfg, 'BEVFusionLoadMultiViewImageFromFiles')

        if load_point_idx == -1 or load_img_idx == -1:
            raise ValueError(
                'Both TAOLoadPointsFromFile and BEVFusionLoadMultiViewImageFromFiles must '
                'be specified the pipeline, but TAOLoadPointsFromFile is '
                f'{load_point_idx == -1} and BEVFusionLoadMultiViewImageFromFiles is '
                f'{load_img_idx}')

        load_cfg = pipeline_cfg[load_point_idx]
        self.coord_type, self.load_dim = load_cfg['coord_type'], load_cfg[
            'load_dim']
        self.use_dim = list(range(load_cfg['use_dim'])) if isinstance(
            load_cfg['use_dim'], int) else load_cfg['use_dim']

        load_point_args = pipeline_cfg[load_point_idx]
        load_point_args.pop('type')
        load_img_args = pipeline_cfg[load_img_idx]
        load_img_args.pop('type')

        load_idx = min(load_point_idx, load_img_idx)
        pipeline_cfg.pop(max(load_point_idx, load_img_idx))

        pipeline_cfg[load_idx] = {'type': 'TAOMultiModalDet3DInferencerLoader',
                                  'default_cam_key': cfg.infer_data_config.default_cam_key,
                                  'load_point_args': load_point_args,
                                  'load_img_args': load_img_args}
        self.per_sequence = cfg['infer_data_config']['per_sequence']

        return Compose(pipeline_cfg)

    def _dispatch_kwargs(self,
                         out_dir: str = '',
                         cam_type: str = '',
                         **kwargs) -> Tuple[Dict, Dict, Dict, Dict]:
        """Dispatch kwargs to preprocess(), forward(), visualize() and
        postprocess() according to the actual demands.

        Args:
            out_dir (str): Dir to save the inference results.
            cam_type (str): Camera type. Defaults to ''.
            **kwargs (dict): Key words arguments passed to :meth:`preprocess`,
                :meth:`forward`, :meth:`visualize` and :meth:`postprocess`.
                Each key in kwargs should be in the corresponding set of
                ``preprocess_kwargs``, ``forward_kwargs``, ``visualize_kwargs``
                and ``postprocess_kwargs``.

        Returns:
            Tuple[Dict, Dict, Dict, Dict]: kwargs passed to preprocess,
            forward, visualize and postprocess respectively.
        """
        kwargs['img_out_dir'] = out_dir
        kwargs['pred_out_dir'] = out_dir
        if cam_type != '':
            kwargs['cam_type_dir'] = cam_type
            kwargs['cam_type'] = cam_type

        method_kwargs = self.preprocess_kwargs | self.forward_kwargs | \
            self.visualize_kwargs | self.postprocess_kwargs

        union_kwargs = method_kwargs | set(kwargs.keys())
        if union_kwargs != method_kwargs:
            unknown_kwargs = union_kwargs - method_kwargs
            raise ValueError(
                f'unknown argument {unknown_kwargs} for `preprocess`, '
                '`forward`, `visualize` and `postprocess`')

        preprocess_kwargs = {}
        forward_kwargs = {}
        visualize_kwargs = {}
        postprocess_kwargs = {}

        for key, value in kwargs.items():
            if key in self.preprocess_kwargs:
                preprocess_kwargs[key] = value
            elif key in self.forward_kwargs:
                forward_kwargs[key] = value
            elif key in self.visualize_kwargs:
                visualize_kwargs[key] = value
            else:
                postprocess_kwargs[key] = value
        return (
            preprocess_kwargs,
            forward_kwargs,
            visualize_kwargs,
            postprocess_kwargs,
        )

    def _get_chunk_data(self, inputs: Iterable, chunk_size: int):
        """Get batch data from inputs.

        Args:
            inputs (Iterable): An iterable dataset.
            chunk_size (int): Equivalent to batch size.

        Yields:
            list: batch data.
        """
        inputs_iter = iter(inputs)
        while True:
            try:
                chunk_data = []
                for _ in range(chunk_size):
                    inputs_ = next(inputs_iter)
                    ori_inputs_ = {}
                    if isinstance(inputs_, dict):
                        if 'img' in inputs_:
                            ori_inputs_['img'] = inputs_['img']
                        else:
                            ori_inputs_['img'] = inputs_['img_path']

                        if 'points' in inputs_:
                            ori_inputs_['points'] = inputs_['points']

                        chunk_data.append(
                            (ori_inputs_,
                             self.pipeline(copy.deepcopy(inputs_))))
                    else:
                        chunk_data.append((inputs_, self.pipeline(inputs_)))
                yield chunk_data
            except StopIteration:
                if chunk_data:
                    yield chunk_data
                break

    def preprocess(self, inputs: InputsType, batch_size: int = 1, **kwargs):
        """Process the inputs into a model-feedable format.

        Customize your preprocess by overriding this method. Preprocess should
        return an iterable object, of which each item will be used as the
        input of ``model.test_step``.

        ``BaseInferencer.preprocess`` will return an iterable chunked data,
        which will be used in __call__ like this:

        .. code-block:: python

            def __call__(self, inputs, batch_size=1, **kwargs):
                chunked_data = self.preprocess(inputs, batch_size, **kwargs)
                for batch in chunked_data:
                    preds = self.forward(batch, **kwargs)

        Args:
            inputs (InputsType): Inputs given by user.
            batch_size (int): batch size. Defaults to 1.

        Yields:
            Any: Data processed by the ``pipeline`` and ``collate_fn``.
        """
        chunked_data = self._get_chunk_data(inputs, batch_size)
        yield from map(self.collate_fn, chunked_data)

    def __call__(self,
                 inputs: InputsType,
                 batch_size: int = 1,
                 return_datasamples: bool = False,
                 **kwargs) -> Optional[dict]:
        """Call the inferencer.

        Args:
            inputs (InputsType): Inputs for the inferencer.
            batch_size (int): Batch size. Defaults to 1.
            return_datasamples (bool): Whether to return results as
                :obj:`BaseDataElement`. Defaults to False.
            **kwargs: Key words arguments passed to :meth:`preprocess`,
                :meth:`forward`, :meth:`visualize` and :meth:`postprocess`.
                Each key in kwargs should be in the corresponding set of
                ``preprocess_kwargs``, ``forward_kwargs``, ``visualize_kwargs``
                and ``postprocess_kwargs``.


        Returns:
            dict: Inference and visualization results.
        """
        (
            preprocess_kwargs,
            forward_kwargs,
            visualize_kwargs,
            postprocess_kwargs,
        ) = self._dispatch_kwargs(**kwargs)

        cam_type = preprocess_kwargs.pop('cam_type', 'CAM2')
        ori_inputs = self._inputs_to_list(inputs, cam_type=cam_type)

        inputs = self.preprocess(
            ori_inputs, batch_size=batch_size, **preprocess_kwargs)

        results_dict = {'predictions': [], 'visualization': []}
        for ori_input, data in (track(inputs, description='Inference')
                                if self.show_progress else inputs):
            preds = self.forward(data, **forward_kwargs)
            visualization = self.visualize(ori_input, preds, **visualize_kwargs)
            results = self.postprocess(preds, visualization, return_datasamples,
                                       **postprocess_kwargs)
            results_dict['predictions'].extend(results['predictions'])
            if results['visualization'] is not None:
                results_dict['visualization'].extend(results['visualization'])
        return results_dict

    def pred2dict(self,
                  data_sample: Det3DDataSample,
                  pred_out_dir: str = '') -> Dict:
        """Extract elements necessary to represent a prediction into a
        dictionary.

        It's better to contain only basic data elements such as strings and
        numbers in order to guarantee it's json-serializable.

        Args:
            data_sample (:obj:`DetDataSample`): Predictions of the model.
            pred_out_dir: Dir to save the inference results w/o
                visualization. If left as empty, no file will be saved.
                Defaults to ''.

        Returns:
            dict: Prediction results.
        """
        result = {}
        if 'pred_instances_3d' in data_sample:
            pred_instances_3d = data_sample.pred_instances_3d.numpy()
            result = {
                'labels_3d': pred_instances_3d.labels_3d.tolist(),
                'scores_3d': pred_instances_3d.scores_3d.tolist(),
                'bboxes_3d': pred_instances_3d.bboxes_3d.tensor.cpu().tolist()
            }

        if 'pred_pts_seg' in data_sample:
            pred_pts_seg = data_sample.pred_pts_seg.numpy()
            result['pts_semantic_mask'] = \
                pred_pts_seg.pts_semantic_mask.tolist()

        if pred_out_dir != '':
            if 'lidar_path' in data_sample:
                lidar_path = osp.basename(data_sample.lidar_path)
                lidar_path = osp.splitext(lidar_path)[0]
                out_json_path = osp.join(pred_out_dir, 'preds',
                                         lidar_path + '.json')
            elif 'img_path' in data_sample:
                if len(data_sample.img_path) > 1:
                    raise NotImplementedError
                # single-view image only supported
                img_path = osp.basename(data_sample.img_path[0])
                img_path = osp.splitext(img_path)[0]
                out_json_path = osp.join(pred_out_dir, 'preds',
                                         img_path + '.json')
            else:
                out_json_path = osp.join(
                    pred_out_dir, 'preds',
                    f'{str(self.num_visualized_imgs).zfill(8)}.json')
            mmengine.dump(result, out_json_path)

        return result

    def visualize(self,
                  inputs: InputsType,
                  preds: PredType,
                  return_vis: bool = False,
                  show: bool = False,
                  wait_time: int = 0,
                  draw_gt: bool = False,
                  draw_pred: bool = True,
                  pred_score_thr: float = 0.3,
                  no_save_vis: bool = False,
                  img_out_dir: str = '',
                  cam_type_dir: str = 'CAM2') -> Union[List[np.ndarray], None]:
        """Visualize predictions.

        Args:
            inputs (InputsType): Inputs for the inferencer.
            preds (PredType): Predictions of the model.
            return_vis (bool): Whether to return the visualization result.
                Defaults to False.
            show (bool): Whether to display the image in a popup window.
                Defaults to False.
            wait_time (float): The interval of show (s). Defaults to 0.
            draw_pred (bool): Whether to draw predicted bounding boxes.
                Defaults to True.
            no_save_vis (bool): Whether to save visualization results.
            pred_score_thr (float): Minimum score of bboxes to draw.
                Defaults to 0.3.
            img_out_dir (str): Output directory of visualization results.
                If left as empty, no file will be saved. Defaults to ''.

        Returns:
            List[np.ndarray] or None: Returns visualization results only if
            applicable.
        """
        if no_save_vis is True:
            img_out_dir = ''

        if not show and img_out_dir == '' and not return_vis:
            return None

        if getattr(self, 'visualizer') is None:
            raise ValueError('Visualization needs the "visualizer" term'
                             'defined in the config, but got None.')

        results = []

        for idx, pred in enumerate(preds):

            points_input = inputs['points'][idx]
            if isinstance(points_input, str):
                basename = osp.basename(points_input)
                if basename.endswith('.bin'):
                    pts_bytes = mmengine.fileio.get(points_input)
                    points = np.frombuffer(pts_bytes, dtype=np.float32)
                elif basename.endswith('.npy'):
                    points = np.load(points_input, allow_pickle=True)
                else:
                    points = np.fromfile(points_input, dtype=np.float32)
                points = points.reshape(-1, self.load_dim)
                points = points[:, self.use_dim]
                pc_name = basename.split('.')[0]
                pc_name = f'{pc_name}.png'
            elif isinstance(points_input, np.ndarray):
                points = points_input.copy()
                pc_num = str(self.num_visualized_frames).zfill(8)
                pc_name = f'{pc_num}.png'
            else:
                raise ValueError('Unsupported input type: '
                                 f'{type(points_input)}')

            if img_out_dir != '' and show:
                o3d_save_path = osp.join(img_out_dir, 'vis_lidar', pc_name)
                mmengine.mkdir_or_exist(osp.dirname(o3d_save_path))
            else:
                o3d_save_path = None

            img_input = inputs['img'][idx]
            if isinstance(img_input, str):
                img_bytes = mmengine.fileio.get(img_input)
                img = mmcv.imfrombytes(img_bytes)
                img = img[:, :, ::-1]
                img_name = osp.basename(img_input)
            elif isinstance(img_input, np.ndarray):
                img = img_input.copy()
                img_num = str(self.num_visualized_frames).zfill(8)
                img_name = f'{img_num}.jpg'
            else:
                raise ValueError('Unsupported input type: '
                                 f'{type(img_input)}')

            if self.per_sequence:
                sequence_name = img_input.split(os.sep)[-2]
                file_name = img_input.split(os.sep)[-1]
                out_file = osp.join(img_out_dir, 'visualization', sequence_name, file_name)
            else:
                out_file = osp.join(img_out_dir, 'visualization',
                                    img_name) if img_out_dir != '' else None

            data_input = {"points": points, "img": img}
            self.visualizer.add_datasample(
                pc_name,
                data_input,
                pred,
                show=show,
                wait_time=wait_time,
                draw_gt=False,
                draw_pred=draw_pred,
                pred_score_thr=pred_score_thr,
                o3d_save_path=o3d_save_path,
                out_file=out_file,
                vis_task='multi-modality_det',
            )
            results.append(points)
            self.num_visualized_frames += 1

        return results


@TRANSFORMS.register_module()
class TAOMultiModalDet3DInferencerLoader(BaseTransform):
    """Load point cloud and image in the Inferencer's pipeline.
    Added keys:
      - points
      - img
      - cam2img
      - lidar2cam
      - lidar2img
      - timestamp
      - axis_align_matrix
      - box_type_3d
      - box_mode_3d
    """

    def __init__(self, default_cam_key: str, load_point_args: dict, load_img_args: dict) -> None:
        """Initalization of Inferencer DataLoader. """
        super().__init__()
        self.default_cam_key = default_cam_key
        self.points_from_file = TRANSFORMS.build(
            {"type": 'TAOLoadPointsFromFile', **load_point_args})
        coord_type = load_point_args['coord_type']
        self.box_type_3d, self.box_mode_3d = tao_structures.get_box_type_tao3d(coord_type)

        self.imgs_from_file = TRANSFORMS.build(
            {"type": 'BEVFusionLoadMultiViewImageFromFiles', **load_img_args})

    def transform(self, results: dict) -> dict:
        """Transform function to add image meta information.
        Args:
            results (dict): Single input.

        Returns:
            dict: The dict contains loaded image, point cloud and meta
            information.
        """
        assert 'points' in results and 'img' in results, \
            "key 'points', 'img' and must be in input dict," \
            f'but got {results}'

        if isinstance(results['points'], str):
            inputs = {'lidar_points': {'lidar_path': results['points']},
                      'timestamp': 1,
                      # for ScanNet demo we need axis_align_matrix
                      'axis_align_matrix': np.eye(4),
                      'box_type_3d': self.box_type_3d,
                      'box_mode_3d': self.box_mode_3d
                      }
        else:
            raise ValueError('Unsupported input points type: '
                             f"{type(results['points'])}")

        if 'points' not in inputs:
            points_inputs = self.points_from_file(inputs)
        else:
            raise ValueError('Unsupported input points type: '
                             f"{type(results['points'])}")

        multi_modality_inputs = points_inputs

        box_type_3d, box_mode_3d = tao_structures.get_box_type_tao3d('lidar')

        if isinstance(results['img'], str):
            inputs = {'images': {self.default_cam_key: {'img_path': results['img'], 'cam2img': results['cam2img'],
                                                        'lidar2img': results['lidar2img'], 'lidar2cam': results['lidar2cam']}},
                      'box_mode_3d': box_mode_3d, 'box_type_3d': box_type_3d}
        else:
            raise ValueError('Unsupported input image type: '
                             f"{type(results['img'])}")

        if not isinstance(results['img'], np.ndarray):
            imgs_inputs = self.imgs_from_file(inputs)
        else:
            raise ValueError('Unsupported input image type: '
                             f"{type(results['img'])}")

        multi_modality_inputs.update(imgs_inputs)

        return multi_modality_inputs
