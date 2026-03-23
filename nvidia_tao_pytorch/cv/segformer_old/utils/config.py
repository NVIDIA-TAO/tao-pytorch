# Copyright (c) 2024, NVIDIA CORPORATION.  All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Utils Function"""

import os
import copy
from abc import abstractmethod
from omegaconf import OmegaConf
from nvidia_tao_pytorch.cv.segformer_old.dataloader.data_utils import build_palette, build_target_class_list
from mmengine.logging import print_log

ROOT_DIR = os.getenv("NV_TLT_PYTORCH_TOP", os.getcwd())


class MMSegmentationConfig(object):
    """Segmentation Config Class to convert Hydra config to MMEngine config"""

    def __init__(self,
                 config,
                 phase="train"):
        """Init Function."""
        self.config = OmegaConf.to_container(config, resolve=True)
        self.updated_config = {}
        self.deploy_config = {}
        self.phase = phase
        self.update_config(phase)

    def update_config(self, phase):
        """ Function to update hydra config to mmlab based config"""
        self.update_env()
        self.update_dataset_config()
        self.update_model_config()
        if phase == "train":
            self.update_train_params_config()
            if self.config["train"]["validate"]:
                self.updated_config["val_evaluator"] = {"type": "IoUMetric", "iou_metrics": ["mIoU"]}
                self.updated_config["val_cfg"] = {'type': 'ValLoop'}
        elif phase == "export":
            self.update_onnx_config()
        else:
            self.updated_config["test_evaluator"] = {"type": "IoUMetric", "iou_metrics": ["mIoU"]}
            self.updated_config["test_cfg"] = {'type': 'TestLoop'}

    def update_env(self):
        """Function to update env variables"""
        exp_config = self.config[self.phase]["exp_config"]
        self.updated_config["env_cfg"] = {}
        # The env_cfg has params for the distributed environment
        if 'CUDA_VISIBLE_DEVICES' in os.environ.keys() and len(os.environ['CUDA_VISIBLE_DEVICES'].split(',')) > 1:
            self.updated_config["env_cfg"] = exp_config["env_cfg"]
        self.updated_config["randomness"] = {"seed": exp_config["manual_seed"], "deterministic": exp_config["deterministic"]}
        self.updated_config["default_scope"] = exp_config["default_scope"]
        self.updated_config["log_level"] = exp_config["log_level"]
        self.updated_config["vis_backends"] = [{'type': 'LocalVisBackend'}]
        self.updated_config["visualizer"] = {'type': 'SegLocalVisualizer', 'vis_backends': self.updated_config["vis_backends"], 'name': 'visualizer'}
        self.updated_config["launcher"] = "pytorch"

    @abstractmethod
    def update_dataset_config(self):
        """Update the dataset config"""
        #  Update Dataset config
        dataset_config = self.config["dataset"]

        # Expecting things in terms of dataloaders

        self.updated_config["data_preprocessor"] = {}
        self.updated_config["data_preprocessor"]["size"] = (self.config["model"]["input_height"], self.config["model"]["input_width"])
        self.updated_config["data_preprocessor"]["bgr_to_rgb"] = dataset_config["img_norm_cfg"]["to_rgb"]
        self.updated_config["data_preprocessor"]["mean"] = dataset_config["img_norm_cfg"]["mean"]
        self.updated_config["data_preprocessor"]["std"] = dataset_config["img_norm_cfg"]["std"]
        self.updated_config["data_preprocessor"]["pad_val"] = dataset_config["img_norm_cfg"]["pad_val"]
        self.updated_config["data_preprocessor"]["seg_pad_val"] = dataset_config["img_norm_cfg"]["seg_pad_val"]
        self.updated_config["data_preprocessor"]["size"] = (self.config["model"]["input_height"], self.config["model"]["input_width"])
        self.updated_config["data_preprocessor"]["type"] = dataset_config["img_norm_cfg"]["type"]

        dataloader = {}

        dataloader["batch_size"] = dataset_config["batch_size"]
        dataloader["num_workers"] = dataset_config["workers_per_gpu"]
        dataloader["persistent_workers"] = True
        dataloader["dataset"] = {}
        dataloader["dataset"]["data_root"] = dataset_config["data_root"]
        dataloader["dataset"]["type"] = dataset_config["type"]
        dataloader["dataset"]["img_suffix"] = dataset_config["img_suffix"]
        dataloader["dataset"]["seg_map_suffix"] = dataset_config["seg_map_suffix"]
        dataloader["dataset"]["reduce_zero_label"] = dataset_config["reduce_zero_label"]
        dataloader["dataset"]["data_prefix"] = {}

        # Sets the metainfo param of the dataset to set the classes and palette
        target_classes = build_target_class_list(dataset_config)
        PALETTE, CLASSES, label_map, id_color_map = build_palette(target_classes)

        print_log(f"Palette: {PALETTE}")
        print_log(f"Classes: {CLASSES}")

        self.num_classes = len(set(CLASSES))

        dataloader["dataset"]["metainfo"] = {"classes": tuple(CLASSES), "palette": PALETTE}

        # self.updated_config["data_root"] = dataset_config["data_root"]

        # We don't publicize an augmentation config for val/test, so use params from train that get overwritten by TTA
        img_scale = dataset_config["train_dataset"]["pipeline"]["augmentation_config"]["resize"]["img_scale"]
        if not img_scale:
            img_scale_min = min(self.config["model"]["input_height"], self.config["model"]["input_width"])
            img_scale_max = 1024 if img_scale_min < 1024 else 2048
            img_scale = (img_scale_min, img_scale_max)

        multi_scale = dataset_config["test_dataset"]["pipeline"]["multi_scale"]
        if not multi_scale:
            multi_scale = (self.config["model"]["input_height"], 2048)

        keep_ar = dataset_config["test_dataset"]["pipeline"]["augmentation_config"]["resize"]["keep_ratio"]
        tta_pipeline = [dict(type="LoadImageFromFile"),
                        dict(type="TestTimeAug", transforms=[
                            [dict(type='Resize', scale=tuple(multi_scale), keep_ratio=keep_ar)],
                            [dict(type='RandomFlip', prob=1.), dict(type='RandomFlip', prob=0.)],
                            [dict(type="TAOLoadAnnotations", reduce_zero_label=dataset_config["reduce_zero_label"], input_type=dataset_config["input_type"])],
                            [dict(type='PackSegInputs')]
                        ])
                        ]

        self.updated_config["tta_pipeline"] = tta_pipeline

        if self.phase == "train":
            train_dataloader = copy.deepcopy(dataloader)
            train_dataloader["sampler"] = {'type': 'InfiniteSampler', 'shuffle': True}
            # TODO: @sean should we be supporting multiple training directories?
            train_dataloader["dataset"]["data_prefix"]["img_path"] = dataset_config["train_dataset"]["img_dir"][0]
            train_dataloader["dataset"]["data_prefix"]["seg_map_path"] = dataset_config["train_dataset"]["ann_dir"][0]

            train_pipeline = [dict(type="LoadImageFromFile"),
                              dict(type="TAOLoadAnnotations", reduce_zero_label=dataset_config["reduce_zero_label"], input_type=dataset_config["input_type"]),
                              dict(type="RandomResize", scale=tuple(img_scale),
                                   ratio_range=tuple(dataset_config["train_dataset"]["pipeline"]["augmentation_config"]["resize"]["ratio_range"]),
                                   keep_ratio=dataset_config["train_dataset"]["pipeline"]["augmentation_config"]["resize"]["keep_ratio"]),
                              dict(type="RandomCrop", crop_size=tuple((self.config["model"]["input_height"], self.config["model"]["input_width"])),
                                   cat_max_ratio=dataset_config["train_dataset"]["pipeline"]["augmentation_config"]["random_crop"]["cat_max_ratio"]),
                              dict(type="RandomFlip", prob=dataset_config["train_dataset"]["pipeline"]["augmentation_config"]["random_flip"]["prob"]),
                              dict(type='PhotoMetricDistortion'),
                              dict(type='PackSegInputs'),
                              ]
            train_dataloader["dataset"]["pipeline"] = train_pipeline

            val_dataloader = copy.deepcopy(dataloader)
            val_dataloader["sampler"] = {'type': 'DefaultSampler', 'shuffle': False}
            val_dataloader["dataset"]["data_prefix"]["img_path"] = dataset_config["val_dataset"]["img_dir"]
            val_dataloader["dataset"]["data_prefix"]["seg_map_path"] = dataset_config["val_dataset"]["ann_dir"]

            val_pipeline = [dict(type="LoadImageFromFile"),
                            dict(type="Resize", scale=tuple(img_scale),
                                 keep_ratio=dataset_config["val_dataset"]["pipeline"]["augmentation_config"]["resize"]["keep_ratio"]),
                            dict(type="TAOLoadAnnotations", reduce_zero_label=dataset_config["reduce_zero_label"], input_type=dataset_config["input_type"]),
                            dict(type='PackSegInputs')
                            ]

            val_dataloader["dataset"]["pipeline"] = val_pipeline

            self.updated_config["train_dataloader"] = train_dataloader
            self.updated_config["val_dataloader"] = val_dataloader
            self.updated_config["test_dataloader"] = None

        else:
            test_dataloader = copy.deepcopy(dataloader)
            test_dataloader["sampler"] = {'type': 'DefaultSampler', 'shuffle': False}
            test_dataloader["dataset"]["data_prefix"]["img_path"] = dataset_config["test_dataset"]["img_dir"]
            test_dataloader["dataset"]["data_prefix"]["seg_map_path"] = dataset_config["test_dataset"]["ann_dir"]

            test_pipeline = [dict(type="LoadImageFromFile"),
                             dict(type="Resize", scale=tuple(img_scale),
                                  keep_ratio=dataset_config["test_dataset"]["pipeline"]["augmentation_config"]["resize"]["keep_ratio"]),
                             dict(type="TAOLoadAnnotations", reduce_zero_label=dataset_config["reduce_zero_label"], input_type=dataset_config["input_type"]),
                             dict(type='PackSegInputs')
                             ]

            test_dataloader["dataset"]["pipeline"] = test_pipeline
            # This is for exporting to work
            self.updated_config["test_pipeline"] = test_pipeline

            self.updated_config["train_dataloader"] = None
            self.updated_config["val_dataloader"] = None
            self.updated_config["test_dataloader"] = test_dataloader

    @abstractmethod
    def update_model_config(self):
        """Update the model config"""
        model_config = self.config["model"]
        self.updated_config["model"] = {}
        self.updated_config["model"]["type"] = model_config["type"]
        self.updated_config["model"]["data_preprocessor"] = self.updated_config["data_preprocessor"]
        # self.updated_config["model"]["pretrained"] = model_config["pretrained_model_path"]
        self.updated_config["model"]["backbone"] = model_config["backbone"]
        self.updated_config["model"]["backbone"]["resolution"] = (model_config["input_width"], model_config["input_height"])
        self.updated_config["model"]["backbone"]["init_cfg"]["checkpoint"] = model_config["pretrained_model_path"]
        self.updated_config["model"]["decode_head"] = model_config["decode_head"]
        self.updated_config["model"]["train_cfg"] = {}
        self.updated_config["model"]["test_cfg"] = model_config["test_cfg"]

        channels_map = {"mit_b0": [32, 64, 160, 256],
                        "fan_tiny_8_p4_hybrid": [128, 256, 192, 192],
                        "fan_large_16_p4_hybrid": [128, 256, 480, 480],
                        "fan_small_12_p4_hybrid": [128, 256, 384, 384],
                        "fan_base_16_p4_hybrid": [128, 256, 448, 448],
                        "vit_large_nvdinov2": [1024, 1024, 1024, 1024],
                        "vit_giant_nvdinov2": [1536, 1536, 1536, 1536],
                        "vit_base_nvclip_16_siglip": [768, 768, 768, 768],
                        "vit_huge_nvclip_14_siglip": [1280, 1280, 1280, 1280]}

        if model_config["backbone"]["type"] in channels_map.keys():
            self.updated_config["model"]["decode_head"]["in_channels"] = channels_map[model_config["backbone"]["type"]]

        if "fan" in model_config["backbone"]["type"]:
            self.updated_config["model"]["decode_head"]["channels"] = 256

        self.updated_config["model"]["decode_head"]["num_classes"] = self.num_classes
        self.updated_config["model"]["decode_head"]["img_shape"] = [model_config["input_height"], model_config["input_width"]]
        self.updated_config["model"]["decode_head"]["phase"] = self.phase

        # This is done by recommendation of mmseg
        # It will automatically set threshold = 0.3
        # if self.num_classes == 2:
        #     self.updated_config["model"]["decode_head"]["out_channels"] = 1
        #     self.updated_config["model"]["decode_head"]["loss_decode"]["use_sigmoid"] = True
        # Alternatively, leave as is as defualt args set use_sigmoid = False

        # self.updated_config["norm_cfg"] = model_config["decode_head"]["norm_cfg"]

    def get_updated_optimizer(self, cfg):
        """Get the updated optimizer"""
        optim_wrapper = {"optimizer": {}, "type": "AmpOptimWrapper"}

        optim_wrapper["optimizer"]["lr"] = cfg["sf_optim"]["lr"]
        optim_wrapper["optimizer"]["betas"] = cfg["sf_optim"]["betas"]
        optim_wrapper["optimizer"]["type"] = cfg["sf_optim"]["type"]
        optim_wrapper["optimizer"]["weight_decay"] = cfg["sf_optim"]["weight_decay"]

        optim_wrapper["paramwise_cfg"] = {"custom_keys": cfg["sf_optim"]["paramwise_cfg"]}

        return optim_wrapper

    def get_lr_schedulers(self, old_cfg, new_cfg):
        """Set the learning rate schedulers according to the mmengine spec"""
        linear_lr_config, poly_lr_config = new_cfg

        linear_lr_config["start_factor"] = old_cfg["warmup_ratio"]
        linear_lr_config["end"] = old_cfg["warmup_iters"]
        linear_lr_config["by_epoch"] = old_cfg["by_epoch"]

        poly_lr_config["begin"] = old_cfg["warmup_iters"]
        poly_lr_config["power"] = old_cfg["power"]
        poly_lr_config["eta_min"] = old_cfg["min_lr"]
        poly_lr_config["by_epoch"] = old_cfg["by_epoch"]

        return new_cfg

    def update_train_params_config(self):
        """Update train parameters"""
        #  Update Train Params
        train_param_config = self.config["train"]
        self.updated_config["default_hooks"] = train_param_config["default_hooks"]
        self.updated_config["default_hooks"]["checkpoint"]["interval"] = train_param_config["checkpoint_interval"]
        # self.updated_config["default_hooks"]["logger"]["type"] = "TAOTextLoggerHook"
        self.updated_config["default_hooks"]["logger"]["interval"] = train_param_config["logging_interval"]
        # self.updated_config["auto_scale_lr"] = {"base_batch_size": train_param_config["runner"]["auto_scale_lr_bs"]}
        self.updated_config["train_cfg"] = {'type': 'IterBasedTrainLoop', "max_iters": train_param_config["max_iters"], "val_interval": train_param_config["validation_interval"]}
        self.updated_config["optim_wrapper"] = self.get_updated_optimizer(train_param_config["trainer"])
        self.updated_config["param_scheduler"] = self.get_lr_schedulers(train_param_config["trainer"]["lr_config"], train_param_config["param_scheduler"])
        self.updated_config["load_from"] = train_param_config["resume_training_checkpoint_path"]
        self.updated_config["resume"] = train_param_config["resume"]

        self.updated_config["find_unused_parameters"] = train_param_config["trainer"]["find_unused_parameters"]

    def update_onnx_config(self):
        """Update ONNX parameters for MMDeploy"""
        export_config = self.config[self.phase]
        self.deploy_config["codebase_config"] = export_config["codebase_config"]
        self.deploy_config["backend_config"] = dict(type="onnxruntime")

        self.deploy_config["onnx_config"] = export_config["onnx_config"]
        # self.deploy_config["onnx_config"]["dynamic_axes"]= {'input': {0: 'batch', 2: 'height', 3: 'width'},
        #                                                     'output': {0: 'batch', 2: 'height', 3: 'width'}}
        self.deploy_config["onnx_config"]["dynamic_axes"] = {'input': {0: 'batch'}, 'output': {0: 'batch'}}
        self.deploy_config["onnx_config"]["input_names"] = ['input']
        self.deploy_config["onnx_config"]["output_names"] = ['output']
        self.deploy_config["onnx_config"]["save_file"] = export_config["onnx_file"].split('/')[-1]
        self.deploy_config["onnx_config"]["input_shape"] = (export_config["input_width"], export_config["input_height"])
