# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""RADIO Data Module."""

import os
from typing import Optional, Any, Dict, List
from torch.utils.data import DataLoader, distributed, RandomSampler, BatchSampler
import pytorch_lightning as pl
import torch

from nvidia_tao_pytorch.core.distributed.comm import is_dist_avail_and_initialized
from nvidia_tao_pytorch.multimodal.radio.dataloader.dataset import CLDataset


def _merge_stochastic_resolutions(per_teacher_dicts: List[Any]) -> Optional[Dict[int, float]]:
    """Merge per-teacher stochastic_resolutions: for each resolution take max prob
    across teachers, then renormalize. Returns a single dict or None if no teacher defines it.
    """
    merged: Dict[int, float] = {}
    for src in per_teacher_dicts:
        if src is None or not hasattr(src, 'items'):
            continue
        for res, prob in src.items():
            r = int(res)
            merged[r] = max(merged.get(r, 0.0), float(prob))
    if not merged:
        return None
    total = sum(merged.values())
    if total <= 0:
        return None
    return {r: p / total for r, p in merged.items()}


class RadioDataModule(pl.LightningDataModule):
    """Lightning DataModule for RADIO distillation."""

    def __init__(self, dataset_config, experiment_config: Optional[Any] = None):
        """Lightning DataModule Initialization.

        Args:
            dataset_config: Configuration for the dataset.
            experiment_config: Full experiment config (optional). When
                provided and distillation with multi-view is used, the
                train dataset will be built with the equivariant pipeline.
        """
        super().__init__()
        self.dataset_config = dataset_config
        self.experiment_config = experiment_config
        self.batch_size = dataset_config["batch_size"]
        self.val_batch_size = dataset_config.get("val_batch_size", None) or dataset_config["batch_size"]
        self.num_workers = dataset_config["workers"]
        self.root_dir = dataset_config["root_dir"]
        self.img_size = dataset_config["img_size"]
        self.val_img_size = dataset_config.get("val_img_size", None) or dataset_config["img_size"]
        self.augmentation = dataset_config["augmentation"]
        self.calib_dataset = None

        self._train_loader = None
        self._shared_epoch = None
        self._loader_state = None

        self._val_eval_loader = None
        self._val_train_split_loader = None

    def _build_wds_pipeline(self):
        """Build the equivariant WebDataset pipeline from experiment config.

        Translates TAO config fields into ``get_data_pipeline()`` arguments
        and returns the fully assembled loader, shared epoch, and loader
        state for checkpointing.
        """
        from nvidia_tao_pytorch.multimodal.radio.dataloader.data_pipeline import (
            PipelineConfig,
            get_data_pipeline,
        )
        from nvidia_tao_pytorch.multimodal.radio.dataloader.stages.utils import (
            seed_from_tuple,
        )

        # TODO(heslami): Move this to the dataset config with defaults values close to config def.
        train_cfg = self.dataset_config["train_dataset"]
        tar_sources = train_cfg["tar_data_sources"]

        ds_listing = [
            (src.get("root_dir", src.get("root")), src.get("scale_factor", 1.0))
            for src in tar_sources
        ]

        steps_per_epoch = max(
            src.get("steps_per_epoch", 2000) for src in tar_sources
        )

        pipeline_config = PipelineConfig(
            steps_per_epoch=steps_per_epoch,
            workers=self.num_workers,
        )

        student_size = self.img_size
        student_patch_size = int(self.augmentation.get("patch_size", 16))

        input_sizes = [student_size]
        patch_sizes = [student_patch_size]
        upsample_factors = []
        stochastic_teachers = []
        per_teacher_stochastic = []

        distill = getattr(self.experiment_config, "distill", None)
        teachers = []
        if distill is not None:
            teachers = getattr(distill, "teacher", [])
            if teachers and not hasattr(teachers, '__len__'):
                teachers = [teachers]

        for t in teachers:
            match_student = getattr(t, "match_student_resolution", True)
            teacher_input = getattr(t, "input_size", student_size)
            if match_student:
                input_sizes.append(student_size)
            else:
                input_sizes.append(teacher_input)
            patch_sizes.append(getattr(t, "patch_size", student_patch_size))
            upsample_factors.append(getattr(t, "upsample_factor", 1))
            stochastic_teachers.append(match_student)
            per_teacher_stochastic.append(
                getattr(t, "stochastic_resolutions", None)
            )

        stochastic_size_args = None
        merged_res = _merge_stochastic_resolutions(per_teacher_stochastic)
        if merged_res is not None:
            stochastic_size_args = {
                "resolutions": merged_res,
                "fixed_aspect": True,
            }

        base_seed = train_cfg.get("seed", 42)
        rank = torch.distributed.get_rank() if torch.distributed.is_initialized() else 0
        seed = seed_from_tuple(base_seed, 0, rank, 0)

        full_equivariance = train_cfg.get("full_equivariance", False)
        shift_equivariance = train_cfg.get("shift_equivariance", False)
        data_weight_mode = train_cfg.get("data_weight_mode", "inv_frequency")
        prefetch = train_cfg.get("prefetch", True)
        include_keys = train_cfg.get("include_keys", False)
        include_dataset_source = train_cfg.get("include_dataset_source", False)
        native_resolution_filter = train_cfg.get("native_resolution_filter", None)

        loader, shared_epoch, loader_state = get_data_pipeline(
            args=pipeline_config,
            ds_listing=ds_listing,
            input_sizes=input_sizes,
            patch_sizes=patch_sizes,
            batch_size=self.batch_size,
            is_train=True,
            epoch=0,
            seed=seed,
            upsample_factors=upsample_factors if upsample_factors else None,
            data_weight_mode=data_weight_mode,
            prefetch=prefetch,
            full_equivariance=full_equivariance,
            shift_equivariance=shift_equivariance,
            stochastic_size_args=stochastic_size_args,
            stochastic_teachers=stochastic_teachers if stochastic_teachers else None,
            include_keys=include_keys,
            include_dataset_source=include_dataset_source,
            aug_config=self.augmentation,
            native_resolution_filter=native_resolution_filter,
        )

        return loader, shared_epoch, loader_state

    def _build_wds_val_loader(self, split_name):
        """Build a single WDS evaluation loader for a given split.

        Uses ``get_data_pipeline(is_train=False)`` with a single student
        view (no teachers, no stochastic resolution, no equivariance).

        Args:
            split_name: ``"train"`` or ``"val"`` — appended to each
                tar root to form the shard directory path.

        Returns:
            A DataLoader over the requested split.
        """
        from nvidia_tao_pytorch.multimodal.radio.dataloader.data_pipeline import (
            PipelineConfig,
            get_data_pipeline,
        )
        from nvidia_tao_pytorch.multimodal.radio.dataloader.stages.utils import (
            extract_dict_field,
        )

        val_cfg = self.dataset_config["val_dataset"]
        tar_sources = val_cfg.get("tar_data_sources", [])

        student_patch_size = int(self.augmentation.get("patch_size", 16))

        pipeline_config = PipelineConfig(
            steps_per_epoch=0,
            workers=self.num_workers,
        )

        label_extractor = extract_dict_field(
            src_path="json.id", dest_path="id"
        )

        ds_listing = []
        for src in tar_sources:
            root = src.get("root_dir", "")
            ds_listing.append((f"{root}/{split_name}", 1.0))

        loader, _, _ = get_data_pipeline(
            args=pipeline_config,
            ds_listing=ds_listing,
            input_sizes=[self.val_img_size],
            patch_sizes=[student_patch_size],
            batch_size=self.val_batch_size,
            is_train=False,
            epoch=0,
            seed=42,
            prefetch=True,
            label_extractor=label_extractor,
            label_key="id",
        )

        return loader

    def setup(self, stage: Optional[str] = None):
        """Setup the dataset.

        Args:
            stage: Stage of the dataset.
        """
        is_distributed = is_dist_avail_and_initialized()

        if stage == "fit" or stage is None:
            train_cfg = self.dataset_config["train_dataset"]
            tar_sources = train_cfg.get("tar_data_sources", [])
            has_tar = len(tar_sources) > 0

            if has_tar:
                self._train_loader, self._shared_epoch, self._loader_state = \
                    self._build_wds_pipeline()
                self.dataset = "WebDataset"
                self.train_sampler = None
            else:
                self.train_dataset = CLDataset(
                    root_dir=self.root_dir,
                    augmentation=self.augmentation,
                    split="train",
                    img_size=self.img_size,
                    to_tensor=True,
                    data_path=train_cfg["images_dir"],
                    nolabel_folder=self.dataset_config["train_nolabel"]["folder_path"],
                )
                self.dataset = "CLDataset"
                if is_distributed:
                    self.train_sampler = distributed.DistributedSampler(
                        self.train_dataset, shuffle=True
                    )
                else:
                    self.train_sampler = RandomSampler(self.train_dataset)

            val_cfg = self.dataset_config["val_dataset"]
            val_tar_sources = val_cfg.get("tar_data_sources", [])
            knn_enabled = self.dataset_config.get("knn_validation", False)

            if len(val_tar_sources) > 0:
                self._val_eval_loader = self._build_wds_val_loader("val")
                if knn_enabled:
                    self._val_train_split_loader = self._build_wds_val_loader("train")
                self.val_dataset_type = "WebDataset"
                self.val_dataset = None
            else:
                val_images_dir = val_cfg.get("images_dir", "")
                if not val_images_dir:
                    raise ValueError(
                        "val_dataset.images_dir must be set when not using tar_data_sources."
                    )
                self.val_dataset = CLDataset(
                    root_dir=self.root_dir,
                    augmentation=self.augmentation,
                    split="val",
                    img_size=self.val_img_size,
                    to_tensor=True,
                    data_path=val_images_dir,
                )
                if knn_enabled:
                    train_images_dir = os.path.join(
                        os.path.dirname(val_images_dir), "train",
                    )
                    knn_train_dataset = CLDataset(
                        root_dir=self.root_dir,
                        augmentation=self.augmentation,
                        split="val",
                        img_size=self.val_img_size,
                        to_tensor=True,
                        data_path=train_images_dir,
                    )
                    self._val_train_split_loader = DataLoader(
                        knn_train_dataset,
                        num_workers=self.num_workers,
                        batch_size=self.val_batch_size,
                        shuffle=False,
                        collate_fn=knn_train_dataset.collate_fn,
                        pin_memory=True,
                    )
                self.val_dataset_type = "CLDataset"

        if stage == "test" or stage is None:
            if self.dataset == "CLDataset":
                self.test_dataset = CLDataset(
                    root_dir=self.root_dir,
                    augmentation=self.augmentation,
                    split="val",
                    img_size=self.img_size,
                    to_tensor=True,
                    data_path=self.dataset_config["val_dataset"]["images_dir"],
                )
            else:
                raise NotImplementedError(
                    "Wrong dataset name %s (choose one from [CLDataset,])"
                    % self.dataset
                )

        if stage == "predict" or stage is None:
            if self.dataset == "CLDataset":
                self.predict_dataset = CLDataset(
                    root_dir=self.root_dir,
                    augmentation=self.augmentation,
                    split="test",
                    img_size=self.img_size,
                    to_tensor=True,
                    data_path=self.dataset_config["test_dataset"]["images_dir"],
                )
            else:
                raise NotImplementedError(
                    "Wrong dataset name %s (choose one from [CLDataset,])"
                    % self.dataset
                )

        if stage == "calibration" or stage is None:
            calib_cfg = self.dataset_config.get("quant_calibration_dataset", {})
            calib_images_dir = calib_cfg.get("images_dir", "") if hasattr(calib_cfg, 'get') else getattr(calib_cfg, "images_dir", "")
            if calib_images_dir:
                self.calib_dataset = CLDataset(
                    root_dir=self.root_dir,
                    augmentation=self.augmentation,
                    split="val",
                    img_size=self.img_size,
                    to_tensor=True,
                    data_path=calib_images_dir,
                )
            else:
                raise ValueError("quant_calibration_dataset.images_dir must be provided for calibration stage.")

    def train_dataloader(self):
        """Build the dataloader for training.

        Returns:
            train_loader: PyTorch DataLoader used for training.
        """
        if self.dataset == "WebDataset":
            return self._train_loader

        dataloader_kwargs = {
            "num_workers": self.num_workers,
            "pin_memory": True,
            "persistent_workers": True,
            "drop_last": False,
        }
        dataloader_kwargs["batch_sampler"] = BatchSampler(
            self.train_sampler, self.batch_size, drop_last=True
        )
        dataloader_kwargs["collate_fn"] = self.train_dataset.collate_fn
        train_loader = DataLoader(self.train_dataset, **dataloader_kwargs)
        return train_loader

    def val_dataloader(self):
        """Build the dataloader for validation."""
        if self.val_dataset_type == "WebDataset":
            return self._val_eval_loader

        val_loader = DataLoader(
            self.val_dataset,
            num_workers=self.num_workers,
            batch_size=self.val_batch_size,
            shuffle=False,
            collate_fn=self.val_dataset.collate_fn,
            pin_memory=True,
        )
        return val_loader

    def test_dataloader(self):
        """Build the dataloader for evaluation.

        Returns:
            test_loader: PyTorch DataLoader used for evaluation.
        """
        test_loader = DataLoader(
            self.test_dataset,
            num_workers=self.num_workers,
            batch_size=self.batch_size,
            shuffle=False,
            pin_memory=False,
        )
        return test_loader

    def predict_dataloader(self):
        """Build the dataloader for inference.

        Returns:
            predict_loader: PyTorch DataLoader used for inference.
        """
        predict_loader = DataLoader(
            self.predict_dataset,
            num_workers=self.num_workers,
            batch_size=self.batch_size,
            shuffle=False,
            pin_memory=False,
        )
        return predict_loader

    def calib_dataloader(self):
        """Build the dataloader for quantization calibration."""
        if self.calib_dataset is None:
            raise ValueError("Calibration dataset is not initialized. Please ensure quant_calibration_dataset.images_dir is set in the config.")
        calib_loader = DataLoader(
            self.calib_dataset,
            num_workers=self.num_workers,
            batch_size=self.batch_size,
            shuffle=False,
            pin_memory=False,
            collate_fn=self.calib_dataset.collate_fn,
        )
        return calib_loader

    def set_epoch(self, epoch):
        """Set the current epoch for the dataset.

        For the WebDataset pipeline, updates the shared epoch counter
        used by ``MultiStreamShuffle`` for deterministic shard ordering.
        For regular datasets this is a no-op (Lightning handles epoch
        setting via the sampler).

        Args:
            epoch: The current epoch number.
        """
        if self._shared_epoch is not None:
            self._shared_epoch.set_value(epoch)

    @property
    def loader_state(self):
        """Access the ``LoaderState`` for checkpointing.

        Returns ``None`` when not using the equivariant WebDataset pipeline.
        """
        return self._loader_state

    @property
    def val_eval_loader(self):
        """Loader over the val split when using WDS validation.

        Returns ``None`` when validation uses CLDataset.
        """
        return self._val_eval_loader

    @property
    def val_train_split_loader(self):
        """Loader over the train split for building the KNN embedding index.

        Available when ``knn_validation`` is enabled (for both WDS and
        CLDataset). Returns ``None`` otherwise.
        """
        return self._val_train_split_loader
