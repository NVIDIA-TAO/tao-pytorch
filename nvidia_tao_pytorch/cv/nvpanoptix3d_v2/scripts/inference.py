# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""NVPanoptix3Dv2 inference script.

Variant-agnostic: the panoptic variant writes per-sample panoptic maps, the
reasoning variant writes per-``[SEG]`` masks and fused point clouds.
"""

import json
import os

from pytorch_lightning import Trainer

from nvidia_tao_pytorch.core.decorators.workflow import monitor_status
from nvidia_tao_pytorch.core.hydra.hydra_runner import hydra_runner
from nvidia_tao_pytorch.core.initialize_experiments import initialize_inference_experiment
from nvidia_tao_pytorch.core.tlt_logging import logging

# The NVPanoptix3Dv2 config, dataloader, model, and export packages are
# delivered by companion patches in this feature series, so they may not be
# present on disk yet. Import them defensively -- mirroring the deferred-import
# convention already used elsewhere in this series -- so this script stays
# importable (and statically checkable) in the meantime. Each symbol falls back
# to a placeholder that raises a descriptive ImportError the moment it is
# actually used, rather than failing later with an opaque always-False
# comparison or ``NoneType`` error far from the real cause. Once the companion
# packages land the ``try`` branch simply succeeds and behavior is unchanged.
try:
    from nvidia_tao_pytorch.config.nvpanoptix3d_v2.default_config import ExperimentConfig
    from nvidia_tao_pytorch.cv.nvpanoptix3d_v2.dataloader import build_pl_data_module
    from nvidia_tao_pytorch.cv.nvpanoptix3d_v2.model.build_pl_model import PANOPTIC, load_pl_model
except ImportError as _nvp2_import_error:  # pragma: no cover - only before the companion packages land
    class _MissingNVPanoptix3Dv2Symbol:
        """Placeholder for a symbol whose module has not landed yet.

        Calling the placeholder or comparing it against another value raises a
        descriptive ImportError naming the symbol and its module. ``__repr__``
        stays safe so tracebacks and debuggers can render it without a
        secondary failure.
        """

        def __init__(self, symbol, module, cause):
            """Record the symbol name, its owning module, and the original error."""
            self._symbol = symbol
            self._module = module
            self._cause = cause

        def _raise(self, *_args, **_kwargs):
            """Raise a descriptive ImportError naming the missing symbol."""
            raise ImportError(
                f"'{self._symbol}' requires the '{self._module}' module, which is not "
                f"available in this installation. Original import error: {self._cause}"
            ) from self._cause

        __call__ = _raise
        __eq__ = _raise
        __ne__ = _raise

        def __repr__(self):
            """Render safely so tracebacks do not raise a second error."""
            return f"<missing symbol {self._symbol!r} from {self._module!r}>"

    _CONFIG_MODULE = "nvidia_tao_pytorch.config.nvpanoptix3d_v2.default_config"
    _DATALOADER_MODULE = "nvidia_tao_pytorch.cv.nvpanoptix3d_v2.dataloader"
    _MODEL_MODULE = "nvidia_tao_pytorch.cv.nvpanoptix3d_v2.model.build_pl_model"

    ExperimentConfig = _MissingNVPanoptix3Dv2Symbol("ExperimentConfig", _CONFIG_MODULE, _nvp2_import_error)
    build_pl_data_module = _MissingNVPanoptix3Dv2Symbol("build_pl_data_module", _DATALOADER_MODULE, _nvp2_import_error)
    PANOPTIC = _MissingNVPanoptix3Dv2Symbol("PANOPTIC", _MODEL_MODULE, _nvp2_import_error)
    load_pl_model = _MissingNVPanoptix3Dv2Symbol("load_pl_model", _MODEL_MODULE, _nvp2_import_error)


def run_experiment(experiment_config):
    """Start the inference."""
    model_path, trainer_kwargs = initialize_inference_experiment(experiment_config)

    pl_data = build_pl_data_module(experiment_config)
    pl_model = load_pl_model(experiment_config, model_path, strict=False)

    if str(experiment_config.model.model_type) == PANOPTIC:
        # Deferred import: the panoptic data module ships with the companion
        # dataloader patch (see the module-level note above).
        try:
            from nvidia_tao_pytorch.cv.nvpanoptix3d_v2.dataloader.panoptic.pl_data_module import (
                resolve_vocabulary,
            )
        except ImportError as err:  # pragma: no cover - only before the companion packages land
            raise ImportError(
                "'resolve_vocabulary' requires the 'nvidia_tao_pytorch.cv.nvpanoptix3d_v2.dataloader.panoptic.pl_data_module' "
                "module, which is not available in this installation."
            ) from err
        vocabulary = resolve_vocabulary(experiment_config.dataset.panoptic)

        # An explicit categories JSON replaces the dataset taxonomy for both the
        # text prompts and the category IDs the predicted segments index into.
        categories_json = experiment_config.inference.categories_json
        if categories_json:
            with open(categories_json, "r", encoding="utf-8") as handle:
                categories = json.load(handle)
            classes = [category["name"] for category in categories]
            vocabulary.update(classes=classes, categories=categories)
            logging.info(f"Predicting against {len(classes)} classes from {categories_json}")
        pl_model.set_classes(**vocabulary)

    trainer_kwargs["use_distributed_sampler"] = False
    trainer_kwargs["enable_checkpointing"] = False
    trainer = Trainer(**trainer_kwargs)
    trainer.predict(pl_model, pl_data, return_predictions=False)


spec_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


@hydra_runner(
    config_path=os.path.join(spec_root, "experiment_specs"),
    config_name="spec_panoptic", schema=ExperimentConfig
)
@monitor_status(name="NVPanoptix3Dv2", mode="inference")
def main(cfg: ExperimentConfig) -> None:
    """Run the inference process."""
    run_experiment(experiment_config=cfg)


if __name__ == "__main__":
    main()
