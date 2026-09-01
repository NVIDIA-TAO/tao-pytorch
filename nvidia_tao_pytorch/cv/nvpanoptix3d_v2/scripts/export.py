# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Export the NVPanoptix3Dv2 panoptic model to ONNX.

Only the panoptic variant is exportable; the reasoning variant decodes text
autoregressively and needs an LLM serving stack, not a single graph.

The vocabulary is baked into the graph, so it is resolved here from
``export.categories_json`` when set -- preferred, since it needs no dataset
-- and otherwise from the dataset taxonomy, which requires the preprocessed
roots in ``dataset.panoptic`` to be readable.
"""

import json
import os

import torch

from nvidia_tao_pytorch.core.decorators.workflow import monitor_status
from nvidia_tao_pytorch.core.hydra.hydra_runner import hydra_runner
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
    from nvidia_tao_pytorch.cv.nvpanoptix3d_v2.model.build_pl_model import PANOPTIC, load_pl_model
    from nvidia_tao_pytorch.cv.nvpanoptix3d_v2.export.onnx_exporter import (
        INPUT_NAME,
        NVPanoptix3Dv2PanopticExportWrapper,
        ONNXExporter,
        encode_vocabulary,
        export_safe_metric_scale_head,
        write_vocabulary_sidecar,
    )
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
    _MODEL_MODULE = "nvidia_tao_pytorch.cv.nvpanoptix3d_v2.model.build_pl_model"
    _EXPORT_MODULE = "nvidia_tao_pytorch.cv.nvpanoptix3d_v2.export.onnx_exporter"

    ExperimentConfig = _MissingNVPanoptix3Dv2Symbol("ExperimentConfig", _CONFIG_MODULE, _nvp2_import_error)
    PANOPTIC = _MissingNVPanoptix3Dv2Symbol("PANOPTIC", _MODEL_MODULE, _nvp2_import_error)
    load_pl_model = _MissingNVPanoptix3Dv2Symbol("load_pl_model", _MODEL_MODULE, _nvp2_import_error)
    INPUT_NAME = _MissingNVPanoptix3Dv2Symbol("INPUT_NAME", _EXPORT_MODULE, _nvp2_import_error)
    NVPanoptix3Dv2PanopticExportWrapper = _MissingNVPanoptix3Dv2Symbol("NVPanoptix3Dv2PanopticExportWrapper", _EXPORT_MODULE, _nvp2_import_error)
    ONNXExporter = _MissingNVPanoptix3Dv2Symbol("ONNXExporter", _EXPORT_MODULE, _nvp2_import_error)
    encode_vocabulary = _MissingNVPanoptix3Dv2Symbol("encode_vocabulary", _EXPORT_MODULE, _nvp2_import_error)
    export_safe_metric_scale_head = _MissingNVPanoptix3Dv2Symbol("export_safe_metric_scale_head", _EXPORT_MODULE, _nvp2_import_error)
    write_vocabulary_sidecar = _MissingNVPanoptix3Dv2Symbol("write_vocabulary_sidecar", _EXPORT_MODULE, _nvp2_import_error)


def resolve_export_vocabulary(experiment_config):
    """Resolve the class vocabulary to bake into the exported graph.

    Args:
        experiment_config: The experiment config.

    Returns:
        Tuple of (classes, categories). ``categories`` is None when the source
        carries no category metadata.
    """
    categories_json = experiment_config.export.categories_json
    if categories_json:
        with open(categories_json, "r", encoding="utf-8") as handle:
            categories = json.load(handle)
        classes = [category["name"] for category in categories]
        logging.info(f"Exporting against {len(classes)} classes from {categories_json}")
        return classes, categories

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

    classes = list(vocabulary["classes"])
    categories = vocabulary.get("categories")
    logging.info(f"Exporting against {len(classes)} classes from the dataset taxonomy")
    return classes, categories


def run_export(experiment_config):
    """Export the panoptic model to ONNX.

    Args:
        experiment_config: The experiment config.

    Returns:
        No explicit returns.
    """
    model_type = str(experiment_config.model.model_type)
    if model_type != PANOPTIC:
        raise ValueError(
            f"ONNX export supports model.model_type='{PANOPTIC}' only, got '{model_type}'. "
            "The reasoning variant decodes text autoregressively and is not a "
            "single feed-forward graph."
        )

    export_config = experiment_config.export
    on_cpu = export_config.on_cpu
    if not on_cpu:
        torch.cuda.set_device(export_config.gpu_id)
    device = torch.device("cpu" if on_cpu else "cuda")

    model_path = export_config.checkpoint
    if not model_path:
        raise ValueError("export.checkpoint must point at the checkpoint to export.")
    input_height = export_config.input_height
    input_width = export_config.input_width
    num_views = export_config.num_views
    opset_version = export_config.opset_version
    batch_size = export_config.batch_size
    traced_batch_size = 1 if batch_size in (None, -1) else batch_size

    patch_size = experiment_config.model.patch_size
    if input_height % patch_size or input_width % patch_size:
        raise ValueError(
            f"export.input_height/input_width ({input_height}x{input_width}) must both be "
            f"divisible by model.patch_size={patch_size}."
        )

    output_file = export_config.onnx_file
    if not output_file:
        output_file = f"{os.path.splitext(model_path)[0]}.onnx"
    # Never silently overwrite: the weights land in external-data files beside
    # the graph, so a partial overwrite leaves an unloadable pair behind.
    if os.path.exists(output_file):
        raise FileExistsError(f"ONNX file {output_file} already exists; remove it or set export.onnx_file")
    os.makedirs(os.path.dirname(os.path.realpath(output_file)), exist_ok=True)

    classes, categories = resolve_export_vocabulary(experiment_config)

    pl_model = load_pl_model(experiment_config, model_path, strict=False)
    pl_model.to(device)
    pl_model.eval()

    # Encode the vocabulary before wrapping: this is the last point at which
    # the real SigLIP text encoder is still attached.
    class_embeddings = encode_vocabulary(pl_model, classes, categories)

    dummy_input = torch.rand(
        traced_batch_size, num_views, 3, input_height, input_width, device=device,
    )

    with export_safe_metric_scale_head():
        # Constructing the wrapper freezes the text encoder and then probes the
        # model once to learn which outputs this build actually produces.
        wrapper = NVPanoptix3Dv2PanopticExportWrapper(
            pl_model.model.eval(),
            class_embeddings.to(device),
            probe_input=dummy_input,
        ).to(device).eval()

        logging.info(
            "Tracing %s with input [%d, %d, 3, %d, %d] -> outputs: %s",
            INPUT_NAME, traced_batch_size, num_views, input_height, input_width,
            ", ".join(wrapper.output_names),
        )

        exporter = ONNXExporter(opset_version=opset_version)
        exporter.export_model(
            wrapper,
            output_file,
            dummy_input,
            input_names=[INPUT_NAME],
            output_names=wrapper.output_names,
            batch_size=batch_size,
            verbose=export_config.verbose,
        )

    exporter.check_onnx(output_file)
    sidecar = write_vocabulary_sidecar(output_file, classes, categories)

    if batch_size in (None, -1):
        logging.info("Exported with a dynamic batch axis; the %d-view axis is static.", num_views)
    else:
        logging.info("Exported with static shapes (batch=%d, views=%d).", batch_size, num_views)
    logging.info("ONNX file stored at %s", output_file)
    logging.info("Class vocabulary stored at %s", sidecar)


spec_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


@hydra_runner(
    config_path=os.path.join(spec_root, "experiment_specs"),
    config_name="spec_panoptic", schema=ExperimentConfig
)
@monitor_status(name="NVPanoptix3Dv2", mode="export")
def main(cfg: ExperimentConfig) -> None:
    """Run the export process."""
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cuda.matmul.allow_tf32 = False
    run_export(cfg)


if __name__ == "__main__":
    main()
