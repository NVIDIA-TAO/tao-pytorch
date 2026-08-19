# SPDX-FileCopyrightText: Copyright (c) 2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Train Grounding DINO model."""

import os

import torch

from pytorch_lightning import Trainer

from nvidia_tao_pytorch.core.decorators.workflow import monitor_status
from nvidia_tao_pytorch.core.hydra.hydra_runner import hydra_runner
from nvidia_tao_pytorch.core.tlt_logging import logging
from nvidia_tao_pytorch.core.initialize_experiments import initialize_train_experiment
from nvidia_tao_pytorch.core.utils.ptm_utils import load_pretrained_weights

from nvidia_tao_pytorch.config.grounding_dino.default_config import ExperimentConfig
from nvidia_tao_pytorch.cv.grounding_dino.dataloader.pl_odvg_data_module import ODVGDataModule
from nvidia_tao_pytorch.cv.grounding_dino.model.pl_gdino_model import GDINOPlModel
from nvidia_tao_pytorch.cv.grounding_dino.model.utils import grounding_dino_parser, ptm_adapter


def _configure_gemm_workarounds_for_device(precision):
    """Work around the cuBLASLt SM 12.x batched-GEMM defect (RTX PRO 6000 Blackwell and siblings).

    cuBLASLt's ``nvjet_sm120_*_tmaAB_*`` TMA GEMM kernels make illegal memory accesses on SM 12.x
    for the skinny-N strided-batched GEMMs this model issues (text encoder / text cross-attention
    and the vision-text fusion layer). The fault happens inside the library, poisons the CUDA
    context, and surfaces as ``CUBLAS_STATUS_INTERNAL_ERROR`` followed by
    ``cudaErrorIllegalAddress``, usually with a traceback pointing at unrelated code.

    One kernel in that family is selected per compute type, so the defect is not tied to a single
    precision:

    ==========  =============================================  ===========================
    Precision   Faulting kernel                                cuBLASLt entry point
    ==========  =============================================  ===========================
    fp16        ``nvjet_sm120_hsh_mma_16x128x32_...``          ``cublasLtHSHMatmul``
    bf16        ``nvjet_sm120_tst_mma_128x8x32_...``           ``cublasLtTSTMatmul``
    fp32/TF32   ``nvjet_sm120_sss_tf32_mma_32x64x32_...``      ``cublasLtSSSMatmul``
    ==========  =============================================  ===========================

    The fp32 variant is only reachable because this container's PyTorch build enables TF32 matmul
    by default, so plain fp32 training still lands on a tensor-core kernel. Turning TF32 off for
    matmul routes fp32 onto an unaffected kernel and is enough to make fp32 training work; it does
    not help fp16/bf16, whose kernels are selected by their own compute type, so those stay gated.

    Datacenter Blackwell (SM 10.x) is unaffected and is not touched here. Remove all of this once
    the container ships a cuBLAS with the fix.
    """
    if not torch.cuda.is_available():
        return

    props = torch.cuda.get_device_properties(0)
    if props.major != 12:
        return

    if precision != '32-true':
        raise RuntimeError(
            f"Mixed-precision training (train.precision={precision}) is not supported on "
            f"{props.name} (SM {props.major}{props.minor}). The cuBLAS shipped in this container "
            f"has a defect in its SM 12.x batched-GEMM kernels that corrupts the CUDA context and "
            f"crashes training with 'CUBLAS_STATUS_INTERNAL_ERROR' / 'illegal memory access'. "
            f"Set 'train.precision: fp32' in your spec to train on this GPU "
            f"(you may need to reduce dataset.batch_size to fit). ONNX export is unaffected."
        )

    # fp32 on its own is not sufficient: with TF32 matmul enabled (the default in this container)
    # fp32 still dispatches to the faulting tensor-core kernel. Guarded so the log line is emitted
    # once even though this runs both before and after the FSDP precision override.
    if torch.backends.cuda.matmul.allow_tf32:
        torch.backends.cuda.matmul.allow_tf32 = False
        logging.info(
            "Disabling TF32 matmul on %s (SM %d%d): the cuBLAS in this container has a defect in "
            "its SM 12.x TF32 batched-GEMM kernel that crashes training. This costs some "
            "throughput and is removed once a fixed cuBLAS ships.",
            props.name, props.major, props.minor
        )


def run_experiment(experiment_config):
    """Start the training."""
    if experiment_config.train.precision.lower() == 'fp16':
        precision = '16-mixed'
    elif experiment_config.train.precision.lower() == 'bf16':
        precision = 'bf16-mixed'
    elif experiment_config.train.precision.lower() == 'fp32':
        precision = '32-true'
    else:
        raise NotImplementedError(f"{experiment_config.train.precision} is not supported. \
                                  Only bf16, fp16, and fp32 are supported")

    # Resolved up front so an unsupported precision fails before the model and dataloaders are
    # built, rather than minutes later.
    _configure_gemm_workarounds_for_device(precision)

    resume_ckpt, trainer_kwargs = initialize_train_experiment(experiment_config)

    dm = ODVGDataModule(experiment_config.dataset)
    dm.setup(stage="fit")
    cap_lists = dm.val_dataset.cap_lists

    # find_unuser_parameters=False and activation_checkpoint combination
    # requires every output in forward function to participate in
    # loss calculation. When return_interm_indices < 4, we must disable
    # activation checkpointing
    if experiment_config.train.activation_checkpoint and \
        len(experiment_config.model.return_interm_indices) < 4 and \
            experiment_config.train.num_gpus > 1:
        experiment_config.train.activation_checkpoint = False
        logging.info("Disabling  activation checkpointing since model is smaller")

    activation_checkpoint = experiment_config.train.activation_checkpoint

    # Load pretrained model as starting point if pretrained path is provided,
    pretrained_path = experiment_config.train.pretrained_model_path
    if pretrained_path:
        # Ignore backbone weights if we get pretrained path for the entire detector
        experiment_config.model.pretrained_backbone_path = None
        pt_model = GDINOPlModel(experiment_config, cap_lists=cap_lists)
        current_model_dict = pt_model.model.state_dict()
        checkpoint = load_pretrained_weights(
            pretrained_path,
            parser=grounding_dino_parser,
            ptm_adapter=ptm_adapter
        )
        new_checkpoint = {}
        for k in sorted(current_model_dict.keys()):
            # Handle PTL format
            v = checkpoint.get(k, None)
            if v is not None:
                if v.size() == current_model_dict[k].size():
                    new_checkpoint[k] = v
                else:
                    # Skip layers that mismatch
                    logging.warning(
                        "skip layer: %s, checkpoint layer size: %s, current model layer size: %s",
                        k, list(v.size()), list(current_model_dict[k].size())
                    )
                    new_checkpoint[k] = current_model_dict[k]
            else:
                logging.warning("skip layer %s as it doesn't exist in the checkpoint", k)
        # Load pretrained weights
        m = pt_model.model.load_state_dict(new_checkpoint, strict=False)
        logging.info("Loading pretrained weights from %s \nm: %s", pretrained_path, m)
    else:
        pt_model = GDINOPlModel(experiment_config, cap_lists=cap_lists)

    num_nodes = experiment_config.train.num_nodes
    clip_grad_norm = experiment_config.train.clip_grad_norm
    is_dry_run = experiment_config.train.is_dry_run
    distributed_strategy = experiment_config.train.distributed_strategy

    strategy = 'auto'
    if len(trainer_kwargs['devices']) > 1:
        # By default find_unused_parameters is set to False in Lightning
        # If true, it introduces extra overhead and can't work with activation checkpointing
        if distributed_strategy.lower() == "ddp" and activation_checkpoint:
            strategy = 'ddp'
        elif distributed_strategy.lower() == "ddp" and not activation_checkpoint:
            strategy = 'ddp_find_unused_parameters_true'
        elif distributed_strategy.lower() == "fsdp":
            strategy = 'fsdp'
            # Override to FP16 for fsdp as there's an error with FP32 during Positional Embedding forward pass
            logging.info("Overriding Precision to FP16 for fsdp")
            precision = '16-mixed'
        else:
            raise NotImplementedError(f"{distributed_strategy} is not implemented. Only ddp and fsdp are supported")

    # Checked after the FSDP branch above, which overrides precision to 16-mixed regardless of
    # what the user configured -- so the guard has to see the precision actually handed to the Trainer.
    _configure_gemm_workarounds_for_device(precision)

    trainer = Trainer(**trainer_kwargs,
                      num_nodes=num_nodes,
                      strategy=strategy,
                      precision=precision,
                      gradient_clip_val=clip_grad_norm,
                      use_distributed_sampler=False,
                      fast_dev_run=is_dry_run)

    trainer.fit(pt_model, dm, ckpt_path=resume_ckpt)


spec_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


# Load experiment specification, additially using schema for validation/retrieving the default values.
# --config_path and --config_name will be provided by the entrypoint script.
@hydra_runner(
    config_path=os.path.join(spec_root, "experiment_specs"), config_name="train", schema=ExperimentConfig
)
@monitor_status(name="Grounding DINO", mode="train")
def main(cfg: ExperimentConfig) -> None:
    """Run the training process."""
    run_experiment(experiment_config=cfg)


if __name__ == "__main__":
    main()
