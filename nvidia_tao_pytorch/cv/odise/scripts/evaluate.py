# SPDX-FileCopyrightText: Copyright (c) 2023 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""ODISE evaluation script."""
import argparse
import logging
import os.path as osp
from contextlib import ExitStack
from iopath.common.s3 import S3PathHandler
from omegaconf import OmegaConf
from typing import MutableSequence

from detectron2.config import LazyConfig, instantiate
from detectron2.engine import create_ddp_model, default_argument_parser, hooks, launch
from detectron2.evaluation import print_csv_format
from detectron2.utils import comm
from detectron2.utils.events import JSONWriter
from detectron2.utils.file_io import PathManager
from detectron2.utils.logger import setup_logger

from nvidia_tao_pytorch.core.hydra.hydra_runner import hydra_runner
from nvidia_tao_pytorch.cv.odise.checkpoint import ODISECheckpointer
from nvidia_tao_pytorch.cv.odise.config import auto_scale_workers, instantiate_odise
from nvidia_tao_pytorch.cv.odise.config.default_config import ExperimentConfig
from nvidia_tao_pytorch.cv.odise.config.utils import override_default_cfg
from nvidia_tao_pytorch.cv.odise.data.datasets.register_custom_dataset import register_all
from nvidia_tao_pytorch.cv.odise.engine.defaults import default_setup, get_dataset_from_loader, get_model_from_module
from nvidia_tao_pytorch.cv.odise.engine.hooks import EvalHook
from nvidia_tao_pytorch.cv.odise.engine.train_loop import AMPTrainer, SimpleTrainer
from nvidia_tao_pytorch.cv.odise.evaluation import inference_on_dataset
from nvidia_tao_pytorch.cv.odise.utils.events import CommonMetricPrinter, WandbWriter, WriterStack

PathManager.register_handler(S3PathHandler())
spec_root = osp.dirname(osp.dirname(osp.abspath(__file__)))
logger = logging.getLogger(__name__)


def default_writers(cfg):
    """
    Build a list of :class:`EventWriter` to be used.
    It now consists of a :class:`CommonMetricPrinter`,
    :class:`TensorboardXWriter` and :class:`JSONWriter`.

    Args:
        output_dir: directory to store JSON metrics and tensorboard events
        max_iter: the total number of iterations

    Returns:
        list[EventWriter]: a list of :class:`EventWriter` objects.
    """
    if "log_dir" in cfg.train:
        log_dir = cfg.train.log_dir
    else:
        log_dir = cfg.train.output_dir
    PathManager.mkdirs(log_dir)
    ret = [
        # It may not always print what you want to see, since it prints "common" metrics only.
        CommonMetricPrinter(
            cfg.train.max_iter, run_name=osp.join(cfg.train.run_name, cfg.train.run_tag)
        ),
        JSONWriter(osp.join(log_dir, "metrics.json")),
    ]
    if cfg.train.wandb.enable_writer:
        ret.append(
            WandbWriter(
                max_iter=cfg.train.max_iter,
                run_name=osp.join(cfg.train.run_name, cfg.train.run_tag),
                output_dir=log_dir,
                project=cfg.train.wandb.project,
                config=OmegaConf.to_container(cfg, resolve=False),
                resume=cfg.train.wandb.resume,
            )
        )

    return ret


class InferenceRunner:
    def __init__(self, cfg, model):
        self.cfg = cfg
        self.model = model

    def __call__(self, final_iter=False, next_iter=0):
        return evaluate(self.cfg, self.model, final_iter=final_iter, next_iter=next_iter)


def evaluate(cfg, model, final_iter=False, next_iter=0):
    all_ret = dict()
    # make a copy incase we modify it every time calling evaluate
    cfg = OmegaConf.create(cfg)

    # BC for detectron
    if "evaluator" in cfg.dataloader and "test" in cfg.dataloader:
        task_final_iter_only = cfg.dataloader.get("final_iter_only", False)
        task_eval_period = cfg.dataloader.get("eval_period", 1)
        if not final_iter and (task_final_iter_only or next_iter % task_eval_period != 0):
            logger.info(
                f"Skip test set evaluation at iter {next_iter}, "
                f"since task_final_iter_only={task_final_iter_only}, "
                f"next_iter {next_iter} % task_eval_period {task_eval_period}"
                f"={next_iter % task_eval_period} != 0"
            )
        else:
            loader = instantiate(cfg.dataloader.test)

            if "wrapper" in cfg.dataloader:
                wrapper_cfg = cfg.dataloader.wrapper
                # look for the last wrapper
                while "model" in wrapper_cfg:
                    wrapper_cfg = wrapper_cfg.model
                wrapper_cfg.model = get_model_from_module(model)
                # poping _with_dataset_
                if wrapper_cfg.pop("_with_dataset_", False):
                    wrapper_cfg.dataset = get_dataset_from_loader(loader)
                inference_model = create_ddp_model(instantiate(cfg.dataloader.wrapper))
            else:
                inference_model = model

            # poping _with_dataset_
            if isinstance(cfg.dataloader.evaluator, MutableSequence):
                for i in range(len(cfg.dataloader.evaluator)):
                    if cfg.dataloader.evaluator[i].pop("_with_dataset_", False):
                        cfg.dataloader.evaluator[i].dataset = get_dataset_from_loader(loader)
            else:
                if cfg.dataloader.evaluator.pop("_with_dataset_", False):
                    cfg.dataloader.evaluator.dataset = get_dataset_from_loader(loader)

            ret = inference_on_dataset(
                inference_model,
                loader,
                instantiate(cfg.dataloader.evaluator),
                use_amp=cfg.train.amp.enabled,
            )
            print_csv_format(ret)
            all_ret.update(ret)
    if "extra_task" in cfg.dataloader:
        for task in cfg.dataloader.extra_task:
            task_final_iter_only = cfg.dataloader.extra_task[task].get("final_iter_only", False)
            task_eval_period = cfg.dataloader.extra_task[task].get("eval_period", 1)
            if not final_iter and (task_final_iter_only or next_iter % task_eval_period != 0):
                logger.info(
                    f"Skip {task} evaluation at iter {next_iter}, "
                    f"since task_final_iter_only={task_final_iter_only}, "
                    f"next_iter {next_iter} % task_eval_period {task_eval_period}"
                    f"={next_iter % task_eval_period} != 0"
                )
                continue

            logger.info(f"Evaluating extra task: {task}")
            loader = instantiate(cfg.dataloader.extra_task[task].loader)

            # poping _with_dataset_
            if isinstance(cfg.dataloader.extra_task[task].evaluator, MutableSequence):
                for i in range(len(cfg.dataloader.extra_task[task].evaluator)):
                    if cfg.dataloader.extra_task[task].evaluator[i].pop("_with_dataset_", False):
                        cfg.dataloader.extra_task[task].evaluator[
                            i
                        ].dataset = get_dataset_from_loader(loader)
            else:
                if cfg.dataloader.extra_task[task].evaluator.pop("_with_dataset_", False):
                    cfg.dataloader.extra_task[task].evaluator.dataset = get_dataset_from_loader(
                        loader
                    )

            if "wrapper" in cfg.dataloader.extra_task[task]:
                wrapper_cfg = cfg.dataloader.extra_task[task].wrapper
                # look for the last wrapper
                while "model" in wrapper_cfg:
                    wrapper_cfg = wrapper_cfg.model
                wrapper_cfg.model = get_model_from_module(model)
                # poping _with_dataset_
                if wrapper_cfg.pop("_with_dataset_", False):
                    wrapper_cfg.dataset = get_dataset_from_loader(loader)
                inference_model = create_ddp_model(
                    instantiate(cfg.dataloader.extra_task[task].wrapper)
                )
            else:
                inference_model = model

            ret = inference_on_dataset(
                inference_model,
                loader,
                instantiate(cfg.dataloader.extra_task[task].evaluator),
                use_amp=cfg.train.amp.enabled,
            )
            print_csv_format(ret)
            all_ret.update(ret)
    logger.info("Evaluation results for all tasks:")
    print_csv_format(all_ret)
    return all_ret


def run_evaluation(args) -> None:
    """ODISE evaluation."""
    results_dir = args.evaluate.results_dir or osp.join(args.results_dir, 'eval')
    if args.model.type == 'category':
        default_config_file = 'config/common/category_odise.py'
    elif args.model.type == 'caption':
        default_config_file = 'config/common/caption_odise.py'
    else:
        raise NotImplementedError(f"Only `caption` and `category` are supported. Got {args.model.type} instead.")
    cfg = LazyConfig.load(osp.join(spec_root, default_config_file))
    cfg = override_default_cfg(results_dir, cfg, args, comm.get_world_size())
    default_setup(cfg, args)
    logger = setup_logger(results_dir, distributed_rank=comm.get_rank(), name="ODISE")
    logger.info(f"Running with config:\n{LazyConfig.to_py(cfg)}")
    # register custom dataset
    register_all(args)

    model = instantiate_odise(cfg.model)
    model.to(cfg.train.device)
    model = create_ddp_model(model)
    ODISECheckpointer(model, results_dir).resume_or_load(
        args.evaluate.checkpoint, resume=False
    )
    with ExitStack() as stack:
        stack.enter_context(
            WriterStack(
                logger=logger,
                writers=default_writers(cfg) if comm.is_main_process() else None,
            )
        )
        logger.info(evaluate(cfg, model, final_iter=True))
    # Evaluation may take different time among workers.
    # A barrier make them start the next iteration together.
    comm.synchronize()


@hydra_runner(
    config_path=osp.join(spec_root, "experiment_specs"),
    config_name="spec", schema=ExperimentConfig
)
def main(hydra_cfg: ExperimentConfig):
    """Launch training with detectron2."""
    launch(
        run_evaluation,
        hydra_cfg.evaluate.num_gpus,
        num_machines=hydra_cfg.num_machines,
        machine_rank=hydra_cfg.machine_rank,
        dist_url=hydra_cfg.dist_url,
        args=(hydra_cfg,),
    )


if __name__ == "__main__":
    main()
