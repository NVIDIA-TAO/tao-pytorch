# SPDX-FileCopyrightText: Copyright (c) 2023 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Evaluate a trained pose classification model."""
import csv
import logging
import os
from pytorch_lightning import Trainer

from nvidia_tao_pytorch.core.decorators.workflow import monitor_status
from nvidia_tao_pytorch.core.initialize_experiments import initialize_evaluation_experiment
from nvidia_tao_pytorch.core.hydra.hydra_runner import hydra_runner
from nvidia_tao_pytorch.config.pose_classification.default_config import ExperimentConfig
from nvidia_tao_pytorch.cv.pose_classification.dataloader.pl_pc_data_module import PCDataModule
from nvidia_tao_pytorch.cv.pose_classification.model.pl_pc_model import PoseClassificationModel

logger = logging.getLogger(__name__)


# TODO @seanf: cc says this isn't used
def dump_cm(csv_path, cm, id2name):
    """
    Dump the confusion matrix to a CSV file.

    This function saves the confusion matrix to a CSV file, where each row and column represent a class,
    and the cell values represent the counts.

    Args:
        csv_path (str): The path to the output CSV file.
        cm (np.ndarray): The confusion matrix of shape (num_classes, num_classes).
        id2name (dict): A dictionary mapping class IDs to class names.
    """
    n_class = len(id2name.keys())
    with open(csv_path, "w") as f:
        writer = csv.writer(f)
        label_list = ["class"]
        for idx in range(n_class):
            label_list.append(id2name[idx])
        writer.writerow(label_list)
        for row_id in range(n_class):
            row = [id2name[row_id]]
            for col_id in range(n_class):
                row.append(cm[row_id][col_id])
            writer.writerow(row)


def run_experiment(experiment_config, key):
    """
    Run the evaluation process.

    This function initializes the necessary components for evaluation, including the model, data loader,
    and inferencer. It performs evaluation on the test dataset and computes evaluation metrics and the confusion matrix.

    Args:
        experiment_config (dict): The experiment configuration containing the model and evaluation parameters.
        key (str): The encryption key for intermediate checkpoints.

    Raises:
        Exception: If any error occurs during the evaluation process.
    """
    model_path, trainer_kwargs = initialize_evaluation_experiment(experiment_config, key)
    if len(trainer_kwargs['devices']) > 1:
        trainer_kwargs['devices'] = [trainer_kwargs['devices'][0]]
        logger.info(f"Pose Classification does not support multi-GPU evaluation at this time. Using only GPU {trainer_kwargs['devices']}")

    dm = PCDataModule(experiment_config)
    model = PoseClassificationModel.load_from_checkpoint(model_path,
                                                         map_location="cpu",
                                                         experiment_spec=experiment_config)

    trainer = Trainer(**trainer_kwargs)

    trainer.test(model, datamodule=dm)


spec_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


# Load experiment specification, additially using schema for validation/retrieving the default values.
# --config_path and --config_name will be provided by the entrypoint script.
@hydra_runner(
    config_path=os.path.join(spec_root, "experiment_specs"), config_name="experiment", schema=ExperimentConfig
)
@monitor_status(name="Pose Classification", mode="evaluate")
def main(cfg: ExperimentConfig) -> None:
    """
    Run the evaluation process.

    This function serves as the entry point for the evaluation script.
    It loads the experiment specification, updates the results directory, and calls the 'run_experiment' function.

    Args:
        cfg (ExperimentConfig): The experiment configuration retrieved from the Hydra configuration files.
    """
    run_experiment(experiment_config=cfg,
                   key=cfg.encryption_key)


if __name__ == "__main__":
    main()
