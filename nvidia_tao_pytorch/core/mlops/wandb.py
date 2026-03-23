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

"""Module containing integrations for W&B."""

from datetime import datetime
import logging

from omegaconf import DictConfig, OmegaConf

from pytorch_lightning.loggers import WandbLogger
import wandb
from wandb import AlertLevel
import os

DEFAULT_WANDB_CONFIG = "~/.netrc"
logger = logging.getLogger(__name__)

_WANDB_INITIALIZED = False


def alert(title, text, duration=300, level=0, is_rank_zero=True):
    """Send alert."""
    alert_levels = {
        0: AlertLevel.INFO,
        1: AlertLevel.WARN,
        2: AlertLevel.ERROR
    }
    if is_wandb_initialized() and is_rank_zero:
        wandb.alert(
            title=title,
            text=text,
            level=alert_levels[level],
            wait_duration=duration
        )


def is_wandb_initialized():
    """Check if wandb has been initialized."""
    global _WANDB_INITIALIZED  # pylint: disable=W0602,W0603
    return _WANDB_INITIALIZED


def check_wandb_logged_in():
    """Check if weights and biases have been logged in."""
    wandb_logged_in = False
    try:
        wandb_api_key = os.getenv("WANDB_API_KEY", None)
        if wandb_api_key is not None or os.path.exists(os.path.expanduser(DEFAULT_WANDB_CONFIG)):
            wandb_logged_in = wandb.login(key=wandb_api_key)
            return wandb_logged_in
    except wandb.errors.UsageError:
        logger.warning("WandB wasn't logged in.")
    return False


def initialize_wandb(project: str = "TAO Toolkit",
                     entity: str = None,
                     sync_tensorboard: bool = True,
                     save_code: bool = False,
                     tags: list = None,
                     run_id: str = None,
                     name: str = "train",
                     config=None,
                     wandb_logged_in: bool = False,
                     results_dir: str = os.getcwd(),
                     group: str = None):
    """Function to initialize wandb client with the weights and biases server.

    If wandb initialization fails, then the function just catches the exception
    and prints an error log with the reason as to why wandb.init() failed.

    Args:
        project (str): Name of the project to sync data with.
        entity (str): Name of the wandb entity.
        sync_tensorboard (bool): Boolean flag to synchronize
            tensorboard and wandb visualizations.
        save_code (bool): Choose to save the entypoint script or not.
        tags (list): List of string tags for the experiment.
        name (str): Name of the task running.
        config (OmegaConf.DictConf): Configuration element of the task that's being.
            Typically, this is the yaml container generated from the `experiment_spec`
            file used to run the job.
        wandb_logged_in (bool): Boolean flag to check if wandb was logged in.
        results_dir (str): Output directory of the experiment.

    Returns:
        No explicit returns.
    """
    logger.info("Initializing wandb.")
    try:
        assert wandb_logged_in, (
            "WandB client wasn't logged in. Please make sure to set "
            "the WANDB_API_KEY env variable or run `wandb login` in "
            "over the CLI and copy the ~/.netrc file to the container."
        )
        start_time = datetime.now()
        time_string = start_time.strftime("%m:%d:%y_%H:%M:%S")
        name = f"{name}_{time_string}"
        # Format config for wandb
        cfg_dict = (
            OmegaConf.to_container(config, resolve=True)
            if isinstance(config, DictConfig)
            else dict(config) if config is not None else {}
        )
        init_kwargs = {
            "name": name,
            "project": project,
            "entity": entity,
            "log_model": False,
            "save_dir": results_dir,
            "sync_tensorboard": sync_tensorboard,
            "save_code": save_code,
            "config": cfg_dict,
            "tags": tags
        }
        if group:
            init_kwargs["group"] = group
        if run_id:
            init_kwargs["id"] = run_id

        # Initialize and setup wandb logger. WandB logger **kwargs in the
        # class definition takes kwargs that internally gets routed to wandb.init().
        # So you can add kwargs from wandb.init() as part of this constructor along
        # with it's own args, kwargs.
        wandb_logger = WandbLogger(**init_kwargs)

        global _WANDB_INITIALIZED  # pylint: disable=W0602,W0603
        _WANDB_INITIALIZED = True
        return wandb_logger
    except Exception as e:
        logger.warning("Wandb logging failed with error %s", e)
        return None
