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

"""Utils for loading/parsing pre-trained models."""
import os
from typing import Any, Mapping, Union

import torch
from safetensors.torch import load_file
from torch.serialization import FileLike

from nvidia_tao_pytorch.core.cookbooks.tlt_pytorch_cookbook import TLTPyTorchCookbook
from nvidia_tao_pytorch.core.utilities import patch_decrypt_checkpoint


class StateDictAdapter:
    """Adapter for handling Pre-Trained Model (PTM) state dictionary key prefixes.

    This class manages model-specific prefixes that need to be stripped from
    state dictionary keys when loading pretrained weights. Different models
    may have different key prefixes in their state dictionaries, and this
    adapter provides a centralized way to handle these variations.

    The adapter maintains a registry of supported models and their corresponding
    prefixes, then can be used to process state dictionaries by removing the
    appropriate prefixes from keys.

    Attributes:
        supported (dict): Dictionary mapping model names (and optional model
            types) to their prefixes.

    Example:
        >>> adapter = StateDictAdapter()
        >>> adapter.add("mae", "model.encoder.")
        >>> adapter.add("classification", "model.")
        >>> adapter.add("visual_changenet_classify", "model.backbone.radio.", model_type="radio_learnable")
        >>>
        >>> # Process a state dict
        >>> state_dict = {"model.encoder.layer1.weight": tensor, "other.weight": tensor}
        >>> cleaned_dict = adapter("mae", state_dict)
        >>> # Result: {"layer1.weight": tensor, "other.weight": tensor}
        >>>
        >>> # Process a state dict with the provided model type.
        >>> state_dict = {"model.backbone.radio.radio.model.blocks.0.attn.qkv.weight": tensor, "other.weight": tensor}
        >>> cleaned_dict = adapter("visual_changenet_classify", state_dict, model_type="radio_learnable")
        >>> # Result: {"radio.model.blocks.0.attn.qkv.weight": tensor, "other.weight": tensor}
    """

    def __init__(self):
        """Initialize the PTM adapter with an empty registry."""
        self.supported = {}  # Key: (model_name, model_type).

    def add(self, model_name, prefix, model_type=None):
        """Add a model and its corresponding prefix to the supported registry.

        Args:
            model_name (str): Name of the model to add support for.
            prefix (str): The prefix string that should be stripped from
                state dictionary keys for this model.
            model_type (str): Optional name of the model type to add support
                for. This is useful when providing `model_name` is not
                sufficient to uniquely identify the model.

        Raises:
            ValueError: If the model_name is already registered.
        """
        model_name = str(model_name)
        model_type = str(model_type) if model_type is not None else None
        if (model_name, model_type) not in self.supported:
            self.supported[(model_name, model_type)] = prefix
        else:
            raise ValueError(f"{(model_name, model_type)} is already supported")

    def get(self, model_name, model_type=None):
        """Get the prefix for a specific model.

        Args:
            model_name (str): Name of the model to get the prefix for.
            model_type (str): Optional name of the model type to get the prefix
                for.

        Returns:
            str: The prefix string for the specified model.

        Raises:
            ValueError: If the model_name is not registered.
        """
        if (model_name, model_type) not in self.supported:
            raise ValueError(f"{(model_name, model_type)} is not supported")
        return self.supported[(model_name, model_type)]

    def __call__(self, model_name, state_dict, model_type=None):
        """Process a state dictionary by removing model-specific prefixes.

        This method removes the registered prefix from all keys in the state
        dictionary that start with that prefix. Keys that don't start with
        the prefix are left unchanged.

        Args:
            model_name (str): Name of the model whose prefix should be removed.
            state_dict (dict): The state dictionary to process, typically
                containing model weights and parameters.
            model_type (str): Optional name of the model type.

        Returns:
            dict: A new state dictionary with prefixes removed from applicable keys.
                If no keys were modified, returns the original state_dict.

        Raises:
            ValueError: If the model_name is not registered.
        """
        prefix = self.get(model_name, model_type=model_type)
        keys = list(state_dict.keys())
        for k in keys:
            v = state_dict.pop(k)
            if k.startswith(prefix):
                k = k.replace(prefix, "", 1)
            state_dict[k] = v
        return state_dict


def load_pretrained_weights(path_or_checkpoint: Union[FileLike, Mapping[str, Any]],
                            map_location="cpu",
                            weights_only=False,
                            ptm_adapter=None,
                            parser=None,
                            **kwargs):
    """Load the pretrained weights.

    Args:
        path_or_checkpoint (str or dict): Path to the pretrained weights file or a checkpoint containing the
            weights.
        map_location (str): A function, `torch.device`, string or a dict specifying how to remap storage locations.
            Default: `"cpu"`.
        weights_only (bool): Indicates whether unpickler should be restricted to loading only tensors, primitive
            types, dictionaries and any types added via `torch.serialization.add_safe_globals`. Default: `False`.
        ptm_adapter (StateDictAdapter): instance of StateDictAdapter to adapt the state dict of a TAO model.
        parser (function): function to parse the state dict for a custom/public model.
        kwargs: Additional arguments passed to the `torch.load` function.
    """
    # Get the checkpoint from the path.
    path = None
    if not isinstance(path_or_checkpoint, dict) and (
        isinstance(path_or_checkpoint, (str, os.PathLike)) or hasattr(path_or_checkpoint, "read")
    ):
        path = path_or_checkpoint
        # Support safetensors files.
        if isinstance(path, (str, os.PathLike)) and path.endswith(".safetensors"):
            checkpoint = load_file(path, device=map_location)
        else:
            checkpoint = torch.load(path, map_location=map_location, weights_only=weights_only, **kwargs)
    else:
        checkpoint = path_or_checkpoint

    # Decrypt the checkpoint if needed.
    if "state_dict_encrypted" in checkpoint:
        key = TLTPyTorchCookbook.get_passphrase()
        if key is None:
            raise PermissionError("Cannot access model state dict without the encryption key.")
        checkpoint = patch_decrypt_checkpoint(checkpoint, key)

    tao_model = checkpoint.get("tao_model", None)
    tao_model_type = checkpoint.get("tao_model_type", None)
    if tao_model is not None:  # for TAO models
        state_dict = ptm_adapter(tao_model, checkpoint["state_dict"], model_type=tao_model_type)
    else:  # for public models
        if 'state_dict' in checkpoint:
            state_dict = checkpoint['state_dict']
        elif 'model' in checkpoint:
            state_dict = checkpoint['model']
        else:
            state_dict = checkpoint
        if parser is not None:
            state_dict = parser(state_dict)
    return state_dict


def common_parser(original):
    """Parse public checkpoints."""
    final = {}
    for k, v in original.items():
        if k.startswith('module.'):
            k = k.replace('module.', '')
        final[k] = v
    return final
