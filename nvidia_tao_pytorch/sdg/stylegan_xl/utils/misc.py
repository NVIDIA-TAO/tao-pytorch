# SPDX-FileCopyrightText: Copyright (c) 2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Utilities for generate images."""

import contextlib
import torch
import warnings

from nvidia_tao_pytorch.sdg.stylegan_xl.utils import dnnlib


# Replace NaN/Inf with specified numerical values.
try:
    nan_to_num = torch.nan_to_num  # 1.8.0a0
except AttributeError:
    def nan_to_num(input, nan=0.0, posinf=None, neginf=None, *, out=None):  # pylint: disable=redefined-builtin
        """Replace NaN values in a tensor with a specified number and clamp values.


        Args:
            input (torch.Tensor): The input tensor to process.
            nan (float, optional): The value to replace NaN values with. Default is 0.0.
            posinf (float, optional): The value to clamp positive infinity. Default is None, which uses the maximum
                representable value of the input tensor's dtype.
            neginf (float, optional): The value to clamp negative infinity. Default is None, which uses the minimum
                representable value of the input tensor's dtype.
            out (torch.Tensor, optional): A tensor to store the output. If specified, must have the same shape
                as the output.

        Raises:
            AssertionError: If the input is not a tensor or if the nan value is not 0.

        Returns:
            torch.Tensor: A tensor with NaN values replaced and clamped to specified limits.
        """
        assert isinstance(input, torch.Tensor)
        if posinf is None:
            posinf = torch.finfo(input.dtype).max
        if neginf is None:
            neginf = torch.finfo(input.dtype).min
        assert nan == 0
        return torch.clamp(input.unsqueeze(0).nansum(0), min=neginf, max=posinf, out=out)


# Symbolic assert.
try:
    symbolic_assert = torch._assert  # 1.8.0a0 # pylint: disable=protected-access
except AttributeError:
    symbolic_assert = torch.Assert  # 1.7.0


# Context manager to temporarily suppress known warnings in torch.jit.trace().
# Note: Cannot use catch_warnings because of https://bugs.python.org/issue29672
@contextlib.contextmanager
def suppress_tracer_warnings():
    """Temporarily suppresses known warnings in torch.jit.trace() by modifying the warnings filter.

    Yields:
        None
    """
    flt = ('ignore', None, torch.jit.TracerWarning, None, 0)
    warnings.filters.insert(0, flt)
    yield
    warnings.filters.remove(flt)


# Assert that the shape of a tensor matches the given list of integers.
# None indicates that the size of a dimension is allowed to vary.
# Performs symbolic assertion when used in torch.jit.trace().
def assert_shape(tensor, ref_shape):
    """Assert that the shape of a tensor matches the reference shape.

    Args:
        tensor (torch.Tensor): The tensor to check.
        ref_shape (tuple): The reference shape to compare against.

    Raises:
        AssertionError: If the number of dimensions or the size of any dimension does not match the reference shape.
    """
    if tensor.ndim != len(ref_shape):
        raise AssertionError(f'Wrong number of dimensions: got {tensor.ndim}, expected {len(ref_shape)}')
    for idx, (size, ref_size) in enumerate(zip(tensor.shape, ref_shape)):
        if ref_size is None:
            pass
        elif isinstance(ref_size, torch.Tensor):
            with suppress_tracer_warnings():  # as_tensor results are registered as constants
                symbolic_assert(torch.equal(torch.as_tensor(size), ref_size), f'Wrong size for dimension {idx}')
        elif isinstance(size, torch.Tensor):
            with suppress_tracer_warnings():  # as_tensor results are registered as constants
                symbolic_assert(torch.equal(size, torch.as_tensor(ref_size)), f'Wrong size for dimension {idx}: expected {ref_size}')
        elif size != ref_size:
            raise AssertionError(f'Wrong size for dimension {idx}: got {size}, expected {ref_size}')


def named_params_and_buffers(module):
    """Get a list of named parameters and buffers from a module.

    Args:
        module (torch.nn.Module): The module to get parameters and buffers from.

    Returns:
        list: A list of named parameters and buffers.
    """
    assert isinstance(module, torch.nn.Module)
    return list(module.named_parameters()) + list(module.named_buffers())


def copy_params_and_buffers(src_module, dst_module, require_all=False):
    """Copy parameters and buffers from a source module to a destination module.

    Args:
        src_module (torch.nn.Module): The source module.
        dst_module (torch.nn.Module): The destination module.
        require_all (bool, optional): Whether to require all parameters and buffers to be present in the source module. Default is False.

    Raises:
        AssertionError: If the source or destination module is not an instance of torch.nn.Module.
    """
    assert isinstance(src_module, torch.nn.Module)
    assert isinstance(dst_module, torch.nn.Module)
    src_tensors = dict(named_params_and_buffers(src_module))
    for name, tensor in named_params_and_buffers(dst_module):
        assert (name in src_tensors) or (not require_all)
        if name in src_tensors:
            # tensor.copy_(src_tensors[name].detach()).requires_grad_(tensor.requires_grad) # RuntimeError: a leaf Variable that requires grad is being used in an in-place operation.
            # Save the original requires_grad state
            requires_grad_state = tensor.requires_grad
            # Temporarily set requires_grad to False to allow in-place operations
            tensor.requires_grad_(False)
            # Perform the in-place operation
            tensor.copy_(src_tensors[name].detach())
            # Restore the original requires_grad state
            tensor.requires_grad_(requires_grad_state)


# Print summary table of module hierarchy.
def print_module_summary(module, inputs, max_nesting=3, skip_redundant=True):
    """Print a summary of a PyTorch module, including its parameters, buffers, and output shapes.

    Args:
        module (torch.nn.Module): The module to summarize.
        inputs (tuple or list): The inputs to the module.
        max_nesting (int, optional): The maximum nesting level for submodules. Default is 3.
        skip_redundant (bool, optional): Whether to skip redundant entries. Default is True.

    Returns:
        Any: The outputs of the module.
    """
    assert isinstance(module, torch.nn.Module)
    assert not isinstance(module, torch.jit.ScriptModule)
    assert isinstance(inputs, (tuple, list))

    # Register hooks.
    entries = []
    nesting = [0]

    def pre_hook(_mod, _inputs):
        """Pre-hook to increment nesting level.

        Args:
            _mod (torch.nn.Module): The module being hooked.
            _inputs (tuple): The inputs to the module.
        """
        nesting[0] += 1

    def post_hook(mod, _inputs, outputs):
        """Post-hook to decrement nesting level and record outputs.

        Args:
            mod (torch.nn.Module): The module being hooked.
            _inputs (tuple): The inputs to the module.
            outputs (tuple): The outputs from the module.
        """
        nesting[0] -= 1
        if nesting[0] <= max_nesting:
            outputs = list(outputs) if isinstance(outputs, (tuple, list)) else [outputs]
            outputs = [t for t in outputs if isinstance(t, torch.Tensor)]
            entries.append(dnnlib.EasyDict(mod=mod, outputs=outputs))
            # if (dnnlib.EasyDict(mod=mod, outputs=outputs).mod is module):
            #     print("dnnlib.EasyDict(mod=mod, outputs=outputs): ", dnnlib.EasyDict(mod=mod, outputs=outputs))
    hooks = [mod.register_forward_pre_hook(pre_hook) for mod in module.modules()]
    hooks += [mod.register_forward_hook(post_hook) for mod in module.modules()]

    # Run module.
    outputs = module(*inputs)
    for hook in hooks:
        hook.remove()

    # Identify unique outputs, parameters, and buffers.
    tensors_seen = set()
    for e in entries:
        e.unique_params = [t for t in e.mod.parameters() if id(t) not in tensors_seen]
        e.unique_buffers = [t for t in e.mod.buffers() if id(t) not in tensors_seen]
        e.unique_outputs = [t for t in e.outputs if id(t) not in tensors_seen]
        tensors_seen |= {id(t) for t in e.unique_params + e.unique_buffers + e.unique_outputs}

    # Filter out redundant entries.
    if skip_redundant:
        entries = [e for e in entries if len(e.unique_params) or len(e.unique_buffers) or len(e.unique_outputs)]

    # Construct table.
    rows = [[type(module).__name__, 'Parameters', 'Buffers', 'Output shape', 'Datatype']]
    rows += [['---'] * len(rows[0])]
    param_total = 0
    buffer_total = 0
    submodule_names = {mod: name for name, mod in module.named_modules()}
    for e in entries:
        name = '<top-level>' if e.mod is module else submodule_names[e.mod]
        param_size = sum(t.numel() for t in e.unique_params)
        buffer_size = sum(t.numel() for t in e.unique_buffers)
        output_shapes = [str(list(t.shape)) for t in e.outputs]
        output_dtypes = [str(t.dtype).rsplit('.', maxsplit=1)[-1] for t in e.outputs]
        rows += [[
            name + (':0' if len(e.outputs) >= 2 else ''),
            str(param_size) if param_size else '-',
            str(buffer_size) if buffer_size else '-',
            (output_shapes + ['-'])[0],
            (output_dtypes + ['-'])[0],
        ]]
        for idx in range(1, len(e.outputs)):
            rows += [[name + f':{idx}', '-', '-', output_shapes[idx], output_dtypes[idx]]]
        param_total += param_size
        buffer_total += buffer_size
    rows += [['---'] * len(rows[0])]
    rows += [['Total', str(param_total), str(buffer_total), '-', '-']]

    # Print table.
    widths = [max(len(cell) for cell in column) for column in zip(*rows)]
    print()
    for row in rows:
        print('  '.join(cell + ' ' * (width - len(cell)) for cell, width in zip(row, widths)))
    print()
    return outputs
