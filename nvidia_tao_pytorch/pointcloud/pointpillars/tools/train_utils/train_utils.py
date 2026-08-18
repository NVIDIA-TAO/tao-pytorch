# SPDX-FileCopyrightText: Copyright (c) 2023 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Training utilities for PointPillars."""
import glob
import os
import shutil
import struct
import tempfile
import time
from datetime import timedelta

import torch
from torch.nn.utils import clip_grad_norm_
import tqdm
from eff.core.codec import encrypt_stream


def train_one_epoch(model, optimizer, train_loader, model_func, lr_scheduler, accumulated_iter, optim_cfg,
                    rank, tbar, total_it_each_epoch, total_epochs, current_epoch, dataloader_iter, tb_log=None,
                    status_logging=None, leave_pbar=False):
    """Train for one epoch."""
    if total_it_each_epoch == len(train_loader):
        dataloader_iter = iter(train_loader)
    tic = toc = 0
    if rank == 0:
        pbar = tqdm.tqdm(total=total_it_each_epoch, leave=leave_pbar, desc='train', dynamic_ncols=True)
    tic = time.perf_counter()
    for _ in range(total_it_each_epoch):
        try:
            batch = next(dataloader_iter)
        except StopIteration:
            dataloader_iter = iter(train_loader)
            batch = next(dataloader_iter)
            print('new iters')

        lr_scheduler.step(accumulated_iter)

        try:
            cur_lr = float(optimizer.lr)
        except:  # noqa: E722
            cur_lr = optimizer.param_groups[0]['lr']

        if tb_log is not None:
            tb_log.add_scalar('meta_data/learning_rate', cur_lr, accumulated_iter)

        model.train()
        optimizer.zero_grad()

        loss, tb_dict, disp_dict = model_func(model, batch)

        loss.backward()
        clip_grad_norm_(model.parameters(), optim_cfg.grad_norm_clip)
        optimizer.step()

        accumulated_iter += 1
        disp_dict.update({'loss': loss.item(), 'lr': cur_lr})
        # log to console and tensorboard
        if rank == 0:
            pbar.update()
            pbar.set_postfix(dict(total_it=accumulated_iter))
            tbar.set_postfix(disp_dict)
            tbar.refresh()

            if tb_log is not None:
                tb_log.add_scalar('train/loss', loss, accumulated_iter)
                tb_log.add_scalar('meta_data/learning_rate', cur_lr, accumulated_iter)
                for key, val in tb_dict.items():
                    tb_log.add_scalar('train/' + key, val, accumulated_iter)
    if rank == 0:
        pbar.close()
    toc = time.perf_counter()
    if status_logging is not None:
        status_logging.get_status_logger().kpi = {
            "learning_rate": cur_lr,
            "loss": loss.item()
        }
        data = {
            "epoch": current_epoch,
            "time_per_epoch": str(timedelta(seconds=toc - tic)),
            "max_epoch": total_epochs,
            "eta": str(timedelta(seconds=(total_epochs - current_epoch) * (toc - tic)))
        }
        status_logging.get_status_logger().write(
            data=data,
            message="Train metrics generated.",
            status_level=status_logging.Status.RUNNING
        )
    return accumulated_iter


def train_model(model, optimizer, train_loader, model_func, lr_scheduler, optim_cfg,
                start_epoch, total_epochs, start_iter, rank, tb_log, status_logging,
                ckpt_save_dir, key,
                train_sampler=None, lr_warmup_scheduler=None, ckpt_save_interval=1,
                max_ckpt_save_num=50, merge_all_iters_to_one_epoch=False):
    """Train model."""
    accumulated_iter = start_iter
    with tqdm.trange(start_epoch, total_epochs, desc='epochs', dynamic_ncols=True, leave=(rank == 0)) as tbar:
        total_it_each_epoch = len(train_loader)
        if merge_all_iters_to_one_epoch:
            assert hasattr(train_loader.dataset, 'merge_all_iters_to_one_epoch')
            train_loader.dataset.merge_all_iters_to_one_epoch(merge=True, epochs=total_epochs)
            total_it_each_epoch = len(train_loader) // max(total_epochs, 1)

        dataloader_iter = iter(train_loader)
        for cur_epoch in tbar:
            if train_sampler is not None:
                train_sampler.set_epoch(cur_epoch)

            # train one epoch
            if lr_warmup_scheduler is not None and cur_epoch < optim_cfg.warmup_epoch:
                cur_scheduler = lr_warmup_scheduler
            else:
                cur_scheduler = lr_scheduler
            accumulated_iter = train_one_epoch(
                model, optimizer, train_loader, model_func,
                lr_scheduler=cur_scheduler,
                accumulated_iter=accumulated_iter, optim_cfg=optim_cfg,
                rank=rank, tbar=tbar, tb_log=tb_log,
                status_logging=status_logging,
                leave_pbar=(cur_epoch + 1 == total_epochs),
                total_it_each_epoch=total_it_each_epoch,
                total_epochs=total_epochs,
                current_epoch=cur_epoch,
                dataloader_iter=dataloader_iter
            )

            # save trained model
            suffix = 'pth'
            if key:
                suffix = 'tlt'
            trained_epoch = cur_epoch + 1
            if trained_epoch % ckpt_save_interval == 0 and rank == 0:

                ckpt_list = glob.glob(str(ckpt_save_dir / f'checkpoint_epoch_*.{suffix}'))
                ckpt_list.sort(key=os.path.getmtime)

                if ckpt_list.__len__() >= max_ckpt_save_num:
                    for cur_file_idx in range(0, len(ckpt_list) - max_ckpt_save_num + 1):
                        os.remove(ckpt_list[cur_file_idx])

                ckpt_name = ckpt_save_dir / (f'checkpoint_epoch_{trained_epoch}.{suffix}')
                save_checkpoint(
                    checkpoint_model(model, optimizer, trained_epoch, accumulated_iter),
                    ckpt_name,
                    key
                )


def model_state_to_cpu(model_state):
    """Move model states to CPU."""
    model_state_cpu = type(model_state)()  # ordered dict
    for key, val in model_state.items():
        model_state_cpu[key] = val.cpu()
    return model_state_cpu


def checkpoint_state(model=None, optimizer=None, epoch=None, it=None):
    """Get checkpoint states."""
    optim_state = optimizer.state_dict() if optimizer is not None else None
    if model is not None:
        if isinstance(model, torch.nn.parallel.DistributedDataParallel):
            model_state = model_state_to_cpu(model.module.state_dict())
        else:
            model_state = model.state_dict()
    else:
        model_state = None

    try:
        import pointcloud
        version = 'pointcloud+' + pointcloud.__version__
    except:  # noqa: E722
        version = 'none'

    return {'epoch': epoch, 'it': it, 'model_state': model_state, 'optimizer_state': optim_state, 'version': version}


def checkpoint_model(model=None, optimizer=None, epoch=None, it=None):
    """Get checkpoint states from model."""
    optim_state = optimizer.state_dict() if optimizer is not None else None
    try:
        import pointcloud
        version = 'pointcloud+' + pointcloud.__version__
    except:  # noqa: E722
        version = 'none'
    if isinstance(model, torch.nn.parallel.DistributedDataParallel):
        model = model.module
    return {'epoch': epoch, 'it': it, 'model': model, 'optimizer_state': optim_state, 'version': version}


def encrypt_onnx(tmp_file_name, output_file_name, key):
    """Encrypt the onnx model"""
    with open(tmp_file_name, "rb") as open_temp_file, open(output_file_name,
                                                           "wb") as open_encoded_file:
        # set the input name magic number
        open_encoded_file.write(struct.pack("<i", 0))
        encrypt_stream(
            input_stream=open_temp_file, output_stream=open_encoded_file,
            passphrase=key, encryption=True
        )


def encrypt_pytorch(tmp_file_name, output_file_name, key):
    """Encrypt the pytorch model"""
    with open(tmp_file_name, "rb") as open_temp_file, open(output_file_name,
                                                           "wb") as open_encoded_file:
        encrypt_stream(
            input_stream=open_temp_file, output_stream=open_encoded_file,
            passphrase=key, encryption=True
        )


def save_checkpoint(state, filename, key):
    """Save the checkpoint."""
    if False and 'optimizer_state' in state:
        optimizer_state = state['optimizer_state']
        state.pop('optimizer_state', None)
        optimizer_filename = '{}_optim.pth'.format(filename)
        torch.save({'optimizer_state': optimizer_state}, optimizer_filename)
    handle, temp_name = tempfile.mkstemp(".tlt")
    os.close(handle)
    torch.save(state, temp_name)
    if key:
        encrypt_pytorch(temp_name, filename, key)
        os.remove(temp_name)
    else:
        shutil.move(temp_name, filename)
