# SPDX-FileCopyrightText: Copyright (c) 2023 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import pytest
import torch
from torchvision.models import resnet18
from nvidia_tao_pytorch.cv.ocdnet.lr_schedulers.schedulers import WarmupPolyLR


@pytest.mark.cv_unit
def test_ocdnet_lr_scheduler():
    max_iter = 600 * 63
    model = resnet18()
    op = torch.optim.Adam(model.parameters(), 1e-3)
    sc = WarmupPolyLR(op, max_iters=max_iter, power=0.9, warmup_iters=3*63, warmup_method='constant', epochs=1)
    lr = []
    for i in range(max_iter):
        sc.step()
        #print(i, sc.last_epoch, sc.get_lr()[0])
        lr.append(sc.get_lr()[0])
