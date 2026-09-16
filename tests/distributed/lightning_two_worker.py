# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Standalone real-process Lightning smoke used by the launcher tests."""

import json
import os
from pathlib import Path
import sys

import pytorch_lightning as pl
from pytorch_lightning.plugins.environments import LightningEnvironment
import torch
from torch.utils.data import DataLoader, TensorDataset


class _TinyModel(pl.LightningModule):
    """Minimal trainable model that is safe to run with CPU DDP."""

    def __init__(self):
        super().__init__()
        self.layer = torch.nn.Linear(1, 1)

    def training_step(self, batch, _batch_index):
        features, labels = batch
        return torch.nn.functional.mse_loss(self.layer(features), labels)

    def configure_optimizers(self):
        return torch.optim.SGD(self.parameters(), lr=0.01)


class _RankRecorder(pl.Callback):
    """Publish the rank state seen inside each real Lightning worker."""

    def __init__(self, output: Path):
        self.output = output

    def on_fit_start(self, trainer, _module):
        payload = {
            "pid": os.getpid(),
            "global_rank": trainer.global_rank,
            "local_rank": trainer.local_rank,
            "node_rank": trainer.node_rank,
            "world_size": trainer.world_size,
            "environment_local_rank": int(os.environ["LOCAL_RANK"]),
            "strategy": type(trainer.strategy).__name__,
        }
        destination = self.output / f"rank-{trainer.global_rank}.json"
        destination.write_text(json.dumps(payload), encoding="utf-8")


def main(output: Path, *, num_nodes: int = 1, devices: int = 2) -> None:
    """Run one batch through Lightning's node-level DEFT process topology."""
    dataset = TensorDataset(
        torch.arange(4, dtype=torch.float32).reshape(-1, 1),
        torch.arange(4, dtype=torch.float32).reshape(-1, 1),
    )
    trainer = pl.Trainer(
        accelerator="cpu",
        devices=devices,
        num_nodes=num_nodes,
        strategy="ddp",
        plugins=[LightningEnvironment()] if os.environ.get(
            "TAO_REFINEMENT_LIGHTNING_LAUNCH"
        ) == "1" else None,
        max_epochs=1,
        limit_train_batches=1,
        logger=False,
        enable_checkpointing=False,
        enable_model_summary=False,
        enable_progress_bar=False,
        callbacks=[_RankRecorder(output)],
    )
    trainer.fit(_TinyModel(), train_dataloaders=DataLoader(dataset, batch_size=2))


if __name__ == "__main__":
    main(
        Path(sys.argv[1]),
        num_nodes=int(sys.argv[2]) if len(sys.argv) > 2 else 1,
        devices=int(sys.argv[3]) if len(sys.argv) > 3 else 2,
    )
