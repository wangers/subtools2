# -*- coding:utf-8 -*-
# Copyright xmuspeech (Author: Leo 2024-08)
"""Train e2tts.
"""

from typing import cast

import lightning.pytorch as L

from egrecho.core.teacher import (
    LR_SCHEDULER_TYPE,
    LRSCHEDULERS,
    OPTIMIZER_TYPE,
    Teacher,
)
from egrecho.models.e2_tts.model import E2TTSInferOutput, E2TTSTrainOutput
from egrecho.training.lr_schedulers.eden import Eden

LRSCHEDULERS.register(Eden, name="eden", interval="step")


class EdenEpochCallback(L.Callback):
    """This callbck is used to call eden scheduler's `set_epoch` on epoch ends."""

    def on_train_epoch_end(
        self, trainer: L.Trainer, pl_module: L.LightningModule
    ) -> None:
        for config in trainer.lr_scheduler_configs:
            sch = config.scheduler
            if isinstance(sch, Eden):
                sch.step_epoch()


class E2TTSTeacher(Teacher):
    r"""A teacher to train valle."""

    def __init__(
        self,
        optimizer: OPTIMIZER_TYPE = "",
        lr_scheduler: LR_SCHEDULER_TYPE = "",
        lr: float = 1e-4,
        val_mask_frac: float = 0.7,
        **kwargs,
    ) -> None:
        super().__init__()
        self.optimizer = optimizer or (
            "adamw",
            {
                "weight_decay": 0.1,
                "betas": (0.8, 0.9),
            },
        )
        self.lr_scheduler = lr_scheduler or (
            "eden_s",
            {
                "lr_batches": 25_000,
                "warmup_start": 0.01,
                "warmup_steps": 20_000,
            },
        )
        self.lr = lr

        self.val_mask_frac = min(max(val_mask_frac, 1e-1), 1.0)
        self.val_gen_kwargs = kwargs.pop(
            "val_gen_kwargs",
            {
                "steps": 16,
            },
        )

    def configure_optimizers(self):
        return self.configure_single_optimizer(
            self.optimizer, self.lr_scheduler, self.lr
        )

    def training_step(self, batch, batch_idx):
        outputs: E2TTSTrainOutput = self.model(**batch)

        curr_lr = self.trainer.optimizers[0].param_groups[0]["lr"]

        self.model.log(
            "lr",
            curr_lr,
            prog_bar=True,
        )

        self.model.log(
            "train/flow_loss",
            outputs.loss,
            prog_bar=True,
            on_epoch=True,
        )

        return outputs.loss

    def validation_step(self, batch, batch_idx):
        mel_loss, melouts, ref_mels = self.model.generate_loss_step(
            **batch, **self.val_gen_kwargs
        )
        melouts = cast(E2TTSInferOutput, melouts)
        self.model.log(
            "val_loss",
            mel_loss,
            prog_bar=True,
        )
        logs = {
            "mel_loss": mel_loss,
            "melouts": melouts,
            "ref_mels": ref_mels,
        }
        return logs
