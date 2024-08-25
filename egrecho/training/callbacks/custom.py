# -*- coding:utf-8 -*-
# Copyright xmuspeech (Author: Leo 2023-11)

from typing import Any

import lightning.pytorch as L
import torch

from egrecho.utils.logging import get_logger

logger = get_logger(__name__)


class DataSetEpochCallback(L.Callback):
    """This callbck is used to call dataset's `set_epoch` before every epoch.

    Some case (e.g., IterableDataset) has infinite sampler, we can derirectly call
    the set_epoch if it has that method.
    """

    def on_train_epoch_start(
        self, trainer: L.Trainer, pl_module: L.LightningModule
    ) -> None:
        if (combined_loader := trainer.fit_loop._combined_loader) is not None:
            dls = combined_loader.flattened

        for _, dl in enumerate(dls):
            set_epoch = getattr(dl.dataset, "set_epoch", None)
            if callable(set_epoch):
                set_epoch(trainer.current_epoch)


class LastBatchPatchCallback(L.Callback):
    """Patch to fit loop, cause now lightning can't correctly get last batch info when
    the epoch loop got a StopIteration from data fetcher. Temporary use for Iterable dataset
    and apply ReduceP learning scheduler. Unfortunately, this can't infulence the inner epoch loop
    , means that in the Iterable ds case, epoch level learning scheduler in liggtning's auto
    optimization may invalid now.
    """

    def on_train_epoch_end(
        self, trainer: L.Trainer, pl_module: L.LightningModule
    ) -> None:
        fit_loop = trainer.fit_loop
        if not fit_loop.epoch_loop.batch_progress.is_last_batch and (
            fit_loop._data_fetcher and fit_loop._data_fetcher.done
        ):
            fit_loop.epoch_loop.batch_progress.is_last_batch = True


class ScalarBatchCallback(L.Callback):
    def on_train_batch_end(
        self,
        trainer: L.Trainer,
        pl_module: L.LightningModule,
        outputs,
        batch: Any,
        batch_idx: int,
    ) -> None:

        self.scaler_monitor(trainer, batch_idx)

    def scaler_monitor(self, trainer, batch_idx):
        if (
            (scaler := trainer.scaler) is not None
            and batch_idx
            and batch_idx % 100 == 0
        ):
            cur_grad_scale = scaler._scale.item()
            if cur_grad_scale < 1.0 or (cur_grad_scale < 8.0 and batch_idx % 400 == 0):
                scaler.update(cur_grad_scale * 2.0)
            if cur_grad_scale < 0.01:
                logger.warning(f"Grad scale is small: {cur_grad_scale}")
            if cur_grad_scale < 1.0e-05:
                logger.warning(f"Grad scale is too small: {cur_grad_scale}")
                # raise RuntimeError(f"grad_scale is too small, exiting: {cur_grad_scale}")
            for logger in trainer.loggers:
                logger.log_metrics(
                    {"grad_scale": cur_grad_scale},
                    step=trainer.fit_loop.epoch_loop._batches_that_stepped,
                )
