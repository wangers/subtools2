# -*- coding:utf-8 -*-
# Copyright xmuspeech (Author: Leo 2024-04)
"""Train valle.
"""

from copy import deepcopy
from typing import Dict

import lightning.pytorch as L
from torch.nn.modules import ModuleDict
from torchmetrics.classification import MulticlassAccuracy

from egrecho.core.teacher import (
    LR_SCHEDULER_TYPE,
    LRSCHEDULERS,
    OPTIMIZER_TYPE,
    Teacher,
)
from egrecho.models.valle.model import ValleOutput
from egrecho.models.valle.valle_config import ValleModelConfig
from egrecho.training.lr_schedulers.eden import Eden
from egrecho.utils.logging import get_logger
from egrecho.utils.torch_utils import tensor_has_nan

logger = get_logger()

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


class ValleTeacher(Teacher):
    r"""A teacher to train valle."""

    def __init__(
        self,
        optimizer: OPTIMIZER_TYPE = "",
        lr_scheduler: LR_SCHEDULER_TYPE = "",
        lr: float = 0.05,
    ) -> None:
        super().__init__()
        self.optimizer = optimizer or (
            "scaled_adam",
            {
                "clipping_scale": 2.0,
                "clipping_update_period": 1000,
                "betas": (0.9, 0.95),
            },
        )
        self.lr_scheduler = lr_scheduler or (
            "eden",
            {
                "lr_batches": 5000,
                "lr_epochs": 4,
                "warmup_steps": 500,
            },
        )
        self.lr = lr

    def setup_model(self):
        config: ValleModelConfig = self.model.config
        has_ar, has_nar = config.has_ar, config.has_nar
        names = {"ar_acc_top10": has_ar, "nar_acc_top10": has_nar}
        name_keys = [k for k in names if names[k]]

        acc_metrics = {
            name_key: MulticlassAccuracy(
                config.codebook_size + 1,
                top_k=10,
                average="micro",
                multidim_average="global",
                ignore_index=-100
                if name_key == "ar_acc_top10"
                else config.codebook_size,
            )
            for name_key in name_keys
        }
        self.setup_train_metrics(acc_metrics)
        self.setup_val_metrics(deepcopy(acc_metrics))

    def configure_optimizers(self):
        return self.configure_single_optimizer(
            self.optimizer, self.lr_scheduler, self.lr
        )

    def common_step(self, batch: dict, batch_idx: int, metrics: ModuleDict) -> Dict:

        outs: ValleOutput = self.model(**batch)
        loss = None
        if (ar_logits := outs.ar_logits) is not None:
            loss = outs.ar_loss

            metrics["ar_acc_top10"](ar_logits.detach().permute(0, 2, 1), outs.ar_labels)
        if (nar_logits := outs.nar_logits) is not None:
            if loss is not None:
                loss += outs.nar_loss
                loss /= 2
            else:
                loss = outs.nar_loss

            metrics["nar_acc_top10"](
                nar_logits.detach().permute(0, 2, 1), outs.nar_labels
            )

        outputs = {"loss": loss}
        logs = {}
        logs["loss"] = loss
        logs.update(metrics)  # add metrics to log.
        outputs["logs"] = logs

        return outputs

    def training_step(self, batch, batch_idx):

        outputs = self.common_step(batch, batch_idx, self.train_metrics)
        has_none = False
        from egrecho.utils.torch_utils import save_dislike_batch

        if tensor_has_nan(outputs["loss"]):
            # skip batch if something went wrong for some reason
            logger.warning(
                f"Got NaN loss in batch_idx:{batch_idx}, skip it,{outputs},input_ids shape={batch['input_ids'].shape}.",
            )
            # save_dislike_batch(batch, f"{batch_idx}-train-{self.trainer.global_rank}")
            has_none = True
        self.trainer.strategy.barrier()
        if self.trainer.strategy.reduce_boolean_decision(has_none, all=False):
            return None
        curr_lr = self.trainer.optimizers[0].param_groups[0]["lr"]

        self.model.log(
            "lr",
            curr_lr,
            prog_bar=True,
        )
        self.model.log_dict(
            {f"train_{k}": v for k, v in outputs["logs"].items()},
            on_epoch=True,
            prog_bar=True,
        )

        return outputs["loss"]

    def validation_step(self, batch, batch_idx):
        outputs = self.common_step(batch, batch_idx, self.val_metrics)
        has_none = False
        if tensor_has_nan(outputs["loss"]):
            # skip batch if something went wrong for some reason
            logger.warning(
                f"Got NaN loss in batch_idx:{batch_idx}, {outputs},input_ids shape={batch['input_ids'].shape}.",
            )
            has_none = True
        self.trainer.strategy.barrier()
        if self.trainer.strategy.reduce_boolean_decision(has_none, all=False):
            return None
        self.model.log_dict(
            {f"val_{k}": v for k, v in outputs["logs"].items()},
            prog_bar=True,
        )
        return outputs["loss"]
