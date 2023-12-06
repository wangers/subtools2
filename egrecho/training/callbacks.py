import lightning.pytorch as L
from lightning.pytorch.utilities import rank_zero_only
from egrecho.utils.logging import get_logger

logger = get_logger()


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


class LearningRate(L.callbacks.LearningRateMonitor):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.last_values = None

    def on_train_start(self, trainer, *args, **kwargs):
        if not trainer.lr_schedulers:
            L.utilities.rank_zero_warn(
                "You are using LearningRateMonitor callback with models "
                "that have no learning rate schedulers",
                RuntimeWarning,
            )
        names = self._find_names(trainer.lr_schedulers)
        self.lrs = {name: [] for name in names}
        self.last_values = {}

    @rank_zero_only
    def on_epoch_end(self, trainer, *args, **kwargs):
        super().on_epoch_end(trainer, *args, **kwargs)
        for k, v in self.lrs.items():
            prev_value = self.last_values.get(k, None)
            new_value = v[-1]
            if prev_value is not None and prev_value != new_value:
                L.utilities.rank_zero_info(
                    "E{}: {} {:.3e} ⟶ {:.3e}".format(
                        trainer.current_epoch,
                        k,
                        prev_value,
                        new_value,
                    ),
                )
            self.last_values[k] = new_value
