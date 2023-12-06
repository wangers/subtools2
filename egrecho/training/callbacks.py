import lightning.pytorch as L


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
