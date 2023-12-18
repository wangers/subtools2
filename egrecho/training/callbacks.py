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
