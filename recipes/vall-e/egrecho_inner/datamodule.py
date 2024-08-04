# -*- coding:utf-8 -*-
# (Author: Leo 202406)


import lightning.pytorch as pl
from data_builder import LhotseBuilder
from torch.utils.data.dataloader import DataLoader


class DataModule(pl.LightningDataModule):
    """
    A simple lightning datamoudle wrapper for dataloader.

    Args:
        builder (LhotseBuilder):
            The inner data builder instance is responsible for creating the dataloaders.
    """

    def __init__(
        self,
        builder: LhotseBuilder,
    ) -> None:
        self.data_builder = builder

        self._train_cuts = None
        self._val_cuts = None
        self._test_cuts = None
        super().__init__()

    def setup(self, stage: str) -> None:
        self.setup_data()

    def setup_data(self) -> None:
        """
        Builds datasets and assigns dataloader func to lightning datamodule.
        """
        self.__build_dataset()
        if self._train_cuts:
            self.train_dataloader = self._train_dataloader
        if self._val_cuts:
            self.val_dataloader = self._val_dataloader
        if self._test_cuts:
            self.test_dataloader = self._test_dataloader

    def __build_dataset(self):
        self._train_cuts = self.data_builder.train_cuts()
        self._val_cuts = self.data_builder.dev_cuts()
        self._test_cuts = self.data_builder.test_cuts()

    def _train_dataloader(self) -> DataLoader:
        return self.data_builder.train_dataloaders(self._train_cuts)

    def _val_dataloader(self) -> DataLoader:
        return self.data_builder.valid_dataloaders(self._val_cuts)

    def _test_dataloader(self) -> DataLoader:
        return self.data_builder.test_dataloaders(self._test_cuts)

    # def state_dict(self):
    #     """Called when saving a checkpoint, implement to generate and save datamodule state.

    #     Returns:
    #         A dictionary containing datamodule state.

    #     """
    #     if self.trainer
    #     return {}
