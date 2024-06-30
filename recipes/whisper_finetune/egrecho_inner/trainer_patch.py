# (Author: Leo 2024-06)
from typing import Optional

from olr_datamodule import OlrAsrDataModule
from torch.utils.data import DataLoader, Dataset
from transformers import Seq2SeqTrainer


class CustomTrainer(Seq2SeqTrainer):
    @property
    def datamodule(self) -> Optional[OlrAsrDataModule]:
        return getattr(self, '_datamodule', None)

    @datamodule.setter
    def datamodule(self, datamodule: OlrAsrDataModule):
        assert isinstance(datamodule, OlrAsrDataModule)
        self._datamodule = datamodule

    def get_train_dataloader(self) -> DataLoader:
        """
        Returns the training [`~torch.utils.data.DataLoader`].
        """

        if datamodule := self.datamodule:
            return datamodule.train_dataloaders()
        return super().get_train_dataloader()

    def get_eval_dataloader(self, eval_dataset: Optional[Dataset] = None) -> DataLoader:
        """
        Returns the eval [`~torch.utils.data.DataLoader`].

        """
        eval_dataset = (
            eval_dataset
            if eval_dataset is not None
            else (self.datamodule or self.eval_dataset)
        )
        if eval_dataset is None:
            raise ValueError("Trainer: evaluation requires an eval_dataset.")
        if isinstance(eval_dataset, OlrAsrDataModule):
            return eval_dataset.valid_dataloaders(eval_dataset.valid_cuts())
        else:
            return super().get_eval_dataloader(eval_dataset)
