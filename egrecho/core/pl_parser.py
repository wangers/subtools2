# -*- coding:utf-8 -*-

# Copyright xmuspeech (Author: Leo 2023-04-11)
# modified from Lightning-AI
# https://github.com/Lightning-AI/lightning/blob/master/src/lightning/pytorch/cli.py

import inspect
import os
from typing import List, Type, Union, cast

from jsonargparse import Namespace
from lightning.fabric.utilities.cloud_io import get_filesystem
from lightning.pytorch import Callback, LightningDataModule, LightningModule, Trainer
from lightning.pytorch.utilities.model_helpers import is_overridden

from egrecho.core.parser import CommonParser
from egrecho.utils.constants import DEFAULT_TRAIN_FILENAME


class LightningParser(CommonParser):
    """
    A convenient argument parser for pytorch lighting. referring:
        https://github.com/Lightning-AI/lightning/blob/master/src/lightning/pytorch/cli.py#LightningArgumentParser
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.add_cfg_flag()
        self.callback_keys: List[str] = []
        self.trainer_class = None

    def add_pl_module_args(
        self,
        pl_class: Union[
            Type[LightningModule], Type[LightningDataModule], Type[Callback]
        ],
        nested_key: str,
        subclass_mode: bool = False,
        **kwargs,
    ):
        """Adds pl module {LightningModule, LightningDataModule, Callback} arguments.

        Args:
            pl_class: Subclass of { LightningModule, LightningDataModule, Callback}.
            nested_key: Name of the nested namespace to store arguments.
            subclass_mode: Whether allow any subclass of the given class.
        """
        if pl_class is None:
            return
        if inspect.isclass(pl_class) and issubclass(
            cast(type, pl_class),
            (LightningModule, LightningDataModule, Callback),
        ):
            if issubclass(cast(type, pl_class), Callback):
                self.callback_keys.append(nested_key)
            return self.add_class_args(
                pl_class, nested_key, subclass_mode=subclass_mode, **kwargs
            )

        raise TypeError(
            f"Cannot add arguments from: {pl_class}. You should provide either a callable or a subclass of: "
            "LightningModule, LightningDataModule, or Callback."
        )

    def add_trainer_args(
        self,
        pl_trainer_class: Type[Trainer] = Trainer,
        nested_key: str = "trainer",
        **kwargs,
    ):
        """Adds pl trainer arguments.

        Args:
            pl_trainer_class: class of lightning trainer.
            nested_key: Name of the nested namespace to store arguments.
        """
        self.trainer_class = pl_trainer_class
        return self.add_class_arguments(
            pl_trainer_class,
            nested_key,
            fail_untyped=False,
            instantiate=False,
            **kwargs,
        )


class SaveConfigCallback(Callback):
    """Modified from `Lightning-AI`. Saves a LightningCLI config to the log_dir when training starts.

    Args:
        parser: The parser object used to parse the configuration.
        config: The parsed configuration that will be saved.
        config_filename: Filename for the config file.
        overwrite: Whether to overwrite an existing config file.
        skip_none: whether skip null while saving.
        multifile: When input is multiple config files, saved config preserves this structure.
        save_to_log_dir: Whether to save the config to the log_dir.

    Raises:
        RuntimeError: If the config file already exists in the directory to avoid overwriting a previous run
    """

    def __init__(
        self,
        parser: CommonParser,
        config: Namespace,
        config_filename: str = DEFAULT_TRAIN_FILENAME,
        overwrite: bool = True,
        skip_none: bool = True,
        multifile: bool = False,
        save_to_log_dir: bool = True,
    ) -> None:
        self.parser = parser
        self.config = config
        self.config_filename = config_filename
        self.overwrite = overwrite
        self.skip_none = skip_none
        self.multifile = multifile
        self.save_to_log_dir = save_to_log_dir
        self.already_saved = False

        if not save_to_log_dir and not is_overridden(
            "save_config", self, SaveConfigCallback
        ):
            raise ValueError(
                "`save_to_log_dir=False` only makes sense when subclassing SaveConfigCallback to implement "
                "`save_config` and it is desired to disable the standard behavior of saving to log_dir."
            )

    def setup(self, trainer: Trainer, pl_module: LightningModule, stage: str) -> None:
        if self.already_saved:
            return

        if self.save_to_log_dir:
            log_dir = trainer.log_dir  # this broadcasts the directory
            assert log_dir is not None
            config_path = os.path.join(log_dir, self.config_filename)
            fs = get_filesystem(log_dir)

            if not self.overwrite:
                # check if the file exists on rank 0
                file_exists = (
                    fs.isfile(config_path) if trainer.is_global_zero else False
                )
                # broadcast whether to fail to all ranks
                file_exists = trainer.strategy.broadcast(file_exists)
                if file_exists:
                    raise RuntimeError(
                        f"{self.__class__.__name__} expected {config_path} to NOT exist. Aborting to avoid overwriting"
                        " results of a previous run. You can delete the previous config file,"
                        " set `overwrite=True` to to overwrite the config file."
                    )

            if trainer.is_global_zero:
                # save only on rank zero to avoid race conditions.
                # the `log_dir` needs to be created as we rely on the logger to do it usually
                # but it hasn't logged anything at this point
                fs.makedirs(log_dir, exist_ok=True)
                self.parser.save(
                    self.config,
                    config_path,
                    skip_none=self.skip_none,
                    overwrite=self.overwrite,
                    multifile=self.multifile,
                )

        if trainer.is_global_zero:
            self.save_config(trainer, pl_module, stage)
            self.already_saved = True

        # broadcast so that all ranks are in sync on future calls to .setup()
        self.already_saved = trainer.strategy.broadcast(self.already_saved)

    def save_config(
        self, trainer: Trainer, pl_module: LightningModule, stage: str
    ) -> None:
        """Implement to save the config in some other place additional to the standard log_dir.

        Example:
            def save_config(self, trainer, pl_module, stage):
                if isinstance(trainer.logger, Logger):
                    config = self.parser.dump(self.config, skip_none=False)  # Required for proper reproducibility
                    trainer.logger.log_hyperparams({"config": config})

        Note:
            This method is only called on rank zero. This allows to implement a custom save config without having to
            worry about ranks or race conditions. Since it only runs on rank zero, any collective call will make the
            process hang waiting for a broadcast. If you need to make collective calls, implement the setup method
            instead.
        """
