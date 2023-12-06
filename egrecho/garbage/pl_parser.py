# -*- coding:utf-8 -*-

# Copyright xmuspeech (Author: Leo 2023-04-11)
# modified from Lightning-AI
# https://github.com/Lightning-AI/lightning/blob/master/src/lightning/pytorch/cli.py

import contextlib
import inspect
import functools
import os
from typing import (
    Any,
    Callable,
    Dict,
    List,
    Optional,
    Tuple,
    Type,
    TypeVar,
    Union,
    cast,
)

import torch
from jsonargparse import Namespace
from jsonargparse._actions import _ActionSubCommands
from lightning.fabric.utilities.cloud_io import get_filesystem
from lightning.pytorch import (
    Callback,
    LightningDataModule,
    LightningModule,
    Trainer,
    seed_everything,
)
from lightning.pytorch.utilities.model_helpers import is_overridden
from lightning.pytorch.utilities.rank_zero import rank_zero_warn
from torch.optim import Optimizer

from egrecho.core.data_builder import DataBuilder
from egrecho.core.module import DataMoudle
from egrecho.core.parser import BaseCommand, CommonParser
from egrecho.utils.constants import DEFAULT_PL_CONFIG_FILENAME

ClassType = TypeVar("ClassType")

LRSchedulerTypeTuple = (
    torch.optim.lr_scheduler._LRScheduler,
    torch.optim.lr_scheduler.ReduceLROnPlateau,
)
LRSchedulerType = Union[
    Type[torch.optim.lr_scheduler._LRScheduler],
    Type[torch.optim.lr_scheduler.ReduceLROnPlateau],
]


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

    def add_class_args(
        self,
        theclass: Type,
        nested_key: Optional[str] = None,
        subclass_mode: bool = False,
        instantiate: bool = True,
        **kwargs,
    ):
        """A convenient access of add class/subclass arguments.

        Args:
            theclass: Class from which to add arguments.
            nested_key: Key for nested namespace.
            subclass_mode: Whether allow any subclass of the given class.
            instantiate: Whether the class group should be instantiated by :code:`instantiate_classes`.
            **kwargs: other args will pass to `add_subclass_arguments/add_class_arguments` of jsonargparser.
        """
        if subclass_mode:
            return self.add_subclass_arguments(
                theclass, nested_key, required=True, instantiate=instantiate, **kwargs
            )
        return self.add_class_arguments(
            theclass, nested_key, fail_untyped=False, instantiate=instantiate, **kwargs
        )


class SaveConfigCallback(Callback):
    """Saves a LightningCLI config to the log_dir when training starts.

    Args:
        parser: The parser object used to parse the configuration.
        config: The parsed configuration that will be saved.
        config_filename: Filename for the config file.
        overwrite: Whether to overwrite an existing config file.
        multifile: When input is multiple config files, saved config preserves this structure.
        save_to_log_dir: Whether to save the config to the log_dir.

    Raises:
        RuntimeError: If the config file already exists in the directory to avoid overwriting a previous run
    """

    def __init__(
        self,
        parser: CommonParser,
        config: Namespace,
        config_filename: str = DEFAULT_PL_CONFIG_FILENAME,
        overwrite: bool = False,
        multifile: bool = False,
        save_to_log_dir: bool = True,
    ) -> None:
        self.parser = parser
        self.config = config
        self.config_filename = config_filename
        self.overwrite = overwrite
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
                        " set `LightningCLI(save_config_callback=None)` to disable config saving,"
                        ' or set `LightningCLI(save_config_kwargs={"overwrite": True})` to overwrite the config file.'
                    )

            if trainer.is_global_zero:
                # save only on rank zero to avoid race conditions.
                # the `log_dir` needs to be created as we rely on the logger to do it usually
                # but it hasn't logged anything at this point
                fs.makedirs(log_dir, exist_ok=True)
                self.parser.save(
                    self.config,
                    config_path,
                    skip_none=False,
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


class ParseModule:
    def __init__(
        self,
        parser: LightningParser,
        config: Namespace,
        save_config_callback: Optional[Type[SaveConfigCallback]] = SaveConfigCallback,
        save_config_filename: str = DEFAULT_PL_CONFIG_FILENAME,
        save_config_overwrite: bool = False,
    ):
        self.parser = parser
        self.config = config
        self.save_config_callback = save_config_callback
        self.save_config_filename = save_config_filename
        self.save_config_overwrite = save_config_overwrite
        if config.get("seed_everything", None) is not None:
            seed_everything(config["seed_everything"], workers=True)
        self.instantiate_classes()

    @classmethod
    def setup_parser(
        parser: LightningParser,
        model_class: Optional[Type[LightningModule]] = None,
        datamodule_class: Optional[Type[DataMoudle]] = Type[DataMoudle],
    ):
        parser.add_argument(
            "--seed_everything",
            type=int,
            default=42,
            help="Set to an int to run seed_everything with this value before classes instantiation",
        )
        parser.add_trainer_args()
        subcommands = parser.add_subcommands()
        data_command = LightningParser()
        data_command.add_pl_module_args(datamodule_class)
        subcommands.add_subcommand(data_command, datamodule_class.__name__)

    def instantiate_classes(self) -> None:
        """Instantiates the classes using settings from self.config."""
        self.config_init = self.parser.instantiate_classes(self.config)
        self.datamodule = self.config_init.get("data")
        self.model = self.config_init["model"]
        self.instantiate_trainer()

    def instantiate_datamodule(self) -> None:
        """Instantiates the classes using settings from self.config[subcommand]."""
        data_cfg = self.config_init.pop("subcommand")
        # sub_parser = self.parser.get_sub
        self.datamodule = self.config_init.get("data")
        self.model = self.config_init["model"]
        self.instantiate_trainer()

    def instantiate_trainer(self) -> None:
        """Instantiates the trainer using self.config_init['trainer']"""
        if self.config_init["trainer"].get("callbacks") is None:
            self.config_init["trainer"]["callbacks"] = []
        callbacks = [self.config_init[c] for c in self.parser.callback_keys]
        self.config_init["trainer"]["callbacks"].extend(callbacks)
        if "callbacks" in self.trainer_defaults:
            if isinstance(self.trainer_defaults["callbacks"], list):
                self.config_init["trainer"]["callbacks"].extend(
                    self.trainer_defaults["callbacks"]
                )
            else:
                self.config_init["trainer"]["callbacks"].append(
                    self.trainer_defaults["callbacks"]
                )
        if (
            self.save_config_callback
            and not self.config_init["trainer"]["fast_dev_run"]
        ):
            config_callback = self.save_config_callback(
                self.parser,
                self.config,
                self.save_config_filename,
                overwrite=self.save_config_overwrite,
            )
            self.config_init["trainer"]["callbacks"].append(config_callback)
        self.trainer = self.trainer_class(**self.config_init["trainer"])


# class TrainCommand(BaseCommand):
#     def __init__(
#         self,
#         parser: LightningParser,
#         config: Namespace,
#         save_config_callback: Optional[Type[SaveConfigCallback]] = SaveConfigCallback,
#         save_config_filename: str = DEFAULT_PL_CONFIG_FILENAME,
#         save_config_overwrite: bool = False,
#     ):
#         self.parser = parser
#         self.config = config
#         self.save_config_callback = save_config_callback
#         self.save_config_filename = save_config_filename
#         self.save_config_overwrite = save_config_overwrite
#         self.link_optimizers_and_lr_schedulers()
#         if config.get("seed_everything", None) is not None:
#             seed_everything(config["seed_everything"], workers=True)
#         self.instantiate_classes()

#     @classmethod
#     def setup_parser(
#         parser: LightningParser,
#         model_class: Optional[Type[LightningModule]] = None,
#         datamodule_class: Optional[Type[LightningDataModule]] = None,
#         datamodule_attributes=None,
#     ):
#         parser.add_argument(
#             "--seed_everything",
#             type=int,
#             default=42,
#             help="Set to an int to run seed_everything with this value before classes instantiation",
#         )
#         parser.add_trainer_args()
#         # add default optimizer args if necessary
#         if auto_configure_optimizers:
#             if (
#                 not parser._optimizers
#             ):  # already added by the user in `add_arguments_to_parser`
#                 parser.add_optimizer_args((Optimizer,))
#             if (
#                 not parser._lr_schedulers
#             ):  # already added by the user in `add_arguments_to_parser`
#                 parser.add_lr_scheduler_args(LRSchedulerTypeTuple)

#     @staticmethod
#     def run_from_args(args: Namespace, parser: Optional[CommonParser] = None, **kwargs):
#         """
#         Run this command with args.
#         """
#         raise NotImplementedError

#     def instantiate_classes(self) -> None:
#         """Instantiates the classes using settings from self.config."""
#         self.config_init = self.parser.instantiate_classes(self.config)
#         self.datamodule = self.config_init.get("data")
#         self.model = self.config_init["model"]
#         self.instantiate_trainer()

#     def instantiate_trainer(self) -> None:
#         """Instantiates the trainer using self.config_init['trainer']"""
#         if self.config_init["trainer"].get("callbacks") is None:
#             self.config_init["trainer"]["callbacks"] = []
#         callbacks = [self.config_init[c] for c in self.parser.callback_keys]
#         self.config_init["trainer"]["callbacks"].extend(callbacks)
#         if "callbacks" in self.trainer_defaults:
#             if isinstance(self.trainer_defaults["callbacks"], list):
#                 self.config_init["trainer"]["callbacks"].extend(
#                     self.trainer_defaults["callbacks"]
#                 )
#             else:
#                 self.config_init["trainer"]["callbacks"].append(
#                     self.trainer_defaults["callbacks"]
#                 )
#         if (
#             self.save_config_callback
#             and not self.config_init["trainer"]["fast_dev_run"]
#         ):
#             config_callback = self.save_config_callback(
#                 self.parser,
#                 self.config,
#                 self.save_config_filename,
#                 overwrite=self.save_config_overwrite,
#             )
#             self.config_init["trainer"]["callbacks"].append(config_callback)
#         self.trainer = self.trainer_class(**self.config_init["trainer"])


def _global_add_class_path(
    class_type: Type, init_args: Optional[Union[Namespace, Dict[str, Any]]] = None
) -> Dict[str, Any]:
    if isinstance(init_args, Namespace):
        init_args = init_args.as_dict()
    return {
        "class_path": class_type.__module__ + "." + class_type.__name__,
        "init_args": init_args,
    }


def _add_class_path_generator(
    class_type: Type,
) -> Callable[[Dict[str, Any]], Dict[str, Any]]:
    def add_class_path(init_args: Dict[str, Any]) -> Dict[str, Any]:
        return _global_add_class_path(class_type, init_args)

    return add_class_path


def instantiate_class(args: Union[Any, Tuple[Any, ...]], init: Dict[str, Any]) -> Any:
    """Instantiates a class with the given args and init.

    Args:
        args: Positional arguments required for instantiation.
        init: Dict of the form {"class_path":...,"init_args":...}.

    Returns:
        The instantiated class object.

    """
    kwargs = init.get("init_args", {})
    if not isinstance(args, tuple):
        args = (args,)
    class_module, class_name = init["class_path"].rsplit(".", 1)
    module = __import__(class_module, fromlist=[class_name])
    args_class = getattr(module, class_name)
    return args_class(*args, **kwargs)
