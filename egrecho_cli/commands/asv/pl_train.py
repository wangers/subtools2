# -*- coding:utf-8 -*-
# Copyright xmuspeech (Author: Leo 2023-09)

import os
import traceback
from pathlib import Path
from typing import Any, Dict, List, Optional, Type, Union

import torch
from jsonargparse import Namespace, lazy_instance
from lightning.pytorch import seed_everything
from lightning.pytorch.callbacks import (
    Callback,
    EarlyStopping,
    LearningRateMonitor,
    ModelCheckpoint,
    RichProgressBar,
)
from lightning.pytorch.loggers import CSVLogger, Logger, TensorBoardLogger
from lightning.pytorch.utilities import rank_zero_warn

from egrecho.core.loads import resolve_ckpt, save_ckpt_conf_dir
from egrecho.core.module import DataModule, TopVirtualModel
from egrecho.core.parser import BaseCommand
from egrecho.core.pl_parser import LightningParser, SaveConfigCallback
from egrecho.core.teacher import Teacher
from egrecho.data.builder.asv import ASVPipeBuilder, DataBuilder
from egrecho.models.groups.asv_group import SVTeacher, XvectorMixin
from egrecho.training.callbacks import DataSetEpochCallback, LastBatchPatchCallback
from egrecho.utils.cuda_utils import release_memory
from egrecho.utils.logging import _infer_rank, get_logger
from egrecho_cli.register import register_command

logger = get_logger()

DESCRIPTION = "Train asv model"


@register_command(name="train-asv", aliases=[], help=DESCRIPTION)
class TrainASV(BaseCommand):
    @classmethod
    def get_dummy_parser(cls) -> LightningParser:
        return LightningParser(description=DESCRIPTION)

    @classmethod
    def setup_parser(cls, parser: LightningParser):
        try:
            parser.add_argument(
                "--save_dir",
                type=Union[Path, None],
                default=None,
                help="Default path for save logs and weights, Default: ``os.getcwd()``. If None will be ``os.getcwd()``.",
            )
            parser.add_argument(
                "--seed_everything",
                type=int,
                default=42,
                help="Set to an int to run seed_everything with this value before classes instantiation.",
            )

            parser.add_class_args(
                DataBuilder,
                "data_builder",
                default=lazy_instance(
                    ASVPipeBuilder, config={"class_path": "ASVBuilderConfig"}
                ),
                subclass_mode=True,
            )
            parser.add_argument(
                "--data_attrs",
                type=set,
                default={"inputs_dim", "num_classes"},
                help="Data attributes infers from data builder will pass to model/teacher before instantiatting.",
            )

            parser.add_class_args(DataModule, "data", skip={"batch_size"})

            parser.link_arguments(
                "data_builder", "data.builder", apply_on="instantiate"
            )
            subcommands = parser.add_subcommands(title=None)
            train_parser = LightningParser(description="Run trainer")
            subcommands.add_subcommand("run", train_parser)

            train_parser.add_subclass_arguments(
                XvectorMixin,
                "model",
                fail_untyped=False,
            )

            train_parser.add_argument(
                "--resume_ckpt",
                type=Optional[str],
                default=None,
                help="Path/URL of the checkpoint from which training is resumed. Could also be "
                'one of two special keywords ``"last"`` and ``"hpc"``. If there is no checkpoint file at '
                "the path, an exception is raised.",
            )
            train_parser.add_argument(
                "--init_weight",
                "-init-wt",
                action="store_true",
                help="Whether init model weight, and you must proved params --init_weight_params to resolve a ckpt. "
                "Similar but different to --resume_ckpt, this init weight will ignore other previous training status. "
                "(e.g., optimizer & lr scheduler).",
            )
            train_parser.add_function_arguments(
                resolve_ckpt,
                "init_weight_params",
            )
            train_parser.add_subclass_arguments(
                Teacher,
                "teacher",
                default=lazy_instance(SVTeacher),
            )

            train_parser.add_trainer_args(skip={"logger"})
            train_parser.add_pl_module_args(
                ModelCheckpoint,
                "mckpt",
            )
            train_parser.add_argument(
                "--use_early_stopping",
                "-es",
                type=bool,
                default=False,
                help="Whether use early stopping callback set by following early_stoping.",
            )

            train_parser.add_class_args(
                EarlyStopping,
                "early_stopping",
            )
            train_parser.add_argument(
                "--set_dataset_epoch",
                type=bool,
                help="Wheter call the dataset's `set_epoch` method, this is useful for "
                "IterableDataset if it has `set_epoch` method while its sampler is infinite.",
                default=True,
            )
            train_parser.add_argument(
                "--patch_last_b",
                type=bool,
                help="Wheter patch to set last batch flag in fit loop in IterableDataset case, "
                "can remove when lighning fix this or manually handle it.",
                default=True,
            )
            displaydata_parser = LightningParser(description="Display data.")
            subcommands.add_subcommand("display", displaydata_parser)
        except Exception as ex:  # noqa
            rank_zero_warn(
                f"{traceback.format_exc()}\nSome errors occurs while adding args, skip it and this parser "
                "is invalid. You need check and fix it."
            )
        return parser

    @staticmethod
    def run_from_args(args, parser: LightningParser):
        executor = TrainASV(args, parser)
        executor.run()

    def __init__(
        self,
        args: Namespace,
        parser: LightningParser,
        trainer_defaults: Dict[str, Any] = None,
        save_config_callback: Optional[Type[SaveConfigCallback]] = SaveConfigCallback,
        save_config_kwargs: Optional[Dict[str, Any]] = None,
    ):
        self.save_dir = os.fspath(args.save_dir) if args.save_dir else os.getcwd()
        self.config_tosave = args
        self.parser = parser
        self.trainer_defaults = trainer_defaults or {}
        self.default_callbacks = []
        self.save_config_callback = save_config_callback
        self.save_config_kwargs = save_config_kwargs or {}

        # self._save_config_callback_init = None

        args = args.clone()
        self.subcommand_name = args.pop("subcommand")

        self.sub_config = args.pop(self.subcommand_name) if self.subcommand_name else {}
        self.sub_parser = (
            parser._subcommands_action._name_parser_map.get(self.subcommand_name)
            if self.subcommand_name
            else None
        )

        seed_everything(args.seed_everything, workers=True)
        data_cfg_init = parser.instantiate_classes(args)
        self.data = data_cfg_init["data"]
        self.data_builder = data_cfg_init["data_builder"]
        if self.subcommand_name == "run":
            self.data_attrs_map = {
                data_attr: getattr(self.data_builder, data_attr, None)
                for data_attr in data_cfg_init["data_attrs"]
            }
            self.link_data_attr()
            self.config_tosave["run"] = self.sub_config

            # initiate model
            self.subcommand_init = self.sub_parser.instantiate_classes(self.sub_config)
            self.model: TopVirtualModel = self.subcommand_init["model"]
            self.teacher: Teacher = self.subcommand_init["teacher"]
            self.setup_model_teacher()

            # resume whole training or just initiate weights from previous ckpt.
            self.resume_ckpt = self.subcommand_init["resume_ckpt"]
            self.init_weight = self.subcommand_init["init_weight"]
            if bool(self.resume_ckpt) and self.init_weight:
                logger.warning(
                    f"Got both resume_ckpt={self.resume_ckpt} and init_weight={self.init_weight}, "
                    f"invalids the init_weight and resumes this tranining from resume_ckpt."
                )
                self.init_weight = False

            # TODO: move this to a callback?
            if self.init_weight:
                self.init_weight_params = self.subcommand_init["init_weight_params"]
                init_ckpt = resolve_ckpt(**self.init_weight_params)

                logger.info(
                    f"Loading {type(self.model).__name__} from ckpt ({init_ckpt}) to cpu, "
                    "skips mismatch weight keys with strict=False.",
                    ranks=0,
                )
                states = torch.load(init_ckpt, map_location="cpu")
                states = states["state_dict"] if "state_dict" in states else states
                self.model.load_state_dict(states, strict=False)
                release_memory(states)

            # callbacks
            self.model_checkpoint: ModelCheckpoint = self.subcommand_init["mckpt"]
            self.default_callbacks += self._get_default_callbacks()
            self.instantiate_trainer()

            # prepare fit
            self.prepare_fit_kwargs()

            # TODO: move this out of cli and consider rank problem
            self.save_ckpt_conf_dir()

    def link_data_attr(self):
        """Links outside data atters (e.g., num_classes) to subcommand `run`."""
        model_cfg = self.sub_config["model"]
        teacher_cfg = self.sub_config["teacher"]
        for data_attr in self.data_attrs_map:
            for cfg in (model_cfg, teacher_cfg):
                if (
                    self.data_attrs_map[data_attr] is not None
                    and data_attr in cfg["init_args"]
                ):
                    cfg["init_args"][data_attr] = self.data_attrs_map[data_attr]

    def setup_model_teacher(self):
        """Link model with teacher."""
        self.model.setup_teacher(self.teacher)

    def instantiate_trainer(self) -> None:
        """Instantiates the trainer using self.subcommand_init['trainer']"""
        self._gather_callbacks(self.subcommand_init, self.sub_parser)

        self.subcommand_init["trainer"]["logger"] = self._get_train_loggers()
        self.trainer = self.sub_parser.trainer_class(**self.subcommand_init["trainer"])

    def _get_train_loggers(self) -> Optional[List[Logger]]:
        """Use CSVLogger and TensorBoardLogger."""
        if not bool(self.subcommand_init["trainer"].get("barebones")):
            return [
                TensorBoardLogger(save_dir=self.save_dir, name=""),
                CSVLogger(save_dir=self.save_dir, name="csv_logs"),
            ]
        return None

    def _get_default_callbacks(self) -> Optional[List[Callback]]:
        callbacks = [
            RichProgressBar(refresh_rate=1, leave=True),
            # LearningRateMonitor(),
        ]
        if self.subcommand_init.get("use_early_stopping", False):
            callbacks.append(self.subcommand_init["early_stopping"])
        if self.subcommand_init.get("set_dataset_epoch"):
            callbacks.append(DataSetEpochCallback())
        if self.subcommand_init.get("patch_last_b"):
            callbacks.append(LastBatchPatchCallback())
        return callbacks

    def _gather_callbacks(self, config: Namespace, parser: LightningParser):
        """Gathers callbacks.

        From:
            - `config['trainer']['callbacks']`
            - `self.default_callbacks`: `[RichProgressBar, LearningRateMonitor, ...]`
            - `config['callbacks']`
            - `self.trainer_defaults['callbacks']`
            - `SaveConfigCallback`

        To: `config['trainer']['callbacks']`

        Args:
            config:
                Parsed Namespace via parser which contains instantiated callbacks.
            parser:
                Corresponded parser of `config`.
        """
        # trainer cbs
        if config["trainer"].get("callbacks") is None:
            config["trainer"]["callbacks"] = []

        # default cbs
        config["trainer"]["callbacks"] += self.default_callbacks
        # parser cbs
        config["trainer"]["callbacks"] += [config[c] for c in parser.callback_keys]

        # trainer_defaults['callbacks']
        if "callbacks" in self.trainer_defaults:
            value = self.trainer_defaults["callbacks"]
            config["trainer"]["callbacks"] += (
                value if isinstance(value, list) else [value]
            )

        # SaveConfigCallback
        if self.save_config_callback and not config["trainer"]["fast_dev_run"]:
            config_callback = self.save_config_callback(
                self.parser,
                self.config_tosave,
                **self.save_config_kwargs,
            )
            config["trainer"]["callbacks"].append(config_callback)
            # self._save_config_callback_init = config_callback

    def prepare_fit_kwargs(self) -> None:
        """Prepares fit_kwargs including datamodule."""
        self.fit_kwargs = {"model": self.model}
        self.fit_kwargs["datamodule"] = self.data
        self.fit_kwargs["ckpt_path"] = self.resume_ckpt

    # TODO: move this out of cli and consider rank problem
    def save_ckpt_conf_dir(self):

        # and self.resume_ckpt is None
        if (_infer_rank() or 0) == 0 and not self.subcommand_init["trainer"][
            "fast_dev_run"
        ]:
            ckpt_dir = Path(self.trainer.log_dir) / "checkpoints"
            model_type = self.model.__class__

            extractor = self.data_builder.feature_extractor
            save_ckpt_conf_dir(ckpt_dir, model_type=model_type, extractor=extractor)

    def run(self):
        if self.subcommand_name == "run":
            self.run_train()
        elif self.subcommand_name == "display":
            self.run_display()

    def run_train(self):
        """Fit."""
        self.trainer.fit(**self.fit_kwargs)
        if not self.subcommand_init["trainer"]["fast_dev_run"]:
            self.model_checkpoint.to_yaml()

    def run_display(self):
        """TODO: display data."""
        os.makedirs(self.save_dir, exist_ok=True)
        self.parser.save(
            self.config_tosave,
            os.path.join(self.save_dir, "train_data.yaml"),
            overwrite=True,
            skip_none=True,
        )


if __name__ == "__main__":
    pass
