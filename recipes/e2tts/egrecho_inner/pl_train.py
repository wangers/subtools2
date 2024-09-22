# -*- coding:utf-8 -*-
# Copyright xmuspeech (Author: Leo 2023-09)

import os
from pathlib import Path
from typing import Any, List, Optional, Union

import lightning.pytorch as L
import torch
from data_builder import LhotseBuilder, LhotseBuilderConfig
from datamodule import DataModule
from jsonargparse import Namespace
from lightning.pytorch import seed_everything
from lightning.pytorch.callbacks import (
    Callback,
    ModelCheckpoint,
    RichModelSummary,
    RichProgressBar,
)
from lightning.pytorch.loggers import Logger, TensorBoardLogger
from utils import plot_spectrogram

from egrecho.core.loads import resolve_ckpt
from egrecho.core.parser import BaseCommand
from egrecho.core.pl_parser import LightningParser, SaveConfigCallback
from egrecho.models.e2_tts.model import E2TTS, E2TTSConfig
from egrecho.models.e2_tts.task import E2TTSTeacher, EdenEpochCallback
from egrecho.utils.constants import DEFAULT_MODEL_FILENAME
from egrecho.utils.cuda_utils import release_memory
from egrecho.utils.logging import _infer_rank, get_logger
from egrecho_cli.register import register_command

logger = get_logger(__name__)

DESCRIPTION = "Train e2tts"


@register_command(name="train-e2tts", aliases=["train-e2"], help=DESCRIPTION)
class TrainE2TTS(BaseCommand):
    @classmethod
    def get_dummy_parser(cls) -> LightningParser:
        return LightningParser(description=DESCRIPTION)

    @classmethod
    def setup_parser(cls, parser: LightningParser):

        parser.add_argument(
            "--save_dir",
            type=Union[Path, None],
            default='exp/e2tts_test',
            help="Default path for save logs and weights, Default: ``os.getcwd()``. If None will be ``os.getcwd()``.",
        )
        parser.add_argument(
            "--seed_everything",
            type=int,
            default=42,
            help="Set to an int to run seed_everything with this value before classes instantiation.",
        )

        parser.add_class_args(
            LhotseBuilderConfig,
            "data",
        )
        parser.add_class_args(
            LhotseBuilder,
            "data_builder",
        )
        parser.link_arguments("data", "data_builder.config", apply_on="instantiate")
        parser.add_class_args(
            E2TTSConfig,
            "model",
        )
        parser.link_arguments(
            "data_builder.vocab_size", "model.vocab_size", apply_on="instantiate"
        )
        parser.link_arguments(
            "data_builder.pad_text_token_id",
            "model.pad_text_token_id",
            apply_on="instantiate",
        )
        parser.link_arguments(
            "data_builder.pad_feats_val",
            "model.pad_feats_val",
            apply_on="instantiate",
        )

        parser.add_argument(
            "--resume_ckpt",
            type=Optional[str],
            default=None,
            help="Path/URL of the checkpoint from which training is resumed. Could also be "
            'one of two special keywords ``"last"`` and ``"hpc"``. If there is no checkpoint file at '
            "the path, an exception is raised.",
        )
        parser.add_argument(
            "--init_weight",
            "-init-wt",
            action="store_true",
            help="Whether init model weight, and you must proved params --init_weight_params to resolve a ckpt. "
            "Similar but different to --resume_ckpt, this init weight will ignore other previous training status. "
            "(e.g., optimizer & lr scheduler).",
        )
        parser.add_function_arguments(
            resolve_ckpt,
            "init_weight_params",
        )
        parser.add_class_args(
            E2TTSTeacher,
            "teacher",
        )

        parser.add_trainer_args(skip={"logger", "callbacks", "barebones"})
        parser.add_pl_module_args(
            ModelCheckpoint,
            "mckpt",
        )
        parser.add_argument(
            "use_compile", type=bool, default=False, help="Whether use torch.compile"
        )
        return parser

    @staticmethod
    def run_from_args(args, parser: LightningParser):
        executor = TrainE2TTS(args, parser)
        executor.run()

    def __init__(
        self,
        args: Namespace,
        parser: LightningParser,
    ):
        self.save_dir = os.fspath(args.save_dir) if args.save_dir else os.getcwd()
        self.config_tosave = args
        self.parser = parser

        # self._save_config_callback_init = None

        args = args.clone()

        seed_everything(args.seed_everything, workers=True)
        self.cfg_init = parser.instantiate_classes(args)
        self.data_builder: LhotseBuilder = self.cfg_init.data_builder

        # wrapper datamodule
        self.data = DataModule(self.data_builder)
        # instance model & teacher
        model_cfg = self.cfg_init.model
        self.model = E2TTS(model_cfg)
        self.teacher = self.cfg_init.teacher
        self.setup_model_teacher()

        # resume whole training or just initiate weights from previous ckpt.
        self.resume_ckpt = self.cfg_init["resume_ckpt"]
        self.init_weight = self.cfg_init["init_weight"]
        if bool(self.resume_ckpt) and self.init_weight:
            logger.warning(
                f"Got both resume_ckpt={self.resume_ckpt} and init_weight={self.init_weight}, "
                f"invalids the init_weight and resumes this tranining from resume_ckpt."
            )
            self.init_weight = False

        # TODO: move this to a callback?
        if self.init_weight:
            self.init_weight_params = self.cfg_init["init_weight_params"]
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
        self.model_checkpoint: ModelCheckpoint = self.cfg_init["mckpt"]

        self.instantiate_trainer()

        # prepare fit
        self.prepare_fit_kwargs()

        # TODO: move this out of cli and consider rank problem
        self.save_ckpt_conf_dir()

    def setup_model_teacher(self):
        """Link model with teacher."""

        self.model.setup_teacher(self.teacher)

    def instantiate_trainer(self) -> None:
        """Instantiates the trainer using self.subcommand_init['trainer']"""
        self._gather_callbacks(self.cfg_init, self.parser)

        self.cfg_init["trainer"]["logger"] = self._get_train_loggers()
        self.trainer = self.parser.trainer_class(**self.cfg_init["trainer"])

    def _get_train_loggers(self) -> Optional[List[Logger]]:
        """Use CSVLogger and TensorBoardLogger."""
        if not bool(self.cfg_init["trainer"].get("barebones")):
            return [
                TensorBoardLogger(save_dir=self.save_dir, name="", version=''),
            ]
        return None

    def _get_default_callbacks(self) -> Optional[List[Callback]]:
        from egrecho.training.callbacks import ScalarBatchCallback

        callbacks = [
            RichProgressBar(refresh_rate=1, leave=True),
            EdenEpochCallback(),
            RichModelSummary(max_depth=3),
            ScalarBatchCallback(),
            PlotMelCallback(),
            # LearningRateMonitor(),
        ]
        return callbacks

    def _gather_callbacks(self, config: Namespace, parser: LightningParser):
        """Gathers callbacks.

        From:
            - `config['trainer']['callbacks']`
            - `self.default_callbacks`: `[RichProgressBar, LearningRateMonitor, ...]`
            - `config['callbacks']`
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
        config["trainer"]["callbacks"] += self._get_default_callbacks()
        # parser cbs
        config["trainer"]["callbacks"] += [config[c] for c in parser.callback_keys]

        # SaveConfigCallback
        if not config["trainer"]["fast_dev_run"]:
            config_callback = SaveConfigCallback(
                self.parser,
                self.config_tosave,
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
        if (_infer_rank() or 0) == 0 and not self.cfg_init["trainer"]["fast_dev_run"]:
            ckpt_dir = Path(self.trainer.log_dir) / "checkpoints"
            if not self.resume_ckpt or not ckpt_dir.exists():
                ckpt_dir = Path(self.trainer.log_dir) / "checkpoints"
                tokenizer = self.data_builder.tokenizer
                feats_extractor = self.data_builder.feature_extractor
                self.model.save_load_helper.save_to(
                    ckpt_dir, components=(tokenizer, feats_extractor)
                )
                self.model.to_cfg_file(ckpt_dir / "config" / DEFAULT_MODEL_FILENAME)

    def run(self):
        """Fit."""
        self.trainer.fit(**self.fit_kwargs)
        if not self.cfg_init["trainer"]["fast_dev_run"]:
            self.model_checkpoint.to_yaml()


class PlotMelCallback(Callback):
    def __init__(
        self,
    ) -> None:
        self.last_generated_mel = None
        self.last_ref_mel = None
        self.last_cond_mask = None

    def on_validation_batch_end(
        self,
        trainer: "L.Trainer",
        pl_module: "L.LightningModule",
        outputs: dict,
        batch: Any,
        batch_idx: int,
        dataloader_idx: int = 0,
    ) -> None:

        self.last_generated_mel = outputs['melouts'].mel[-1]
        self.last_cond_mask = outputs['melouts'].cond_mask[-1]
        self.last_ref_mel = outputs['ref_mels'][-1]

    def on_validation_epoch_end(
        self, trainer: "L.Trainer", pl_module: "L.LightningModule"
    ) -> None:
        if (
            trainer.is_global_zero
            and trainer.logger
            and isinstance(trainer.logger, TensorBoardLogger)
        ):
            trainer.logger.experiment.add_figure(
                "mel/target",
                plot_spectrogram(self.last_ref_mel.detach().cpu().numpy()),
                trainer.global_step,
            )
            trainer.logger.experiment.add_figure(
                "mel/prediction",
                plot_spectrogram(self.last_generated_mel.detach().cpu().numpy()),
                trainer.global_step,
            )
            # (T, 1) * (T, F)
            cond_mel = self.last_cond_mask.float()[..., None] * self.last_ref_mel
            trainer.logger.experiment.add_figure(
                "mel/cond",
                plot_spectrogram(cond_mel.detach().cpu().numpy()),
                trainer.global_step,
            )
            self.last_generated_mel = None
            self.last_ref_mel = None
            self.last_cond_mask = None


if __name__ == "__main__":
    parser = TrainE2TTS.get_dummy_parser()
    parser = TrainE2TTS.setup_parser(parser)
    args = parser.parse_args()
    logger.info(
        f"Got parsed args: \n{parser.dump(args.clone(),skip_default=True)}", ranks=[0]
    )
    TrainE2TTS.run_from_args(args, parser)
