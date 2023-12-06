# -*- coding:utf-8 -*-
# Copyright xmuspeech (Author: Leo 2023-07)

import collections
import functools
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, Dict, List, Literal, Optional, Tuple, Union

import torch

import egrecho.data.datasets.audio.functional as audio_functional
from egrecho.core.data_builder import DataBuilder, DataBuilderConfig
from egrecho.data.builder.voyages import VoyageTemplate, build_table
from egrecho.data.datasets.audio.samples import ASVSamples
from egrecho.data.datasets.constants import (
    AUDIO_COLUMN,
    OFFLINE_FEAT_COLUMN,
    SAMPLE_RATE_COLUMN,
    SPEAKER_COLUMN,
)
from egrecho.data.features.feature_extractor_audio import (
    KaldiFeatureExtractor,
    OfflineFeatureExtractor,
)
from egrecho.data.iterable import Processor, processors
from egrecho.utils.common import dict_union
from egrecho.utils.data_utils import ClassLabel, Split, get_num_batch
from egrecho.utils.io import resolve_file
from egrecho.utils.logging import get_logger

logger = get_logger()
if TYPE_CHECKING:
    from torch.utils.data import IterableDataset

AUDIO_SKETCH = (
    "id",
    AUDIO_COLUMN,
    SAMPLE_RATE_COLUMN,
    SPEAKER_COLUMN,
)
OFFFLINE_FEAT_SKETCH = ("features", "lens", "labels")


@dataclass
class ASVTemplate(VoyageTemplate):
    task: str = "automatic-speaker-verification"
    offline_feat: bool = False
    start_sketch: Tuple[str, ...] = (
        "id",
        AUDIO_COLUMN,
        SAMPLE_RATE_COLUMN,
        SPEAKER_COLUMN,
    )
    end_sketch: Tuple[str, ...] = ("features", "lens", "labels")

    def __post_init__(self):
        if self.offline_feat:
            self.start_sketch = ("id", OFFLINE_FEAT_COLUMN, SPEAKER_COLUMN)
        super().__post_init__()


def get_extractor_param():
    return {
        "mean_norm": True,
        "std_norm": False,
        "return_attention_mask": False,
        "feat_conf": {"feature_type": "kaldi-fbank"},
    }


@dataclass
class ASVBuilderConfig(DataBuilderConfig):
    """
    `ASVBuilderConfig` extends `PipeBuilderConfig` and represents the configuration for ASV data pipeline builder.

    Args:
        data_type (Literal['shard', 'raw', 'offline_feat']):
            The type of data, which can be one of 'shard', 'raw', or 'offline_feat'.
            Defaults to 'shard'.
        lazyload_source (bool):
            Whether to lazy-load manifest. Defaults to False.
        partition (bool):
            Passing to sampler. Whether apply sharding according to the ddp env. Defaults to True.
        shuffle (bool):
            Passing to sampler. Defaults to True.
        exbuild_src_kws (Dict):
            Any extend args to build src datapipe.
        start_sketch (tuple):
            This means there will be a validation of colunms' name when building datapipe, e.g.,
            start_sketch=('id', 'audio', 'sample_rate', 'speaker'),
            we can call :method:``egrecho.core.voyages.build_table`` to make sure the
            desired columns are existed, and whether to drop extras columns.
        label_fname (str):
            class label mapping file. rel to `data_dir` (speaker.yaml) or abs path
            if given abs pattern (/path/speaker.yaml)
        resample_rate (int):
            Global sample_rate, applys resamplings if accept a wav with different sample_rate. Defaults to 16000.
        pre_sp_factors (tuple[float]):
            If not provided (None), will auto initiate with `(0.9, 1.0, 1.1)` for training split only.
            Defaults to None. This will change speaker label.
        rand_chunksize (int):
            fix chunk training, defaults to 200 means 2s audio with 0.01 `frame_shift`.
        frame_shift (float):
            frame shift in `seconds(s)`. Defaults to 0.01.
        speech_aug (bool):
            wheter do signal-level augment, only apply on train split. Defaults to True.
        speech_aug_config (dict):
            config for speechaug. see :class:`egrecho.data.audio.functional.SpeechAugPipline`.
        shard_shuffle_size (int):
            buffer shuffle size of shuffeling for webdataset-like data loading.
        batch_size (int):
            batch size is defined here, defaults to 128. For test set force change to 1.
        drop_last (bool):
            defaults to True
        extractor_param (dict):
            args for kaldi feature extractor.

    """

    data_type: Literal["shard", "raw", "offline_feat"] = "shard"
    shuffle: bool = True
    partition: bool = True
    start_sketch: Optional[Tuple[str, ...]] = None
    exbuild_src_kw: Dict[str, Any] = None
    label_fname: str = "speaker.yaml"
    filter_conf: Optional[Dict] = field(
        default_factory=lambda: dict(max_length=15.0, truncate=True)
    )
    resample_rate: int = 16_000
    pre_sp_factors: Optional[Tuple[float, ...]] = field(
        default_factory=lambda: (0.9, 1.0, 1.1)
    )

    rand_chunksize: Optional[int] = 200
    frame_shift: float = 0.01

    speech_aug: bool = True
    speech_aug_config: Optional[Dict] = field(
        default_factory=lambda: dict(batch_size=1, db_dir="/data2/ldx/speech_aug")
    )
    shard_shuffle_size: int = 1500

    batch_size: int = 128
    drop_last: bool = True

    extractor_param: dict = field(default_factory=lambda: get_extractor_param())

    def __post_init__(self):
        super().__post_init__()
        sample_rate = self.resample_rate
        self.extractor_param = dict_union(get_extractor_param(), self.extractor_param)
        self.extractor_param["feat_conf"]["sampling_rate"] = sample_rate
        self.speech_aug_config["sample_rate"] = sample_rate
        if self.start_sketch is None:
            self.start_sketch = (
                OFFFLINE_FEAT_SKETCH
                if self.data_type == "offline_feat"
                else AUDIO_SKETCH
            )
        self.exbuild_src_kw = self.exbuild_src_kw or {}

    @property
    def build_src_kwargs(self):
        return dict(
            data_type=self.data_type,
            shuffle=self.shuffle,
            partition=self.partition,
            **self.exbuild_src_kw,
        )

    @property
    def train_mode(self):
        if not hasattr(self, "__train_mode"):
            self.__train_mode = True
        return self.__train_mode

    @train_mode.setter
    def train_mode(self, mode: bool):
        self.__train_mode = mode


class ASVPipeBuilder(DataBuilder):
    CONFIG_CLS = ASVBuilderConfig

    def __init__(self, config: ASVBuilderConfig):
        super().__init__(config)

    def get_val_config(self):
        cfg = ASVBuilderConfig.from_config(
            self.config,
            shuffle=False,
            filter_conf=None,
            partition=False,
            speech_aug=False,
        )
        cfg.train_mode = False
        return cfg

    def get_test_config(self):
        cfg = ASVBuilderConfig.from_config(
            self.config,
            shuffle=False,
            partition=False,
            speech_aug=False,
            rand_chunksize=None,
            filter_conf=None,
            label_fname=None,
            batch_size=1,
            drop_last=False,
        )
        cfg.train_mode = False
        return cfg

    def train_dataset(
        self,
    ) -> "IterableDataset":
        return self.get_raw_dataset(Split.TRAIN, self.config)

    def val_dataset(self) -> "IterableDataset":
        return self.get_raw_dataset(Split.VALIDATION, self.get_val_config())

    def test_dataset(self) -> "IterableDataset":
        return self.get_raw_dataset(Split.TEST, self.get_test_config())

    def get_raw_dataset(
        self,
        split: Optional[Union[str, Split]],
        config: ASVBuilderConfig,
    ) -> "IterableDataset":
        """Builds a single datapipe through split config."""
        split_files = self.data_files[split]
        datapipe = ASVSamples.build_source_dataset(
            split_files, **config.build_src_kwargs
        )

        # format dict sample
        build_table_fn = functools.partial(build_table, sketch=config.start_sketch)
        datapipe = Processor(datapipe, processors.maps, build_table_fn)

        # element-wise
        datapipe = self._ele_processor(datapipe, config)

        # local shuffle
        if config.data_type == "shard" and config.shuffle:
            datapipe = Processor(
                datapipe,
                processors.buffer_shuffle,
                config.shard_shuffle_size,
            )

        # speaker2label
        if config.label_fname:
            class_label = self.class_label
            datapipe = Processor(
                datapipe,
                processors.maps,
                class_label.encode_label,
                input_col=SPEAKER_COLUMN,
                output_col="label",
            )

        # batch
        datapipe = datapipe.apply(
            processors.batch, config.batch_size, drop_last=config.drop_last
        )

        # extract feature
        batch_exc = BatchExtractor(config.extractor_param, data_type=config.data_type)
        datapipe = Processor(
            datapipe,
            processors.maps,
            batch_exc,
        )

        # bound __len__ method.
        if infer_nsamples := datapipe.src_length():
            length = get_num_batch(infer_nsamples, config.batch_size, config.drop_last)
            datapipe.add_length(length)
        return datapipe

    @classmethod
    def _ele_processor(self, datapipe: Processor, config: ASVBuilderConfig):
        """
        Applys processors on element-wise audio samples (i.e., not batch).

        select_channel -> filter -> resample -> speed perturb -> random chunk -> speechaug
        """
        if config.data_type in ("raw", "shard"):
            # select first channel.
            datapipe = Processor(datapipe, audio_functional.select_channel)
            if config.filter_conf:
                datapipe = Processor(
                    datapipe, audio_functional.filter, **config.filter_conf
                )
            datapipe = Processor(
                datapipe,
                audio_functional.resample,
                config.resample_rate,
            )
            if config.speech_aug and config.pre_sp_factors:
                datapipe = Processor(
                    datapipe,
                    audio_functional.PreSpeedPerturb(
                        sample_rate=config.resample_rate, factors=config.pre_sp_factors
                    ),
                )

        if config.rand_chunksize is not None:
            chunk_len = int(
                config.rand_chunksize * config.frame_shift * config.resample_rate
                if config.data_type in ("raw", "shard")
                else config.rand_chunksize
            )

            datapipe = Processor(
                datapipe,
                audio_functional.random_chunk,
                chunk_len,
                data_type=config.data_type,
                train_mode=getattr(config, "train_mode", True),
            )
        if config.data_type in ("raw", "shard") and config.speech_aug:
            speech_aug = audio_functional.SpeechAugPipline(**config.speech_aug_config)
            logger.info(f"Got speech aug: {speech_aug} ", ranks=0)
            datapipe = datapipe.apply(
                speech_aug, ignore_lengths=(config.rand_chunksize is not None)
            )

        return datapipe

    @property
    def class_label(self) -> ClassLabel:
        """Property that returns the labels."""
        config: ASVBuilderConfig = self.config
        if config.label_fname:
            label_file = resolve_file(config.label_fname, self.data_dir)
            class_label = ClassLabel.from_file(label_file)
            if (
                config.pre_sp_factors is not None
                and config.data_type != "offline_feat"
                and config.speech_aug
            ):
                class_label = ClassLabel(
                    names=affix_labels(class_label.names, config.pre_sp_factors)
                )
            return class_label
        else:
            return None

    @property
    def feature_extractor(self):
        batch_exc = BatchExtractor(
            self.config.extractor_param, data_type=self.config.data_type
        )
        return batch_exc.extractor

    @property
    def inputs_dim(self) -> int:
        """Property that returns the output channels (feat dim)."""
        return self.feature_extractor.feature_size


def affix_labels(names: List[str], factors: Tuple[float, ...]) -> List[str]:
    names_copy = names.copy()

    affix_names = list(
        audio_functional._affix_speaker(name, factor, prefix="sp")
        for factor in factors
        if factor != 1
        for name in names
    )
    names_copy.extend(affix_names)

    return names_copy


class BatchExtractor:
    def __init__(self, extractor_conf: Dict, data_type: str = "raw") -> None:
        self.data_type = data_type
        if data_type == "offline_feat":
            self.extractor = OfflineFeatureExtractor.from_dict(extractor_conf)
        else:
            self.extractor = KaldiFeatureExtractor.from_dict(extractor_conf)

    def __call__(self, batch: List[Dict]):
        assert isinstance(batch, (list, tuple)) and isinstance(
            batch[0], collections.abc.Mapping
        )
        try:
            batch = {key: [sample[key] for sample in batch] for key in batch[0].keys()}
        except Exception as e:
            raise ValueError(
                f"Failed to convert batch from a list of dicts to a dict contains lists.\n{e}"
            )
        has_label = False
        if "label" in batch and batch["label"] is not None:
            labels = batch.pop("label")
            first = (
                labels[0].item() if isinstance(labels[0], torch.Tensor) else labels[0]
            )
            if isinstance(first, int):
                labels = torch.tensor(labels, dtype=torch.long)
            else:
                labels = torch.tensor(labels)
            has_label = True
        if self.data_type == "offline_feat":
            inputs = self.extractor(batch[OFFLINE_FEAT_COLUMN])
        else:
            sampling_rate = batch[SAMPLE_RATE_COLUMN][0]
            assert all(sr == sampling_rate for sr in batch[SAMPLE_RATE_COLUMN])
            inputs = self.extractor(batch[AUDIO_COLUMN], sampling_rate=sampling_rate)

        if has_label:
            inputs["labels"] = labels
        return inputs
