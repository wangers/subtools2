import functools
from pathlib import Path
import collections
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, Dict, Literal, Optional, Tuple, Union, cast, List

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
    OfflineFeatureExtractor,
    KaldiFeatureExtractor,
)
from egrecho.data.iterable import Processor, processors
from egrecho.utils.data_utils import ClassLabel, Split
from egrecho.utils.io import resolve_file
from egrecho.utils.common import alt_none, ObjectDict, dict_union, field_dict
from egrecho.utils.patch import default_value as dc_default_value

if TYPE_CHECKING:
    from torch.utils.data import IterableDataset


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
            Passing to sampler. Whether apply sharding according to the ddp env. Defaults to False.
        shuffle (bool):
            Passing to sampler. Defaults to False.
        sampler_kwargs (dict, optional):
            other kwargs passed to sampler for dataset.
        exbuild_src_kws (Dict):
            Any extend args to build src datapipe.
        voyage (ASVTemplate):
            This means there will be a validation of colunms' name when building datapipe, e.g.,
            ``voyage = ASVTemplate(task='automatic-speaker-verification',
            start_sketch=('id', 'audio', 'sample_rate', 'speaker'),
            end_sketch=('features', 'lens', 'labels')``, in the start and end of processing,
            we can call the function: ``egrecho.core.voyages.build_table`` to make sure the
            desired columns are existed, and whether to drop extras columns.
        label_fname (str):
            class label mapping file. rel to `data_dir` (speaker.yaml) or abs path
            if given abs pattern (/export/speaker.yaml)
        resample_rate (int):
            If given, applys resamplings. Defaults to None.
        aug_sr (int):
            Set for some audio augments moudules which need sample_rate.
            if `resample_rate` is given, auto change to `resample_rate`. Defaults to 16000.

        pre_sp_factors (tuple[float]):
            If not provided (None), will auto initiate with `(0.9, 1.0, 1.1)` for training split only.
            Defaults to None.

    """

    data_type: Literal["shard", "raw", "offline_feat"] = field(
        default="shard", metadata={"cmd": True}
    )
    lazyload_source: bool = field(default=False, metadata={"cmd": True})
    shuffle: bool = False
    partition: bool = False
    voyage: ASVTemplate = field(default_factory=ASVTemplate)
    sampler_kwargs: Dict[str, Any] = field(default_factory=dict)
    exbuild_src_kw: Dict[str, Any] = field(default_factory=dict)
    label_fname: str = field(
        default="speaker.yaml", metadata={"cmd": True, "to_dict": False}
    )
    filter_conf: Optional[Dict] = None
    resample_rate: int = 16_000
    pre_sp_factors: Optional[Tuple[float, ...]] = None

    rand_chunksize: Optional[int] = 200
    frame_shift: float = 0.01

    speech_aug: bool = True
    speech_aug_config: Optional[Dict] = field(
        default_factory=lambda: dict(batch_size=2, db_dir="/data2/ldx/speech_aug")
    )
    shard_shuffle_size: int = 2500

    batch_size: int = 12

    extractor_param: dict = field(default_factory=lambda: {})

    def __post_init__(self):
        super().__post_init__()
        self.extractor_param = dict_union(get_extractor_param(), self.extractor_param)
        if self.data_type == "offline_feat":
            self.extractor_param.pop("feat_conf", None)

        self.voyage = ASVTemplate(offline_feat=(self.data_type == "offline_feat"))
        self.training = self.split_name == Split.TRAIN
        if self.training:
            self.pre_sp_factors = alt_none(self.pre_sp_factors, (0.9, 1.0, 1.1))

    @property
    def build_src_kwargs(self):
        sampler_kwargs = dict(
            shuffle=self.shuffle, partition=self.partition, **self.sampler_kwargs
        )
        return dict(
            data_type=self.data_type,
            lazyload_source=self.lazyload_source,
            sampler_kwargs=sampler_kwargs,
            **self.exbuild_src_kw,
        )


class ASVPipeBuilder(DataBuilder):
    CONFIG_CLS = ASVBuilderConfig

    def __init__(self, config, data_dir: Optional[str] = None):
        super().__init__(config, data_dir)

    def build_single_dataset(
        self,
        split: Optional[Union[str, Split]] = Split.TRAIN,
    ) -> "IterableDataset":
        """Builds a single datapipe through split config."""
        split_files = self.data_files[split]
        config = cast(ASVBuilderConfig, self.configs[split])

        datapipe = ASVSamples.build_source_dataset(
            split_files, **config.build_src_kwargs
        )

        # format dict sample
        build_table_fn = functools.partial(
            build_table, sketch=config.voyage.start_sketch
        )
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
            label_file = resolve_file(config.label_fname, self.data_dir)
            class_label = ClassLabel.from_file(label_file)

            if config.pre_sp_factors is not None and config.data_type != "offline_feat":
                class_label = ClassLabel(
                    names=affix_labels(class_label.names, config.pre_sp_factors)
                )
            datapipe = Processor(
                datapipe,
                processors.maps,
                class_label.encode_label,
                input_col=SPEAKER_COLUMN,
                output_col="label",
            )
        datapipe = datapipe.apply(processors.batch, config.batch_size)
        batch_exc = BatchExtractor(config.extractor_param, data_type=config.data_type)
        datapipe = Processor(
            datapipe,
            processors.maps,
            batch_exc,
        )
        return datapipe

    @classmethod
    def _ele_processor(cls, datapipe: Processor, config: ASVBuilderConfig):
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
            if config.pre_sp_factors:
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
            )
        if config.data_type in ("raw", "shard") and config.speech_aug:
            speech_aug = audio_functional.SpeechAugPipline(**config.speech_aug_config)
            datapipe = datapipe.apply(
                speech_aug, ignore_lengths=(config.rand_chunksize is not None)
            )

        return datapipe


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
            dtype = torch.long if isinstance(first, int) else torch.float
            labels = torch.tensor(labels, dtype=dtype)
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

    @property
    def monitor(self):
        """Quantity with mode to monitor.

        Useful for model checkpointing or early stopping, e.g., tells
        out `val_loss`, `min`. This should be implemented
        by your detail subclasses. Defaults returns `None, None`.


        Returns:
            monitor : str
                Name of quantity to monitor.
            mode : {'min', 'max}
                Minimize

        See also:

            - lightning.pytorch.callbacks.ModelCheckpoint
            - lightning.pytorch..callbacks.EarlyStopping
        """
        monitor, mode = None, None
        return monitor, mode
