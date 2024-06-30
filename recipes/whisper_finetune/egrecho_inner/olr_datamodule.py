# Copyright      2021  Piotr Żelasko
# Copyright      2022  Xiaomi Corporation     (Author: Mingshuang Luo)
#
# See ../../../../LICENSE for clarification regarding multiple authors
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

# (Author: Leo 2024-06-04)

import inspect
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Any, Dict, List, Literal, Optional

import torch
from lhotse import (
    CutSet,
    WhisperFbank,
    WhisperFbankConfig,
    load_manifest,
    load_manifest_lazy,
)
from lhotse.cut.base import Cut
from lhotse.dataset import (
    CutConcatenate,
    CutMix,
    DynamicBucketingSampler,
    K2SpeechRecognitionDataset,
    PrecomputedFeatures,
    SimpleCutSampler,
    SpecAugment,
)
from lhotse.dataset.input_strategies import OnTheFlyFeatures
from lhotse.utils import fastcopy, fix_random_seed
from torch.utils.data import DataLoader
from tqdm.auto import tqdm

from egrecho.core.parser import ArgumentParser
from egrecho.utils.common import alt_none
from egrecho.utils.logging import _infer_rank, get_logger
from egrecho.utils.mask import make_non_pad_mask
from egrecho.utils.seeder import isolate_rng

logger = get_logger(__name__)


class _SeedWorkers:
    def __init__(self, seed: int):
        self.seed = seed

    def __call__(self, worker_id: int):
        fix_random_seed(self.seed + worker_id)


class OlrAsrDataModule:
    """
    DataModule for k2 ASR experiments.
    It assumes there is always one train and valid dataloader,
    but there can be multiple test dataloaders (e.g. LibriSpeech test-clean
    and test-other).

    It contains all the common data pipeline modules used in ASR
    experiments, e.g.:d
    - dynamic batch size,
    - bucketing samplers,
    - cut concatenation,
    - augmentation,
    - on-the-fly feature extraction

    This class should be derived for specific corpora used in ASR tasks.
    """

    def __init__(self, args: ArgumentParser):
        self.args = args
        self.trainer_collator: Optional[BatchProcessor] = None

    @classmethod
    def add_arguments(cls, parser: ArgumentParser):
        group = parser.add_argument_group(
            title="ASR data related options",
            description="These options are used for the preparation of "
            "PyTorch DataLoaders from Lhotse CutSet's -- they control the "
            "effective batch sizes, sampling strategies, applied data "
            "augmentations, etc.",
        )
        group.add_argument(
            "--manifest_dir",
            type=Path,
            default=Path("data/fbank"),
            help="Path to directory with train/valid/test cuts.",
        )
        group.add_argument(
            "--prefix_name",
            type=str,
            default="egs",
            help="Prefix dataset name.",
        )

        group.add_argument(
            "--suffix",
            type=str,
            default="jsonl.gz",
            help="Control manifest format.",
        )
        group.add_argument(
            "--max_cuts",
            type=int,
            default=8,
            help="batch size in lhotse.",
        )
        group.add_argument(
            "--bucketing_sampler",
            type=bool,
            default=True,
            help="When enabled, the batches will come from buckets of "
            "similar duration (saves padding frames).",
        )
        group.add_argument(
            "--num_buckets",
            type=int,
            default=30,
            help="The number of buckets for the DynamicBucketingSampler"
            "(you might want to increase it for larger datasets).",
        )
        group.add_argument(
            "--concatenate_cuts",
            type=bool,
            default=False,
            help="When enabled, utterances (cuts) will be concatenated "
            "to minimize the amount of padding.",
        )
        group.add_argument(
            "--duration_factor",
            type=float,
            default=1.0,
            help="Determines the maximum duration of a concatenated cut "
            "relative to the duration of the longest cut in a batch.",
        )
        group.add_argument(
            "--gap",
            type=float,
            default=1.0,
            help="The amount of padding (in seconds) inserted between "
            "concatenated cuts. This padding is filled with noise when "
            "noise augmentation is used.",
        )
        group.add_argument(
            "--on_the_fly_feats",
            type=bool,
            default=False,
            help="When enabled, use on-the-fly cut mixing and feature "
            "extraction. Will drop existing precomputed feature manifests "
            "if available.",
        )
        group.add_argument(
            "--shuffle",
            type=bool,
            default=True,
            help="When enabled (=default), the examples will be "
            "shuffled for each epoch.",
        )

        # group.add_argument(
        #     "--return_cuts",
        #     type=bool,
        #     default=True,
        #     help="When enabled, each batch will have the "
        #     "field: batch['supervisions']['cut'] with the cuts that "
        #     "were used to construct it.",
        # )

        group.add_argument(
            "--num_workers",
            type=int,
            default=2,
            help="The number of training dataloader workers that "
            "collect the batches.",
        )

        group.add_argument(
            "--enable_spec_aug",
            type=bool,
            default=True,
            help="When enabled, use SpecAugment for training dataset.",
        )

        group.add_argument(
            "--spec_aug_time_warp_factor",
            type=int,
            default=80,
            help="Used only when --enable-spec-aug is True. "
            "It specifies the factor for time warping in SpecAugment. "
            "Larger values mean more warping. "
            "A value less than 1 means to disable time warp.",
        )

        group.add_argument(
            "--enable_musan",
            type=bool,
            default=True,
            help="When enabled, select noise from MUSAN and mix it"
            "with training dataset. ",
        )
        # group.add_argument("--timestamps", type=bool, default=False, help="训练时是否使用时间戳数据(未实现, 只能是False)")

    def infer_sampler_len(self, dataloader=None):
        dataloader = alt_none(dataloader, self.train_dataloaders())
        sampler = dataloader.sampler
        cnt = 0
        rank = _infer_rank() or 0
        with isolate_rng(), tqdm(
            disable=(rank != 0),
            unit=" steps",
            desc=f"Roughly inferring {dataloader.sampler.__class__.__name__} size, rank={rank}.",
        ) as pbar:
            for _ in sampler:
                cnt += 1
                pbar.update()

        return cnt

    def attach_trainer_collator(self, collator: "BatchProcessor"):
        assert isinstance(collator, BatchProcessor), collator
        self.trainer_collator = collator

    def train_dataloaders(
        self,
        cuts_train: Optional[CutSet] = None,
        sampler_state_dict: Optional[Dict[str, Any]] = None,
    ) -> DataLoader:
        """
        Args:
          cuts_train:
            CutSet for training.
          sampler_state_dict:
            The state dict for the training sampler.
        """
        cuts_train = alt_none(cuts_train, self.train_cuts().filter(remove_long_utt))
        logger.info_once("About to get Musan cuts", ranks=0)
        cuts_musan = load_manifest(self.args.manifest_dir / "musan_cuts.jsonl.gz")

        transforms = [CutMergeSuperversions()]
        if self.args.enable_musan:
            logger.info_once("Enable MUSAN", ranks=0)
            transforms.append(
                CutMix(cuts=cuts_musan, p=0.5, snr=(10, 20), preserve_id=True)
            )
        else:
            logger.info_once("Disable MUSAN", ranks=0)
        if self.args.concatenate_cuts:
            logger.info_once(
                f"Using cut concatenation with duration factor "
                f"{self.args.duration_factor} and gap {self.args.gap}.",
                ranks=0,
            )
            # Cut concatenation should be the first transform in the list,
            # so that if we e.g. mix noise in, it will fill the gaps between
            # different utterances.
            transforms = [
                CutConcatenate(
                    duration_factor=self.args.duration_factor, gap=self.args.gap
                )
            ] + transforms

        input_transforms = []
        if self.args.enable_spec_aug:
            logger.info_once("Enable SpecAugment", ranks=0)
            logger.info_once(
                f"Time warp factor: {self.args.spec_aug_time_warp_factor}", ranks=0
            )
            # Set the value of num_frame_masks according to Lhotse's version.
            # In different Lhotse's versions, the default of num_frame_masks is
            # different.
            num_frame_masks = 10
            num_frame_masks_parameter = inspect.signature(
                SpecAugment.__init__
            ).parameters["num_frame_masks"]
            if num_frame_masks_parameter.default == 1:
                num_frame_masks = 2
            logger.info_once(f"Num frame mask: {num_frame_masks}", ranks=0)
            input_transforms.append(
                SpecAugment(
                    time_warp_factor=self.args.spec_aug_time_warp_factor,
                    num_frame_masks=num_frame_masks,
                    features_mask_size=27,
                    num_feature_masks=2,
                    frames_mask_size=100,
                )
            )
        else:
            logger.info_once("Disable SpecAugment", ranks=0)

        logger.info_once("About to create train dataset", ranks=0)
        train = K2SpeechRecognitionDataset(
            cut_transforms=transforms,
            input_transforms=input_transforms,
            return_cuts=True,
        )

        if self.args.on_the_fly_feats:
            # NOTE: the PerturbSpeed transform should be added only if we
            # remove it from data prep stage.
            # Add on-the-fly speed perturbation; since originally it would
            # have increased epoch size by 3, we will apply prob 2/3 and use
            # 3x more epochs.
            # Speed perturbation probably should come first before
            # concatenation, but in principle the transforms order doesn't have
            # to be strict (e.g. could be randomized)
            # transforms = [PerturbSpeed(factors=[0.9, 1.1], p=2/3)] + transforms   # noqa
            # Drop feats to be on the safe side.
            train = K2SpeechRecognitionDataset(
                cut_transforms=transforms,
                input_strategy=OnTheFlyFeatures(
                    WhisperFbank(WhisperFbankConfig(num_filters=80))
                ),
                input_transforms=input_transforms,
                return_cuts=True,
            )

        if self.args.bucketing_sampler:
            logger.info_once("Using DynamicBucketingSampler.", ranks=0)
            train_sampler = DynamicBucketingSampler(
                cuts_train,
                max_cuts=self.args.max_cuts,
                shuffle=self.args.shuffle,
                num_buckets=self.args.num_buckets,
                buffer_size=self.args.num_buckets * 2000,
                shuffle_buffer_size=self.args.num_buckets * 5000,
                drop_last=True,
            )
        else:
            logger.info_once("Using SimpleCutSampler.", ranks=0)
            train_sampler = SimpleCutSampler(
                cuts_train,
                max_cuts=self.args.max_cuts,
                shuffle=self.args.shuffle,
                drop_last=True,
            )
        logger.info_once("About to create train dataloader", ranks=0)

        if sampler_state_dict is not None:
            logger.info_once("Loading sampler state dict", ranks=0)
            train_sampler.load_state_dict(sampler_state_dict)
        # 'seed' is derived from the current random state, which will have
        # previously been set in the main process.
        seed = torch.randint(0, 100000, ()).item()
        worker_init_fn = _SeedWorkers(seed)
        train_dl = DataLoader(
            train,
            sampler=train_sampler,
            batch_size=None,
            num_workers=self.args.num_workers,
            persistent_workers=False,
            worker_init_fn=worker_init_fn,
            collate_fn=self.trainer_collator,
        )

        return train_dl

    def valid_dataloaders(self, cuts_valid: CutSet) -> DataLoader:
        transforms = []
        if self.args.concatenate_cuts:
            transforms = [
                CutConcatenate(
                    duration_factor=self.args.duration_factor, gap=self.args.gap
                )
            ] + transforms

        logger.info_once("About to create dev dataset", ranks=0)
        if self.args.on_the_fly_feats:
            validate = K2SpeechRecognitionDataset(
                cut_transforms=transforms,
                input_strategy=OnTheFlyFeatures(
                    WhisperFbank(WhisperFbankConfig(num_filters=80))
                ),
                return_cuts=True,
            )
        else:
            validate = K2SpeechRecognitionDataset(
                cut_transforms=transforms,
                return_cuts=True,
            )
        valid_sampler = DynamicBucketingSampler(
            cuts_valid.subset(first=1024),
            max_cuts=self.args.max_cuts,
            shuffle=False,
        )
        logger.info_once("About to create dev dataloader", ranks=0)
        valid_dl = DataLoader(
            validate,
            sampler=valid_sampler,
            batch_size=None,
            num_workers=2,
            persistent_workers=False,
            collate_fn=self.trainer_collator,
        )

        return valid_dl

    def test_dataloaders(self, cuts: CutSet) -> DataLoader:
        logger.info_once("About to create test dataset", ranks=0)
        test = K2SpeechRecognitionDataset(
            input_strategy=(
                OnTheFlyFeatures(WhisperFbank(WhisperFbankConfig(num_filters=80)))
                if self.args.on_the_fly_feats
                else PrecomputedFeatures()
            ),
            return_cuts=True,
        )
        sampler = DynamicBucketingSampler(
            cuts,
            max_cuts=self.args.max_cuts,
            shuffle=False,
        )
        test_dl = DataLoader(
            test,
            batch_size=None,
            sampler=sampler,
            num_workers=self.args.num_workers,
        )
        return test_dl

    @lru_cache()
    def train_cuts(self) -> CutSet:
        logger.info("About to get train cuts")
        cuts_train = load_manifest_lazy(
            self.args.manifest_dir
            / f"{self.args.prefix_name}_cuts_train.{self.args.suffix}"
        )
        return cuts_train

    @lru_cache()
    def valid_cuts(self) -> CutSet:
        logger.info("About to get dev cuts")
        return load_manifest_lazy(
            self.args.manifest_dir
            / f"{self.args.prefix_name}_cuts_val.{self.args.suffix}"
        )

    @lru_cache()
    def test_cuts(self) -> List[CutSet]:
        logger.info("About to get test cuts")
        return load_manifest_lazy(
            self.args.manifest_dir
            / f"{self.args.prefix_name}_cuts_test.{self.args.suffix}"
        )

    @lru_cache()
    def test_cuts_split(self, part) -> List[CutSet]:
        logger.info("About to get test cuts partly")
        return load_manifest_lazy(
            self.args.manifest_dir
            / f"{self.args.prefix_name}_cuts_test_{part}.{self.args.suffix}"
        )


class CutMergeSuperversions:
    """
    A transform on batch of cuts (``CutSet``) that merges the cut's superversions.

    :param timestamps: Whether insert timestamp in text. Not Implemented
    """

    def __init__(
        self,
        timestamps: bool = False,
    ) -> None:
        if timestamps:
            raise NotImplementedError
        self.timestamps = timestamps

    def __call__(self, cuts: CutSet) -> CutSet:
        return cuts.map(merge_sup)


def merge_sup(c: Cut, timestamps: bool = False):
    if timestamps:
        raise NotImplementedError

    m_lang = [s.language for s in c.supervisions if s.language]
    if len(set(lang.lower().strip() for lang in m_lang)) > 1:
        logger.warning(
            f"Cut with ID {c.id} contains supervisions over one languages {m_lang}, which is ilegal for merging, fallback to keep the first sup."
        )
        new_c = fastcopy(c, supervisions=[c.supervisions[0]])
        return list(new_c.trim_to_supervisions())[0]
    merged: Cut = c.merge_supervisions()
    merged.supervisions[0].language = m_lang[0]
    return merged


def remove_long_utt(c: Cut):
    # Keep only utterances with duration in 30 seconds
    #
    if c.duration > 30.0:
        # logger.warning(
        #    f"Exclude cut with ID {c.id} from training. Duration: {c.duration}"
        # )
        return False
    return True


@dataclass
class BatchProcessor:
    processor: Any
    padding_to_max: bool = True
    process_for_train: bool = True

    def __post_init__(self):
        self.whisper_pad_args = (
            {"padding": "max_length", "max_length": 3000} if self.padding_to_max else {}
        )
        self.stride = (
            self.processor.feature_extractor.sampling_rate
            // self.processor.feature_extractor.hop_length
        )

    def __call__(self, batch) -> Dict[str, torch.Tensor]:
        inputs = {"input_features": batch["inputs"]}
        langs = [cut.supervisions[0].language for cut in batch["supervisions"]["cut"]]
        if self.process_for_train:
            batch_out = self.processor.feature_extractor.pad(
                inputs,
                return_attention_mask=not self.process_for_train,
                return_tensors="pt",
                **self.whisper_pad_args,
            )
            texts = []

            # one lang for one text
            for text, lang in zip(batch["supervisions"]["text"], langs):
                self.processor.tokenizer.set_prefix_tokens(language=lang)
                texts.append(
                    self.processor.tokenizer(
                        text=text,
                    )["input_ids"]
                )
            label_features = {"input_ids": texts}
            # pad the labels to max length
            labels_batch = self.processor.tokenizer.pad(
                label_features, return_tensors="pt"
            )

            # replace padding with -100 to ignore loss correctly
            labels = labels_batch["input_ids"].masked_fill(
                labels_batch.attention_mask.ne(1), -100
            )

            # if bos token is appended in previous tokenization step,
            # cut bos token here as it's append later anyways
            if (
                (labels[:, 0] == self.processor.tokenizer.bos_token_id)
                .all()
                .cpu()
                .item()
            ):
                labels = labels[:, 1:]

            batch_out["labels"] = labels
        else:

            feature_lens = batch["supervisions"]["num_frames"]
            attention_mask = make_non_pad_mask(feature_lens).long()
            inputs["attention_mask"] = attention_mask
            if batch["inputs"].shape[1] < 3000 and self.padding_to_max:
                batch_out = self.processor.feature_extractor.pad(
                    inputs,
                    return_attention_mask=True,
                    return_tensors="pt",
                    **self.whisper_pad_args,
                )
            else:
                batch_out = self.processor.feature_extractor.pad(
                    inputs,
                    return_attention_mask=True,
                    return_tensors="pt",
                    truncation=False,
                    padding="longest",
                )

            batch_out["texts"] = batch["supervisions"]["text"]
            batch_out["langs"] = langs
            batch_out["length_in_s"] = [
                num_frames / self.stride for num_frames in feature_lens
            ]

        batch_out["input_features"] = batch_out["input_features"].transpose(1, 2)
        return batch_out

    @staticmethod
    def from_pretrained(
        pretrained_model_name_or_path: str = "openai/whisper-tiny",
        language: str = "Chinese",
        task: Literal["transcribe", "translate"] = "transcribe",
        no_timestamps: bool = True,
        local_files_only: bool = True,
        padding_to_max: bool = True,
        process_for_train: bool = True,
        **kwargs,
    ):
        from transformers import WhisperProcessor

        processor = WhisperProcessor.from_pretrained(
            pretrained_model_name_or_path,
            language=language,
            task=task,
            no_timestamps=no_timestamps,
            local_files_only=local_files_only,
            **kwargs,
        )
        return BatchProcessor(processor, padding_to_max, process_for_train)
