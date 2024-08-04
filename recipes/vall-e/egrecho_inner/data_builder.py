# -*- coding:utf-8 -*-
# (Author: Leo 202406)

from dataclasses import dataclass, field
from functools import lru_cache
from pathlib import Path
from typing import Any, Dict, List, Optional

import torch
from dataset_valle import ValleDataset
from lhotse import CutSet, load_manifest_lazy
from lhotse.dataset import CutConcatenate, DynamicBucketingSampler, SimpleCutSampler
from tokenizer_valle import OfflineCodesExtractor, ValleTokenizer, ValleTokenizerConfig
from torch.utils.data import DataLoader

from egrecho.core.data_builder import DataBuilder, DataBuilderConfig
from egrecho.data.processors.renamer import _rename_columns
from egrecho.utils.common import dict_union
from egrecho.utils.logging import get_logger

logger = get_logger()


DEFALUT_TOKENIZER_KW = dict(language="en-us", backend="espeak")


def filter_short_and_long_utterances(
    cuts: CutSet, min_duration: float, max_duration: float
) -> CutSet:
    def remove_short_and_long_utt(c):
        # Keep only utterances with duration between 0.6 second and 20 seconds
        if c.duration < min_duration or c.duration > max_duration:
            # logging.warning(
            #     f"Exclude cut with ID {c.id} from training. Duration: {c.duration}"
            # )
            return False
        return True

    cuts = cuts.filter(remove_short_and_long_utt)

    return cuts


@dataclass
class LhotseBuilderConfig(DataBuilderConfig):
    """TTS data related options

    Args:
        max_duration:
            Maximum pooled recordings duration (seconds) in a single batch.
            You can reduce it if it causes CUDA OOM.
        min_duration:
            Min single utt
        max_duration:
            Max sigle utt
        bucketing_sampler:
            When enabled, the batches will come from buckets of similar duration (saves padding frames).
        num_buckets:
            The number of buckets for the DynamicBucketingSampler, (you might want to increase it for larger datasets)
        concatenate_cuts:
            When enabled, utterances (cuts) will be concatenated to minimize the amount of padding.
        duration_factor:
            Determines the maximum duration of a concatenated cut elative to the duration of the longest cut in a batch
        gap:
            The amount of padding (in seconds) inserted between concatenated cuts.
        shuffle:
            Shuffle train data.
        buffer_size:
            How many cuts (or cut pairs, triplets) we hold at any time across all of the buckets.
            Increasing ``max_duration`` (batch_size) or ``num_buckets`` might require increasing this number.
            It will result in larger memory usage.
        shuffle_buffer_size:
            How many cuts (or cut pairs, triplets) are being held in memory, a buffer used for streaming shuffling.
            Larger number means better randomness at the cost memory usage.
        drop_last:
            Whether to drop last batch. Used by sampler.
        return_cuts:
            When enabled, each batch will have the ield: batch['supervisions']['cut'] with the cuts.
        num_workers:
            The number of training dataloader workers.
        tokenizer_config:
            Tokenizer config dataclass.
    """

    filter_min_dur: float = 0.5
    filter_max_dur: float = 14.0
    max_cuts: Optional[int] = None
    max_duration: float = 40
    bucketing_sampler: bool = True
    num_buckets: int = 10
    concatenate_cuts: bool = False
    duration_factor: float = 1.0
    gap: float = 0.1
    shuffle: bool = True
    buffer_size: int = 40000
    shuffle_buffer_size: int = 100000
    drop_last: bool = True
    return_cuts: bool = True
    num_workers: int = 8
    tokenizer_config: dict = field(default_factory=lambda: DEFALUT_TOKENIZER_KW)

    def __post_init__(self):
        super().__post_init__()
        self.tokenizer_config = dict_union(
            {"extradir": self.data_dir}, self.tokenizer_config
        )
        self.tokenizer_config = ValleTokenizerConfig.from_dict(self.tokenizer_config)
        self.collator = self.batch_collator()

    def batch_collator(self):
        extractor = OfflineCodesExtractor()
        tokenizer = ValleTokenizer(self.tokenizer_config)
        return BatchCollator(extractor, tokenizer)


class LhotseBuilder(DataBuilder):

    CONFIG_CLS = LhotseBuilderConfig

    def __init__(self, config: LhotseBuilderConfig):
        super().__init__(config)

    # alias
    @property
    def args(self) -> LhotseBuilderConfig:
        return self.config

    @property
    def feature_extractor(self):
        return self.args.collator.feature_extractor

    @property
    def tokenizer(self):
        return self.args.collator.tokenizer

    @property
    def vocab_size(self):
        return self.tokenizer.vocab_size

    @property
    def pad_text_token_id(self):
        return self.tokenizer.pad_id

    def train_dataloaders(
        self,
        cuts_train: CutSet,
        sampler_state_dict: Optional[Dict[str, Any]] = None,
    ) -> DataLoader:
        """
        Args:
          cuts_train:
            CutSet for training.
          sampler_state_dict:
            The state dict for the training sampler.
        """
        transforms = []

        if self.args.concatenate_cuts:
            logger.info(
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

        train = ValleDataset(
            cut_transforms=transforms,
            return_text=True,
            return_tokens=True,
            return_cuts=self.args.return_cuts,
        )
        cuts_train = filter_short_and_long_utterances(
            cuts_train, self.args.filter_min_dur, self.args.filter_max_dur
        )

        if self.args.bucketing_sampler:
            logger.info("Using DynamicBucketingSampler", ranks=0)
            train_sampler = DynamicBucketingSampler(
                cuts_train,
                max_duration=self.args.max_duration,
                shuffle=self.args.shuffle,
                buffer_size=self.args.buffer_size,
                shuffle_buffer_size=self.args.shuffle_buffer_size,
                quadratic_duration=10,
                num_cuts_for_bins_estimate=10000,
                drop_last=self.args.drop_last,
                max_cuts=self.args.max_cuts,
            )
        else:
            logger.info(
                "Using SimpleCutSampler and sort by duraton(ascending=True).", ranks=0
            )
            cuts_train = cuts_train.to_eager().sort_by_duration(ascending=True)
            train_sampler = SimpleCutSampler(
                cuts_train,
                max_duration=self.args.max_duration,
                shuffle=self.args.shuffle,
                max_cuts=self.args.max_cuts,
            )
        logger.info("About to create train dataloader")

        if sampler_state_dict is not None:
            logger.info("Loading sampler state dict")
            train_sampler.load_state_dict(sampler_state_dict)

        train_dl = DataLoader(
            train,
            sampler=train_sampler,
            batch_size=None,
            num_workers=self.args.num_workers,
            persistent_workers=False,
            collate_fn=self.args.collator,
        )

        return train_dl

    def valid_dataloaders(self, cuts_valid: CutSet) -> DataLoader:
        validate = ValleDataset(
            return_text=True,
            return_tokens=True,
            return_cuts=self.args.return_cuts,
        )
        cuts_valid = filter_short_and_long_utterances(
            cuts_valid, self.args.filter_min_dur, self.args.filter_max_dur
        )
        valid_sampler = DynamicBucketingSampler(
            cuts_valid,
            max_duration=self.args.max_duration,
            shuffle=False,
            drop_last=True,
            max_cuts=self.args.max_cuts,
        )
        logger.info("About to create dev dataloader", ranks=0)
        valid_dl = DataLoader(
            validate,
            sampler=valid_sampler,
            batch_size=None,
            num_workers=4,
            persistent_workers=False,
            collate_fn=self.args.collator,
        )

        return valid_dl

    def test_dataloaders(self, cuts: CutSet) -> DataLoader:
        test = ValleDataset(
            return_text=True,
            return_tokens=True,
            return_cuts=self.args.return_cuts,
        )
        sampler = DynamicBucketingSampler(
            cuts,
            max_duration=self.args.max_duration,
            shuffle=False,
            drop_last=False,
            max_cuts=self.args.max_cuts,
        )
        logger.debug("About to create test dataloader", ranks=0)

        test_dl = DataLoader(
            test,
            batch_size=None,
            sampler=sampler,
            num_workers=self.args.num_workers,
        )
        return test_dl

    @classmethod
    def get_multi_cuts(cls, files: List[str], for_train: bool = False):
        if not bool(files):
            return None

        if isinstance(files, str):
            files = [files]
        cutsets = []
        weights = [] if for_train else None
        for subfile in files:
            logger.info(f"About to get cuts {Path(subfile).name}", ranks=0)
            cutset = load_manifest_lazy(subfile)
            cutsets.append(cutset)
            if weights is not None:
                weights.append(len(cutset))
        if len(cutsets) > 1:
            return CutSet.mux(*cutsets, weights=weights)
        return cutsets[0]

    @lru_cache()
    def train_cuts(self) -> CutSet:
        logger.info(f"About to get train cuts from {self.data_dir}", ranks=0)
        return self.get_multi_cuts(self.train_data_files, for_train=True)

    @lru_cache()
    def dev_cuts(self) -> CutSet:
        logger.info(f"About to get val cuts from {self.data_dir}", ranks=0)
        return self.get_multi_cuts(self.val_data_files)

    @lru_cache()
    def test_cuts(self) -> CutSet:
        logger.info(f"About to get test cuts from {self.data_dir}", ranks=0)
        return self.get_multi_cuts(self.test_data_files)


@dataclass
class BatchCollator:
    feature_extractor: OfflineCodesExtractor
    tokenizer: ValleTokenizer

    def __call__(self, batch) -> Dict[str, torch.Tensor]:

        input_ids = batch["codes"]
        # input_ids, attention_mask
        batch_out = _rename_columns(
            self.feature_extractor(input_ids), {"input_features": "input_ids"}
        )

        # get the tokenized phonemes.
        phn_outs = self.tokenizer(batch["tokens"], return_tensors="pt")

        batch_out["text_input_ids"] = phn_outs["input_ids"]
        batch_out["text_attention_mask"] = phn_outs["attention_mask"]

        return batch_out


if __name__ == "__main__":
    d = "exp/egs/libritts"
    cfg = LhotseBuilderConfig(data_dir=d, file_patterns={"train": "cuts_train*"})
    # cfg.to_cfg_file("tst.yml")
    db = LhotseBuilder(cfg)
    dl = db.train_dataloaders(db.train_cuts())
    for i, batch in enumerate(dl):
        print(batch)
        print(batch["input_ids"])
        print(batch["text_attention_mask"], batch["text_attention_mask"].shape)
        break
