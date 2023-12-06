# -*- coding:utf-8 -*-
# Copyright xmuspeech (Author: Leo 2023-9)

import shutil
from dataclasses import dataclass
from functools import partial
from pathlib import Path
from typing import Literal, Optional, Tuple, Union

from egrecho.core.parser import BaseCommand, CommonParser, Namespace
from egrecho.data.datasets.audio import KaldiDataset, load_kaldiset_to_asv
from egrecho.utils.common import is_in_range
from egrecho.utils.constants import DATASET_META_FILENAME
from egrecho.utils.data_utils import (
    ClassLabel,
    Split,
    SplitInfo,
    SplitInfoDict,
    try_length,
)
from egrecho.utils.logging import get_logger
from egrecho_cli.register import register_command

logger = get_logger()


@dataclass
class KaldiSetArgs:
    r"""
    Arguments for loading & splitting kaldi-style dir.
    More detail referring to `egrecho.data.datasets.audio.kaldi_dataset.KaldiDataset`.

    Args:
        must_exists:
            Files must exist, if None, auto change to `wav_scp` in `KaldiDataset`.
        maybe_exists:
            Files maybe exist, if None, auto change to
            `['utt2spk', 'reco2dur', 'reco2sr', 'segments', 'utt2num_frames']` in `KaldiDataset`.
        more:
            If False, ignore files not in must_exitsts and maybe_exitsts, else load all supported files exist.
        nj:
            num_jobs when computing duration.
        min_dur:
            utts of dur (seconds) shorter than this will be dropped. if None, skip.
        max_dur:
            utts of dur (seconds) longer than this will be dropped. if None, skip.
        cls_attr:
            which attr treated as supervision key, support ('utt2spk', 'utt2lang', 'utt2gender').
            if set `''` skip supervision-related preprocess (e.g., filter class, generate dict, etc.).
        min_cls2utt:
            drop segs (utts with supervision keys) which belong to one class is too few.
        label_fname:
            class label file name for dumping class label.
        filt_nj:
            nj of filter fn.
        filt_noseg:
            filt utts without `seg_id` for supervision training, must set false for test set as
            test set have no label will result an empty dataset.
        split_valid:
            Whether split to train & validation, set false for test set.
        split_requirement:
            per_class: Each class random sample `split_num`.
            total_class: Select `split_num` segs (i.e., the utts have supervised info)
                in total to contain class label as more as possible.
            utt: Sample `split_num` utts instead of segs.

        split_num:
            The number to be split, referring to `split_requirement`, for its true meaning. If 0, skip split dataset.
        split_attr:
            Split dataset via attr. If `split_requirement` is class-related ('per_class', 'total_class'),
            must set this parameter to a classfication attrs, and the requirement parameter is
            applied to this attr. Support (utt2spk, utt2lang, utt2gender).

    """

    must_exists: Optional[Tuple[str, ...]] = ("wav_scp",)
    maybe_exists: Optional[Tuple[str, ...]] = (
        "utt2spk",
        "reco2dur",
        "reco2sr",
        "segments",
        "utt2num_frames",
    )
    more: bool = False
    nj: int = 1
    min_dur: Optional[float] = None
    max_dur: Optional[float] = None
    cls_attr: str = "utt2spk"
    min_cls2utt: int = 8
    label_fname: str = "speaker"
    filt_nj: int = 1
    filt_noseg: bool = False
    split_valid: bool = True
    split_requirement: str = "total_class"
    split_num: int = 4096
    split_attr: str = "utt2spk"

    @property
    def load_kwargs(self):
        return dict(
            must_exists=self.must_exists,
            maybe_exists=self.maybe_exists,
            more=self.more,
            nj=self.nj,
        )

    @property
    def filter_dur_kwargs(self):
        min_dur = self.min_dur
        max_dur = self.max_dur
        if min_dur is None and max_dur is None:
            return {}
        fn = partial(is_in_range, min_val=min_dur, max_val=max_dur)
        return dict(apply_attr="reco2dur", filter_fn=fn, nj=self.filt_nj)

    @property
    def split_kwargs(self):
        return dict(
            requirement=self.split_requirement,
            num_ids=self.split_num,
            apply_attr=self.split_attr,
        )


@dataclass
class DumpArgs:
    """
    Args for preparing egrecho manifest.

    Args:
        data_type:
            Prepare raw or shard manifest.
        raw_chunksize:
            Number of raw samples per split, maybe infected by `enven_chunksize=True`.
        even_chunksize:
            If True, the max num differ between splits is 1.
        split_prefix:
            prefix for saved manifest name.
        shard_dir:
            Placeholder of saved shards (i.e., tarfiles).
        shardsize:
            Number of samples per shard, maybe infected by `enven_chunksize=True`.
        clear_shards:
            If true, force remove shard_dir dir if exists.
        gzip_out:
            if true, output manifest jsonl.gz
        nj:
            num_jobs for preparing egs.
    """

    data_type: Literal["raw", "shard"] = "raw"
    raw_chunksize: int = None
    even_chunksize: bool = False
    split_prefix: str = ""

    shard_dir: Union[Path, str] = None
    shardsize: int = 1000
    clear_shards: bool = False
    gzip_out: bool = False
    nj: int = 1

    def __post_init__(self):
        if self.data_type == "shard":
            self.valid_shard_dir()

    def valid_shard_dir(self):
        if self.shard_dir is None:
            raise ValueError("Please set a dir to save shards.")
        shard_dir = Path(self.shard_dir)

        if shard_dir.is_dir() and self.clear_shards:
            logger.warning(f"Force clearing ({shard_dir}) ...")
            shutil.rmtree(shard_dir)
        if shard_dir.is_dir():
            raise ValueError(
                f"It seems exists dir {self.shard_dir} to save shards, check it or set `-fcsd true`."
            )

    @property
    def chunk_kwargs(self):
        chunk_size = self.raw_chunksize if self.data_type == "raw" else self.shardsize
        return dict(chunk_size=chunk_size, even_chunksize=self.even_chunksize)


DESCRIPTION = "Prepare egrecho format samples."


@register_command(name="kd2egs", aliases=["kaldi_set2samples"], help=DESCRIPTION)
class KaldiSet2Samples(BaseCommand):
    """
    Prepare egrecho format samples.

    The whole pipe consists of the following steps:
        - step 1: load a kaldi-style dir and get a `KaldiDataset`.
        - step 2 (optional): filter valid examples (e.g., filt wavs of during in valid range).
        - step 3: split into train & valid if needed.
        - step 4: a `ASVSamples` is loaded from `KaldiDataset`, then export to egrecho format file.
            - Case 1: raw samples manifest in `out_dir`.
            - Case 2: shards (i.e., tar files) in a specify dir and a manifest mapping to these shards in `out_dir`.
    """

    @classmethod
    def get_dummy_parser(cls) -> CommonParser:
        return CommonParser(description=DESCRIPTION)

    @classmethod
    def setup_parser(cls, parser: CommonParser):
        parser.add_cfg_flag()
        parser.add_argument("kaldidir", type=str, help="Kaldi-style dir location.")
        parser.add_argument("outdir", type=str, help="output manifest dir location.")
        parser.add_argument(
            "--force-clear-manifest-dir",
            "-fcmd",
            dest="fcmd",
            default=False,
            type=bool,
            help="If true, force remove whole manifest dir if exists, otherwise just overwrite manifests.",
        )

        parser.add_argument(
            "--nj",
            default=16,
            type=int,
            help="global nj.",
        )
        parser.add_class_arguments(KaldiSetArgs, "kd_args")
        parser.add_class_arguments(DumpArgs, "dumps")

        return parser

    @staticmethod
    def run_from_args(args: Namespace, parser: CommonParser):
        args_init = parser.instantiate_classes(args)

        exc = KaldiSet2Samples(**args_init)
        exc.run()

    def __init__(
        self,
        kaldidir: Path,
        outdir: Path,
        kd_args: KaldiSetArgs,
        dumps: DumpArgs,
        fcmd=False,
        nj=1,
        **kwargs,
    ):
        out_dir = Path(outdir)

        if out_dir.is_dir() and fcmd:
            logger.warning(f"Force clearing ({out_dir}) ...")
            shutil.rmtree(out_dir)
        out_dir.mkdir(parents=True, exist_ok=True)
        self.kd_args = kd_args
        self.dumps = dumps
        self.kaldi_dir = kaldidir
        self.out_dir = out_dir
        self.nj = nj

    def run(self):
        kd_args = self.kd_args
        out_dir = self.out_dir
        kd_load_kwargs = kd_args.load_kwargs
        kd_load_kwargs["nj"] = (
            kd_load_kwargs["nj"] if kd_load_kwargs["nj"] > 1 else self.nj
        )
        kd_sets = KaldiDataset(self.kaldi_dir, **kd_load_kwargs)
        if kd_args.filter_dur_kwargs:
            logger.info(f"Ready to get valid duration utts:{kd_args.filter_dur_kwargs}")
            kd_sets = kd_sets.filter(**kd_args.filter_dur_kwargs)

        cls_attr = kd_args.cls_attr

        saved_labelfile = None  # a flag of flabel
        if cls_attr:
            cls_mapping = kd_sets.generate_label2segids(cls_attr)
            min_cls2utt = kd_args.min_cls2utt
            valid_cls = {
                cls_name
                for cls_name in cls_mapping
                if len(cls_mapping[cls_name]) >= min_cls2utt
            }

            logger.info(f"Filter labels with segs fewer than {min_cls2utt}.")
            kd_sets = kd_sets.filter(
                apply_attr=cls_attr,
                filter_fn=lambda x: x in valid_cls,
                nj=kd_args.filt_nj,
            )

            valid_cls_lst = list(valid_cls)
            valid_cls_lst.sort()
            class_labels = ClassLabel(names=valid_cls_lst)
            saved_labelfile = out_dir / Path(kd_args.label_fname).with_suffix(".yaml")
            class_labels.to_yaml(
                saved_labelfile
            )  # ready to copy to shard dir if needed.

        if kd_args.filt_noseg:
            kd_sets = kd_sets.filter_noseg_utts()
        valid_sets = None

        if kd_args.split_valid:
            kd_sets, valid_sets = kd_sets.split(**kd_args.split_kwargs)
            infos = {
                Split.TRAIN: SplitInfo(meta=kd_sets.info),
                Split.VALIDATION: SplitInfo(meta=valid_sets.info),
            }
        else:
            infos = {Split.ALL: SplitInfo(meta=kd_sets.info)}
        dumps = self.dumps
        split_prefix = dumps.split_prefix

        # load ks to samples.
        dew_samples = load_kaldiset_to_asv(kd_sets)
        if valid_sets:
            valid_dew_samples = load_kaldiset_to_asv(valid_sets)
            split_prefix = f"{split_prefix}-" if split_prefix else ""
            split_kd_name = f"{split_prefix}train"
            split_valid_name = f"{split_prefix}validation"
            infos[Split.VALIDATION].num_examples = try_length(valid_dew_samples) or 0
            infos[Split.TRAIN].num_examples = try_length(dew_samples) or 0
        else:
            split_kd_name = split_prefix
            infos[Split.ALL].num_examples = try_length(dew_samples) or 0
        infos = SplitInfoDict.from_dict(infos)

        chunk_kwargs = dumps.chunk_kwargs
        if dumps.data_type == "shard":
            shard_dir = Path(dumps.shard_dir)
            nj = dumps.nj
            nj = nj if nj > 1 else self.nj
            if valid_sets:
                dew_samples.export_shard(
                    shard_dir / "train",
                    shard_manifest_dir=out_dir,
                    split_name=split_kd_name,
                    nj=nj,
                    **chunk_kwargs,
                )
                valid_dew_samples.export_shard(
                    shard_dir / "validation",
                    shard_manifest_dir=out_dir,
                    split_name=split_valid_name,
                    nj=nj,
                    **chunk_kwargs,
                )
            else:
                dew_samples.export_shard(
                    shard_dir,
                    shard_manifest_dir=out_dir,
                    split_name=split_kd_name,
                    nj=nj,
                    **chunk_kwargs,
                )
            infos.to_file(shard_dir / DATASET_META_FILENAME)
            if saved_labelfile:
                shutil.copy(saved_labelfile, shard_dir)
        else:
            dew_samples.split_to_files(
                out_dir, split_name=split_kd_name, **chunk_kwargs
            )
            if valid_sets:
                valid_dew_samples.split_to_files(
                    out_dir, split_name=split_valid_name, **chunk_kwargs
                )
        infos.to_file(out_dir / DATASET_META_FILENAME)
        logger.info(f"Prepare egrecho format samples in ({out_dir}) done")


if __name__ == "__main__":
    pass
