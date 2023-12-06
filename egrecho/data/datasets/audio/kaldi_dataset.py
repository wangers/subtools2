# -*- coding:utf-8 -*-

# Copyright xmuspeech (Author: Snowdar 2020-01-05
#                      reconstruct: Leo 2023-03-10)

import copy
import functools
from collections import OrderedDict, defaultdict, namedtuple
from concurrent.futures import ProcessPoolExecutor
from contextlib import nullcontext
from dataclasses import dataclass
from enum import IntEnum
from itertools import chain
from operator import add
from pathlib import Path
from typing import Any, Callable, Dict, Iterable, List, Optional, Tuple, TypeVar, Union

import numpy as np
import torchaudio
from tqdm.contrib import tqdm

from egrecho.utils.common import DataclassSerialMixin, Timer, alt_none
from egrecho.utils.data_utils import iflatmap_unordered, split_sequence
from egrecho.utils.imports import _KALDI_NATIVE_IO_AVAILABLE
from egrecho.utils.io import (
    DictFileMixin,
    repr_dict,
    save_yaml,
    torchaudio_info_unfixed,
)
from egrecho.utils.logging import get_logger
from egrecho.utils.patch import validate_input_col

logger = get_logger(__name__)


class FileType(IntEnum):
    UTTFIRST = 1
    SEGFIRST = 2
    CUSTOM = 3  # load this type will not support split/subset operation.


class LoadArgs(namedtuple("LoadArgs", ["file_name", "dtype", "vec"])):
    __slots__ = ()

    def __new__(cls, file_name: str, dtype: str = "str", vec: bool = False):
        return super().__new__(cls, file_name, dtype, vec)


class FileConfig(namedtuple("FileConfig", ["file_type", "load_args"])):
    __slots__ = ()

    def __new__(cls, file_type: FileType, load_args: LoadArgs):
        return super().__new__(cls, file_type, load_args)


@dataclass(frozen=True)
class CustomFileConfig:
    load_args: LoadArgs
    file_type: FileType
    dest_load_args: Optional[LoadArgs] = None
    transform_fn: Optional[Callable] = None


KD = TypeVar("KD", bound="KaldiDataset")
T = TypeVar("T")


def transform_spk2gender(src_set: KD, spk2gender: Dict) -> Dict:
    return {utt: spk2gender[spk] for utt, spk in src_set.utt2spk.items()}


@dataclass
class KaldiDatasetInfo(DictFileMixin, DataclassSerialMixin):
    num_utts: Optional[int] = None
    num_segs: Optional[int] = None
    num_spks: Optional[int] = None
    num_spk_segs: Optional[int] = None
    num_lang_segs: Optional[int] = None
    num_gender_segs: Optional[int] = None
    num_text_segs: Optional[int] = None
    feat_dim: Optional[int] = None
    seg_dur: Optional[float] = None
    utt_dur: Optional[float] = None
    num_frames: Optional[int] = None

    def __post_init__(self):
        if (self.num_segs and self.num_utts) and self.num_segs == self.num_utts:
            self.num_utts = None
        if (self.utt_dur and self.seg_dur) and self.utt_dur == self.seg_dur:
            self.utt_dur = None

        # To be more readable
        self.utt_dur = round(self.utt_dur, 4) if self.utt_dur else None
        self.seg_dur = round(self.seg_dur, 4) if self.seg_dur else None

    def to_dict(self):
        return super().to_dict(filt_type="none", init_field_only=False)


class KaldiDataset:
    """Process kaldi style data dir.

    Args:
        data_dir: datadir of kaldi data format.
        must_exists: files list here must exists to be loaded if exists, use this free config carefully.
        maybe_exists: files list here will be loaded if exists.
        more: if true, ignore files in must_exitsts and maybe_exitsts, load all available files which are exist.
        nj: num_jobs when computing duration.

    Possible attr:
    == Mapping files ==
        self.wav_scp: dict{str:str}
        self.reco2dur: dict{str:float}
        self.feats_scp: dict{str:str}
        self.vad_scp: dict{str:str}
        self.segment: dict{str:list[str]}
        self.utt2spk: dict{str:str}
        self.utt2lang: dict{str:str}
        self.text: dict{str:str}
        self.utt2gender:  dict{str:str}

    == Variables ==
        self.data_dir: str, self._loaded_attr: list
    """

    # Fixed definition of str-first mapping files.
    SUPPORT = OrderedDict(
        [
            ("wav_scp", FileConfig(FileType.UTTFIRST, LoadArgs("wav.scp"))),
            (
                "reco2dur",
                FileConfig(FileType.UTTFIRST, LoadArgs("reco2dur", dtype="float")),
            ),
            (
                "reco2sr",
                FileConfig(FileType.UTTFIRST, LoadArgs("reco2sr", dtype="int")),
            ),
            ("vad_scp", FileConfig(FileType.UTTFIRST, LoadArgs("vad.scp"))),
            ("segments", FileConfig(FileType.SEGFIRST, LoadArgs("segments", vec=True))),
            ("feats_scp", FileConfig(FileType.SEGFIRST, LoadArgs("feats.scp"))),
            (
                "utt2num_frames",
                FileConfig(FileType.SEGFIRST, LoadArgs("utt2num_frames", dtype="int")),
            ),
            ("utt2spk", FileConfig(FileType.SEGFIRST, LoadArgs("utt2spk"))),
            ("utt2lang", FileConfig(FileType.SEGFIRST, LoadArgs("utt2lang"))),
            ("text", FileConfig(FileType.SEGFIRST, LoadArgs("text"))),
        ]
    )

    CUSTOM_FILE = OrderedDict(
        [
            (
                "utt2gender",
                CustomFileConfig(
                    load_args=LoadArgs("spk2gender"),
                    file_type=FileType.SEGFIRST,
                    dest_load_args=LoadArgs("utt2gender"),
                    transform_fn=transform_spk2gender,
                ),
            ),
        ]
    )

    for key in CUSTOM_FILE.keys():
        if key in SUPPORT.keys():
            raise KeyError(
                f"CUSTOM_FILE exists keys {key} conflicted with keys in `KaldiDataset.SUPPORT`{SUPPORT.keys()}."
            )

    _classification_attr = ("utt2spk", "utt2lang", "utt2gender")

    def __init__(
        self,
        data_dir: Union[str, Path],
        must_exists: Optional[List[str]] = None,
        maybe_exists: Optional[List[str]] = None,
        more: bool = False,
        nj: int = 1,
    ):
        self.must_exists = alt_none(must_exists, ["wav_scp"])
        self.maybe_exists = alt_none(
            maybe_exists,
            ["utt2spk", "reco2dur", "reco2sr", "segments", "utt2num_frames"],
        )

        self.more = more
        self.nj = nj

        self.utt_first_attrs = [
            attr
            for attr, conf in chain(self.SUPPORT.items(), self.CUSTOM_FILE.items())
            if conf.file_type == FileType.UTTFIRST
        ]
        self.seg_first_attrs = [
            attr
            for attr, conf in chain(self.SUPPORT.items(), self.CUSTOM_FILE.items())
            if conf.file_type == FileType.SEGFIRST
        ]
        # Init and Load files
        self._loaded_attr = []

        self.data_dir = Path(data_dir)
        assert self.data_dir.is_dir()

        self.load_data_()

        self.initiate_stats()

        logger.info(f"Load kaldi dir done. \n {repr(self)}")

    def load_data_(self):
        if self.more:
            logger.info(
                f"Load mapping files from {self.data_dir} as more as possible with more=True."
            )
        else:
            logger.info(
                f"Load mapping files form {self.data_dir},\n"
                f"w.r.t must exist files {self.must_exists},\n"
                f"maybe exist files {self.maybe_exists}."
            )

        for attr, file_config in chain(self.SUPPORT.items(), self.CUSTOM_FILE.items()):
            transform_fn = None

            if isinstance(file_config, CustomFileConfig):
                if (
                    self.data_dir / str(file_config.dest_load_args.file_name)
                ).is_file():
                    load_args = file_config.dest_load_args
                else:
                    load_args = file_config.load_args
                    transform_fn = file_config.transform_fn
            elif isinstance(file_config, FileConfig):
                load_args = file_config.load_args
            else:
                raise ValueError(
                    "kaldi type file_config must be one of (`FileConfig`, `CustomFileConfig`)."
                )

            if self.more:
                self._load_set_attr(attr, load_args, transform_fn)
            elif attr in self.must_exists + self.maybe_exists:
                self._load_set_attr(
                    attr,
                    load_args,
                    transform_fn,
                    must_exists=(attr in self.must_exists),
                )
            else:
                ...

    def _load_set_attr(
        self,
        attr,
        load_args: LoadArgs,
        transform_fn: Optional[Callable] = None,
        must_exists: bool = False,
    ):
        file_name, *read_args = load_args
        file_path = self.data_dir / str(file_name)
        if file_path.is_file():
            data = read_str_first_ark(file_path, *read_args)
            if transform_fn is not None:
                data = transform_fn(self, data)
            setattr(self, attr, data)

            self.add_loaded_attr(attr)
        elif must_exists:
            raise FileExistsError(f"The file {file_path} is not exist.")
        else:
            ...

    def initiate_stats(self):
        self.generate_reco2dur(nj=self.nj)
        self.generate_utt2frames_ifneed()
        self.generate_segments_ifneed()
        num_utts = len(self.utt_ids) if self.utt_ids else None

        seg_id_set = set(self.seg_ids)
        num_seg_id = len(seg_id_set)

        for attr in self.loaded_attr:
            if attr in self.utt_first_attrs:
                if len(getattr(self, attr)) != num_utts:
                    raise ValueError(
                        f"The length of attr {attr}: {len(attr)} is not matched with other utt-first lengths: {num_utts}."
                    )
            if attr in self.seg_first_attrs:
                if attr == "feats_scp" and len(self.feats_scp) != num_seg_id:
                    raise ValueError(
                        f"The length of attr {attr}: {len(attr)} is not matched with other global seg_id set lengths: {num_seg_id}."
                    )
                assert set(getattr(self, attr)).issubset(
                    seg_id_set
                ), f"seg-first attr {attr} is not a subset of global seg_id set: {num_seg_id}, it may cause key error."

        # Cache spk2seg/utt2seg, so we can lookup segs lately.
        self.clear_cache()
        self._spk_mapping_seg_cache()
        self._utt_mapping_seg_cache()

    def save_data_dir(self, save_dir: Union[str, Path]):
        save_dir = Path(save_dir)
        save_dir.mkdir(parents=True, exist_ok=True)
        info = self.info
        logger.info(
            f"Save utt-first/seg-first data to {save_dir}, "
            f"dataset info:\n{repr_dict(info)}"
        )

        for attr, file_config in chain(self.SUPPORT.items(), self.CUSTOM_FILE.items()):
            if isinstance(file_config, CustomFileConfig):
                load_args = (
                    file_config.dest_load_args
                    if file_config.dest_load_args
                    else file_config.load_args
                )
            elif isinstance(file_config, FileConfig):
                load_args = file_config.load_args
            else:
                raise ValueError(
                    "kaldi type file_config must one of (`FileConfig`, `CustomFileConfig`)."
                )
            file_name, _, vector = load_args

            if (attr in self.utt_first_attrs + self.seg_first_attrs) and (
                attr in self.loaded_attr
            ):
                file_path = save_dir / str(file_name)
                attr_dict = getattr(self, attr)
                save_str_first_ark(attr_dict, file_path, vector=vector)
        save_yaml(info, save_dir / "info.yaml")

    def _info(self) -> KaldiDatasetInfo:
        def get_attr_len(attr: str, src: KaldiDataset):
            if attr in src.loaded_attr:
                return len(getattr(src, attr))
            else:
                return None

        return KaldiDatasetInfo(
            num_utts=len(self.utt_ids) if self.utt_ids else 0,
            num_segs=len(self.seg_ids) if self.seg_ids else None,
            utt_dur=self.utt_dur,
            seg_dur=self.seg_dur,
            num_spks=self.num_spks,
            num_spk_segs=get_attr_len("utt2spk", self),
            num_lang_segs=get_attr_len("utt2lang", self),
            num_gender_segs=get_attr_len("utt2gender", self),
            num_text_segs=get_attr_len("text", self),
            feat_dim=self.feat_dim,
            num_frames=self.num_frames,
        )

    @property
    def info(self) -> Dict:
        return self._info().to_dict()

    @property
    def utt_dur(self):
        if "reco2dur" in self.loaded_attr:
            tot_dur = 0.0
            for _, dur in self.reco2dur.items():
                tot_dur += dur
            tot_dur /= 3600
        else:
            tot_dur = None
        return tot_dur

    @property
    def seg_dur(self):
        if "segments" in self.loaded_attr:
            tot_dur = 0.0
            for _, v in self.segments.items():
                start, end = float(v[1]), float(v[2])
                if end < 0:
                    tot_dur = None
                    break
                tot_dur += end - start
            else:
                tot_dur /= 3600
        else:
            tot_dur = None
        return tot_dur

    @property
    def num_frames(self):
        if "utt2num_frames" in self.loaded_attr:
            tot_frames = 0
            for _, v in self.utt2num_frames.items():
                tot_frames += v
        else:
            tot_frames = None
        return tot_frames

    @property
    def num_spks(self):
        spk2seg = self._spk_mapping_seg_cache()
        return len(spk2seg) if spk2seg else None

    @property
    def utt_ids(self):
        for attr in self.loaded_attr:
            if attr in self.utt_first_attrs:
                return getattr(self, attr).keys()
        return None

    @property
    def seg_ids(self):
        if "segments" in self.loaded_attr:
            return self.segments.keys()
        elif "feats_scp" in self.loaded_attr:
            return self.feats_scp.keys()
        else:
            return self.utt_ids

    @property
    def feat_dim(self):
        if not _KALDI_NATIVE_IO_AVAILABLE:
            raise RuntimeError(
                "Read feat dim from kaldi matrix requires modules:kaldi_native_io, \
                                install it by `pip install kaldi_native_io`."
            )
        import kaldi_native_io

        if "feats_scp" in self.loaded_attr:
            for utt, path in self.feats_scp:
                return kaldi_native_io.MatrixShape.read(path).num_cols
        else:
            feat_dim = None
        return feat_dim

    @property
    def loaded_attr(self):
        return self._loaded_attr

    def add_loaded_attr(self, attr: str) -> List[str]:
        attr_set = set(self.loaded_attr)
        attr_set.add(attr)
        # clear duplicated attrs and sort attr list.
        self._loaded_attr = [
            attr
            for attr in chain(KaldiDataset.SUPPORT, KaldiDataset.CUSTOM_FILE)
            if attr in attr_set
        ]

    def del_loaded_attr(self, attr: str) -> List[str]:
        attr_set = set(self.loaded_attr) - {attr}
        self._loaded_attr = [
            attr
            for attr in chain(KaldiDataset.SUPPORT, KaldiDataset.CUSTOM_FILE)
            if attr in attr_set
        ]

    _utt_mapping_seg: Optional[Dict[str, List[str]]] = None

    def _utt_mapping_seg_cache(self):
        if self._utt_mapping_seg is None:
            mapping = defaultdict(list)
            if "segments" in self.loaded_attr:
                for k, v in self.segments.items():
                    mapping[v[0]].append(k)
            else:
                if self.seg_ids:
                    for seg_id in self.seg_ids:
                        mapping[seg_id].append(seg_id)
                else:
                    return None
            self._utt_mapping_seg = mapping
        return self._utt_mapping_seg

    def find_segids_by_utt(self, utt_id: str) -> Iterable[List[str]]:
        utt_mapping_seg = self._utt_mapping_seg_cache()
        return (seg_id for seg_id in utt_mapping_seg.get(utt_id, []))

    _spk_mapping_seg: Optional[Dict[str, List[str]]] = None

    def _spk_mapping_seg_cache(self) -> Dict[str, List[str]]:
        if "utt2spk" in self.loaded_attr and self._spk_mapping_seg is None:
            self._spk_mapping_seg = _class_mapping_segs(self.utt2spk)
        return self._spk_mapping_seg

    def find_segids_by_spk(self, spk_id: str) -> Iterable[List[str]]:
        """
        Use cache for fast searching segs via speaker.
        """
        spk_mapping_seg = self._spk_mapping_seg_cache()
        return (utt_id for utt_id in spk_mapping_seg.get(spk_id, []))

    def generate_label2segids(self, attr: str):
        assert (
            attr in self._classification_attr
        ), f"Unsupported attr {attr} to yield supervision labels."
        if attr == "utt2spk":
            return self._spk_mapping_seg_cache()
        return _class_mapping_segs(getattr(self, attr, {}))

    def clear_cache(self):
        """
        operation (split) should call this function in the end of creating new dataset.
        """
        self._spk_mapping_seg = None
        self._utt_mapping_seg = None

    def generate_reco2dur(self, nj=1):
        if "wav_scp" in self.loaded_attr and (
            "reco2dur" not in self.loaded_attr
            or len(self.reco2dur) != len(self.wav_scp)
        ):
            if self.data_dir:
                reco2dur_context = (self.data_dir / "reco2dur").open("w")
                reco2sr_context = (self.data_dir / "reco2sr").open("w")
                save_msg = f", save reco2dur and reco2sr to {self.data_dir}"
            else:
                reco2dur_context, reco2sr_context = nullcontext(), nullcontext()
                save_msg = ""

            logger.info(f"Making reco2dur and reco2sr, nj={nj}{save_msg} ...")
            reco2dur, reco2sr = {}, {}
            if nj > 1:
                with ProcessPoolExecutor(
                    nj
                ) as ex, reco2dur_context as dur_f, reco2sr_context as sr_f:
                    for utt, val in tqdm(
                        zip(
                            self.wav_scp.keys(),
                            ex.map(compute_duration, self.wav_scp.values()),
                        ),
                        total=len(self.wav_scp),
                    ):
                        duration, sr = val
                        reco2dur[utt] = duration
                        reco2sr[utt] = sr

                        if dur_f is not None:
                            print(utt, duration, file=dur_f)
                        if sr_f is not None:
                            print(utt, sr, file=sr_f)
            else:
                with reco2dur_context as dur_f, reco2sr_context as sr_f:
                    for utt, val in tqdm(
                        zip(
                            self.wav_scp.keys(),
                            map(compute_duration, self.wav_scp.values()),
                        ),
                        total=len(self.wav_scp),
                    ):
                        duration, sr = val
                        reco2dur[utt] = duration
                        reco2sr[utt] = sr

                        if dur_f is not None:
                            print(utt, duration, file=dur_f)
                        if sr_f is not None:
                            print(utt, sr, file=sr_f)
            setattr(self, "reco2dur", reco2dur)
            self.add_loaded_attr("reco2dur")
            setattr(self, "reco2sr", reco2sr)
            self.add_loaded_attr("reco2sr")

    def generate_utt2frames_ifneed(self):
        if "feats_scp" in self.loaded_attr and (
            "utt2frames" not in self.loaded_attr
            or len(self.utt2frames) != len(self.feats_scp)
        ):
            if not _KALDI_NATIVE_IO_AVAILABLE:
                raise RuntimeError(
                    "Read feat dim from kaldi matrix requires modules:kaldi_native_io, \
                                    install it by `pip install kaldi_native_io`."
                )
            import kaldi_native_io

            if self.data_dir:
                utt2frames_context = (self.data_dir / "utt2frames").open("w")
                save_msg = f", save utt2frames to {self.data_dir}."
            else:
                utt2frames_context = nullcontext()
                save_msg = ""
            msg = f"Making utt2frames according to feats_scp{save_msg} ..."
            utt2frames = {}
            with utt2frames_context as dur_f:
                for utt, feats_ark in tqdm(
                    self.feats_scp, total=len(self.feats_scp), desc=msg, leave=False
                ):
                    feats_map = kaldi_native_io.MatrixShape.read(feats_ark)
                    frames = feats_map.num_rows
                    utt2frames[utt] = frames
                    if dur_f is not None:
                        print(utt, frames, file=dur_f)

            setattr(self, "utt2frames", utt2frames)
            self.add_loaded_attr("utt2frames")

    def generate_segments_ifneed(self):
        if (
            "segments" not in self.loaded_attr
            and "feats_scp" not in self.loaded_attr
            and "wav_scp" in self.loaded_attr
        ):
            has_seg_first = False
            for attr in self.loaded_attr:
                has_seg_first = attr in self.seg_first_attrs
            if has_seg_first:
                if "reco2dur" not in self.loaded_attr:
                    self.generate_reco2dur(nj=self.nj)
                end = self.reco2dur
                self.segments = {k: [k, 0, end[k]] for k in self.wav_scp}
                self.add_loaded_attr("segments")
                if self.data_dir:
                    save_str_first_ark(
                        self.segments, self.data_dir / "segments", vector=True
                    )
        else:
            self.fix_segments()

    def fix_segments(self):
        if "segments" in self.loaded_attr:
            negative_items = [
                (k, v) for k, v in self.segments.items() if float(v[2]) == -1
            ]

            if negative_items:
                assert (
                    "wav_scp" in self.loaded_attr
                ), "need wav_scp to compute duration."
                logger.info(
                    "Fix segments if <end-time> has -1 (which means the segment runs till the end of the WAV file)"
                )
                if "reco2dur" not in self.loaded_attr:
                    self.generate_reco2dur(nj=self.nj)

                renew_segments = {}
                before_seg_num = len(self.segments)
                for negative_item in negative_items:
                    seg, item = negative_item
                    reco, start, end = item
                    end = self.reco2dur[seg]
                    renew_segments[seg] = [reco, start, end]
                self.segments.update(renew_segments)

                if len(self.segments) == before_seg_num:
                    raise ValueError(
                        f"After fixing, the seg num is not matched with origin number, from{before_seg_num} to {len(self.segments)}"
                        f"There may exists some error, try to fix the -1 value in segments manually."
                    )
                if self.data_dir:
                    save_str_first_ark(
                        self.segments, self.data_dir / "segments", vector=True
                    )

    def split(
        self,
        num_ids: int,
        requirement: str = "",
        apply_attr: Optional[str] = None,
        mode: str = "split",
        drop: bool = True,
        seed: int = 1024,
    ) -> Union["KaldiDataset", Tuple["KaldiDataset", "KaldiDataset"]]:
        """
        Args:
            num_ids: int
                The number to be split.
            requirement: str (per_class, total_class, '')
                per_class: Each class random sample `num_ids`.
                total_class: Select num_ids segs in total to contain class label as more as possible.
                utt: Sample `num_ids` utts instead of segs.
            apply_attr: str
                Split dataset via attr.
                if `requirement` is class related ('per_class', 'total_class'), must set this parameter to one of classfication attrs.
                the requirement parameter is apply to this attr, support (utt2spk, utt2lang, utt2gender).
            mode: str ('split', 'subset'),
                If split, return (remain_set, split_set)
                else return (split_set).

        Returns:
            (remain_set, split_set): (KaldiDataset, KaldiDataset) | KaldiDataset .
        """
        np.random.seed(seed)
        assert mode in ("split", "subset")
        requirement = requirement.replace("-", "_")
        remain_part = None

        if requirement == "per_class":
            if apply_attr is None:
                for attr in self._classification_attr:
                    if attr in self.loaded_attr:
                        apply_attr = attr
                        break

            assert (
                apply_attr in self._classification_attr
            ), f"Unsupport classification attr {apply_attr}"
            assert (
                apply_attr in self.loaded_attr
            ), f"class related attr ({apply_attr}) is not loaded."

            if apply_attr == "utt2spk":
                class_mapping_segs = self._spk_mapping_seg_cache()
            else:
                class_mapping_segs = _class_mapping_segs(getattr(self, apply_attr))
            num_classes = len(class_mapping_segs)
            logger.info(
                f"Split KaldiDataset to {num_ids * num_classes} segs and others. requirement (per_class) apply to attr {apply_attr}."
            )

            seg_ids_list = []
            for _, segs in class_mapping_segs.items():
                if len(segs) > num_ids:
                    seg_ids_list.extend(
                        list(np.random.choice(segs, num_ids, replace=False))
                    )
                elif not drop:
                    seg_ids_list.extend(segs)
            seg_ids_list = set(seg_ids_list)
            split_part = self.subset(seg_ids_list, id_type=FileType.SEGFIRST)
            if mode == "split":
                remain_part = self.subset(
                    self.seg_ids - seg_ids_list, id_type=FileType.SEGFIRST
                )

        elif requirement == "total_class":
            if apply_attr is None:
                for attr in self._classification_attr:
                    if attr in self.loaded_attr:
                        apply_attr = attr
                        break

            assert (
                apply_attr in self._classification_attr
            ), f"Unsupoort classification attr {apply_attr}"
            assert (
                apply_attr in self.loaded_attr
            ), f"class related attr ({apply_attr}) is not loaded."
            if apply_attr == "utt2spk":
                class_mapping_segs = self._spk_mapping_seg_cache()
            else:
                class_mapping_segs = _class_mapping_segs(getattr(self, apply_attr))
            num_classes = len(class_mapping_segs)
            class_mapping_segs_keys = list(class_mapping_segs.keys())

            logger.info(
                f"Split KaldiDataset to {num_ids} segs and others, requirement (total_class) apply to attr {apply_attr}."
            )

            num_class_segs = len(getattr(self, apply_attr).keys())
            if num_class_segs < num_ids:
                raise ValueError(
                    f"The target num_segs {num_ids} is out of total class segs {num_class_segs}."
                )

            cls2counter = defaultdict(lambda: 0)

            if num_classes >= num_ids:
                classes = list(
                    np.random.choice(class_mapping_segs_keys, num_ids, replace=False)
                )
                for cls in classes:
                    cls2counter[cls] = 1
            else:
                for cls in class_mapping_segs_keys:
                    cls2counter[cls] += num_ids // num_classes

                if num_ids % num_classes > 0:
                    remain_classes = list(
                        np.random.choice(
                            class_mapping_segs_keys,
                            num_ids % num_classes,
                            replace=False,
                        )
                    )
                    for cls in remain_classes:
                        cls2counter[cls] += 1

            seg_ids_list = []
            for cls, segs in class_mapping_segs.items():
                if len(segs) > cls2counter[cls]:
                    seg_ids_list.extend(
                        list(np.random.choice(segs, cls2counter[cls], replace=False))
                    )
                elif not drop:
                    seg_ids_list.extend(segs)

            remain_num_ids = num_ids - len(seg_ids_list)

            if remain_num_ids > 0:
                seg_ids_list.extend(
                    list(
                        np.random.choice(
                            list(getattr(self, apply_attr).keys() - set(seg_ids_list)),
                            remain_num_ids,
                            replace=False,
                        )
                    )
                )

            seg_ids_list = set(seg_ids_list)
            split_part = self.subset(seg_ids_list, id_type=FileType.SEGFIRST)
            if mode == "split":
                remain_part = self.subset(
                    self.seg_ids - seg_ids_list, id_type=FileType.SEGFIRST
                )

            return remain_part, split_part

        elif requirement == "seg":
            logger.info(
                f"Subset KaldiDataset to {num_ids} segs with (seg) requirement."
            )

            if len(self.seg_ids) >= num_ids:
                seg_ids_list = set(
                    np.random.choice(list(self.seg_ids), num_ids, replace=False)
                )
                split_part = self.subset(seg_ids_list, id_type=FileType.SEGFIRST)
                if mode == "split":
                    remain_part = self.subset(
                        self.seg_ids - seg_ids_list, id_type=FileType.SEGFIRST
                    )
            else:
                raise ValueError(
                    f"The target num_segs {num_ids} is out of total segs {len(self.seg_ids)}."
                )

        elif requirement == "speakers":
            raise NotImplementedError
        elif requirement == "first":
            raise NotImplementedError
        elif requirement == "last":
            raise NotImplementedError
        elif requirement == "shortest":
            raise NotImplementedError
        elif requirement == "spk-list":
            raise NotImplementedError
        elif requirement == "utt-list":
            raise NotImplementedError
        else:
            logger.info(
                f"Subset KaldiDataset to {num_ids} utts with default requirement."
            )
            utts = list(self.utt_ids)
            if len(utts) >= num_ids:
                utt_id_list = set(np.random.choice(utts, num_ids, replace=False))
                split_part = self.subset(utt_id_list, id_type=FileType.UTTFIRST)
                if mode == "split":
                    remain_part = self.subset(
                        set(utts) - utt_id_list, id_type=FileType.UTTFIRST
                    )

            else:
                raise ValueError(
                    f"The target num_utts {num_ids} is out of total utts {len(utts)}."
                )

        remain_part_repr = f"\n{repr(remain_part)}" if remain_part else ""
        logger.info(f"Split done, split part: \n{repr(split_part)}{remain_part_repr}")
        if mode == "split":
            return remain_part, split_part
        else:
            return split_part

    def filter(
        self,
        apply_attr: Union[str, List[str]],
        filter_fn: Callable[[T], bool],
        fn_kwargs: Optional[dict] = None,
        nj: Optional[int] = None,
    ) -> "KaldiDataset":
        """
        Filter dataset according to input `filter_fn`.

        NOTE: The filter operation has following steps:
            - Step 1: gathering the `apply_attr`(s) via func `_format_examples_with_ids` we zips
                those attr dicts to one iterator.
                e.g., attr utt2spk is `{'id10001-1zcIwhmdeo4-00001':'tom', ...}`,
                attr text is `{'id10001-1zcIwhmdeo4-00001': 'i want eat', ...}`, They are
                zipped to -> `({'id': 'id10001-1zcIwhmdeo4-00001', 'utt2spk': 'tom', 'text': 'i want eat'}, ...)`
                However, the attr dict may use seg_id `(utt2spk)` or
                utt_id `(wav_scp)` as its key. Without a common key we cann't zip them.
                Thus if the arg `apply_attr` is a list, they must share the same file type.
                e.g., `['utt2spk', 'text']` is valid while `['utt2spk', 'wav_scp']` is invalid .
            - Step 2: get the items of the above zipped data, for each item is a dict, provide a `filter_fn`
                to decide whether keep this item. Supposed we get a item:
                `zd = {'id': 'id10001-1zcIwhmdeo4-00001', 'utt2spk': 'tom', 'text': 'i want eat'}`
                - Case 1: `apply_attr` is one str `(utt2spk)`, only `zd['utt2spk']` is passed to `filter_fn`.
                    e.g., `lamda x: x == 'tom'` means this item is valid.
                - Case 2: `apply_attr` is sequence `(utt2spk, text)`, `tuple(zd['utt2spk'], zd['text']` is
                    passed to `filter_fn`, and you can use index to get them in your customed function.
                    e.g., `lamda x: (x[0] == 'tom') and ('eat' not in x[1])` means this item is invalid.
            - Step 3: after filter invalid ids, function `subset` is called to get a new dataset.

        Args:
            apply_attr: attr(s) of dataset which ``filter_fn`` is applied.
            filter_fn: customized function to filter ids.
            fn_kwargs: kwargs of `filter_fn`.
            nj: num_jobs.

        Returns:
            A new dataset.
        """

        validate_input_col(filter_fn, apply_attr)
        attr_type = self.__get_attrs_type(apply_attr)
        logger.info(f"Format example lists with attrs: {apply_attr} ...")
        seq = list(self._format_examples_with_ids(apply_attr=apply_attr))
        seq_len = len(seq)
        filter_fn = alt_none(filter_fn, lambda x: True)
        fn_kwargs = alt_none(fn_kwargs, {})
        nj = alt_none(nj, 1)
        filter_ids_fn = functools.partial(
            self._get_filtered_ids_single,
            filter_fn,
            apply_attr=apply_attr,
            fn_kwargs=fn_kwargs,
            return_id=True,
        )

        if nj == 1:
            filter_done = []
            with tqdm(
                total=seq_len, unit=" examples", leave=False, desc="Filter"
            ) as pbar:
                for _, done, stats in filter_ids_fn(seq):
                    if done:
                        filter_done = stats
                    else:
                        pbar.update(stats)
        else:
            splits = split_sequence(seq, nj)
            del seq
            kwargs_per_split = [
                {
                    "items": splits[indice],
                    "split_id": indice,
                }
                for indice in range(len(splits))
            ]

            filter_done = [[] for _ in range(len(splits))]
            with tqdm(
                total=seq_len, unit=" examples", leave=False, desc=f"Filter, nj={nj}"
            ) as pbar:
                for split_id, done, stats in iflatmap_unordered(
                    nj=nj, fn=filter_ids_fn, kwargs_iter=kwargs_per_split
                ):
                    if done:
                        filter_done[split_id] = stats
                    else:
                        pbar.update(stats)
                for kwargs in kwargs_per_split:
                    del kwargs["items"]
                filter_done = functools.reduce(add, filter_done)
        new_dataset = self.subset(filter_done, id_type=attr_type)
        logger.info(f"Filter done\nfrom -> {repr(self)}" f"to -> {new_dataset}")
        return new_dataset

    @staticmethod
    def _get_filtered_ids_single(
        filter_fn: Callable,
        items: List[Dict[str, Any]],
        apply_attr: Union[str, List[str]],
        fn_kwargs: Optional[dict] = None,
        split_id: Optional[int] = None,
        return_id: bool = True,
    ) -> Iterable[Tuple[int, bool, Union[int, List[str]]]]:
        num_examples_progress_update = 0
        result = []
        timer = Timer()

        for item in items:
            if isinstance(apply_attr, (list, tuple)):
                args = tuple(item[col] for col in apply_attr)
                condition = filter_fn(*args, **fn_kwargs)
            else:
                condition = filter_fn(item[apply_attr], **fn_kwargs)
            if not isinstance(condition, bool):
                raise ValueError(
                    "Boolean output is required for `filter_fn`, got ", type(condition)
                )
            if condition:
                result.append(item["id"] if return_id else item)
            num_examples_progress_update += 1

            if timer.elapse() > 0.005:
                yield split_id, False, num_examples_progress_update
                num_examples_progress_update = 0
                timer.reset()

        yield split_id, False, num_examples_progress_update
        yield split_id, True, result

    def _format_examples_with_ids(
        self,
        attr_type: Optional[FileType] = None,
        apply_attr: Optional[Union[str, List[str]]] = None,
    ) -> Iterable[Dict[str, Any]]:
        if (attr_type is not None and apply_attr is not None) or (
            attr_type is None and apply_attr is None
        ):
            raise ValueError("specify attr_type or apply_attr.")
        elif attr_type is not None:
            if attr_type == FileType.UTTFIRST:
                format_key = [
                    attr for attr in self.utt_first_attrs if attr in self.loaded_attr
                ]
                ids = self.utt_ids
            elif attr_type == FileType.SEGFIRST:
                format_key = [
                    attr for attr in self.seg_first_attrs if attr in self.loaded_attr
                ]
                ids = self.seg_ids
            else:
                raise ValueError(
                    f"id_type must be either {FileType.UTTFIRST} or {FileType.SEGFIRST}."
                )
        else:
            attr_type = self.__get_attrs_type(apply_attr)
            if attr_type == FileType.UTTFIRST:
                ids = self.utt_ids
            elif attr_type == FileType.SEGFIRST:
                ids = self.seg_ids
            if not isinstance(apply_attr, (list, tuple)):
                apply_attr = [apply_attr]
            format_key = [attr for attr in apply_attr if attr in self.loaded_attr]
        return (
            dict(
                {attr: getattr(self, attr).get(id, None) for attr in format_key}, id=id
            )
            for id in ids
        )

    def __get_attrs_type(self, apply_attr: Union[str, List[str]]) -> FileType:
        if isinstance(apply_attr, (list, tuple)):
            first_attr = apply_attr[0]
            assert all((attr in self.utt_first_attrs) for attr in apply_attr) or all(
                (attr in self.seg_first_attrs) for attr in apply_attr
            ), "attrs should with the same type (utt_first or seg_first)."
            assert all(
                (attr in self.loaded_attr) for attr in apply_attr
            ), f"all attrs {apply_attr} should exists."
        else:
            assert (
                apply_attr in self.loaded_attr
            ), f"attr {apply_attr} should be exists."
            first_attr = apply_attr
        if first_attr in self.utt_first_attrs:
            attr_type = FileType.UTTFIRST
        elif first_attr in self.seg_first_attrs:
            attr_type = FileType.SEGFIRST
        else:
            raise ValueError("Just support `FileType` (utt_first or seg_first) attrs.")
        return attr_type

    def filter_noseg_utts(self) -> "KaldiDataset":
        """
        In some cases, dataset might have utts without seg_ids to mapping. This function filter those utts.
        e.g., after split dataset with seg_ids, while the utt_ids is unchanged, those utt_ids without seg_ids to mapping
        can be dropped for supervision training.
        """
        assert self.utt_ids is not None
        if self._utt_mapping_seg_cache() is not None:
            utt_id_list = self._utt_mapping_seg_cache().keys()
        logger.info("Filt utts without `seg_id` for supervision training.")
        return self.subset(utt_id_list, FileType.UTTFIRST)

    def subset(self, id_list: set, id_type: int = FileType.UTTFIRST) -> "KaldiDataset":
        """
        id_list: a id set w.r.t utt-id or seg-id. Could be list.
        id_type: (FileType.UTTFIRST, FileType.SEGFIRST).

        @return: KaldiDataset. Return a new KaldiDataset rather than itself, keep items in id_list.
        """
        if len(self.loaded_attr) == 0:
            logger.warning("The KaldiDataset has 0 loaded attr.")
            return self

        kaldi_dataset = copy.deepcopy(self)
        kaldi_dataset.data_dir = None

        if not isinstance(id_list, set):
            id_list = set(id_list)

        if id_type == FileType.UTTFIRST:
            for attr in kaldi_dataset.utt_first_attrs:
                if attr in kaldi_dataset.loaded_attr:
                    this_file_dict = getattr(kaldi_dataset, attr)
                    new_file_dict = {
                        k: v for k, v in this_file_dict.items() if k in id_list
                    }
                    setattr(kaldi_dataset, attr, new_file_dict)

            seg_id_list = []
            for utt_id in id_list:
                seg_id_list.extend(
                    [seg_id for seg_id in kaldi_dataset.find_segids_by_utt(utt_id)]
                )
            seg_id_list = set(seg_id_list)

            for attr in kaldi_dataset.seg_first_attrs:
                if attr in kaldi_dataset.loaded_attr:
                    this_file_dict = getattr(kaldi_dataset, attr)
                    new_file_dict = {
                        k: v for k, v in this_file_dict.items() if k in seg_id_list
                    }
                    setattr(kaldi_dataset, attr, new_file_dict)

        elif id_type == FileType.SEGFIRST:
            for attr in kaldi_dataset.seg_first_attrs:
                if attr in kaldi_dataset.loaded_attr:
                    this_file_dict = getattr(kaldi_dataset, attr)
                    new_file_dict = {
                        k: v for k, v in this_file_dict.items() if k in id_list
                    }
                    setattr(kaldi_dataset, attr, new_file_dict)

        else:
            raise ValueError(f"Support id_type {id_type} with utt or seg only.")

        for attr, file_config in chain(
            kaldi_dataset.SUPPORT.items(), kaldi_dataset.CUSTOM_FILE.items()
        ):
            if (
                file_config.file_type == FileType.CUSTOM
                and attr in kaldi_dataset.loaded_attr
            ):
                delattr(kaldi_dataset, attr)
                kaldi_dataset.del_loaded_attr(attr)

        kaldi_dataset.initiate_stats()

        return kaldi_dataset

    @classmethod
    def register_custom_file(cls, attr: str, custom: CustomFileConfig):
        if attr in cls.CUSTOM_FILE or attr in cls.SUPPORT:
            raise KeyError(f"Failed register, attribute name:{attr} is already known.")
        assert custom is isinstance(
            CustomFileConfig
        ), f"cumstom config should be a type of {CustomFileConfig}."
        cls.CUSTOM_FILE[attr] = custom

    def __repr__(self):
        return (
            f"<class KaldiDataset> (data_dir={self.data_dir}, loaded={self.loaded_attr})\n"
            f"{repr_dict(self.info)}"
        )


# Function
def to(to_type: str, value):
    if to_type == "str" or to_type == "float" or to_type == "int":
        return eval("{0}('{1}')".format(to_type, value))
    else:
        raise ValueError(f"Do not support convert type:{to_type}.")


def read_str_first_ark(
    file_path: Union[str, Path],
    value_type: str = "str",
    vector: bool = False,
    every_bytes: int = 10000000,
) -> Dict:
    this_dict = defaultdict(lambda: None)
    logger.info(f"Load data from {file_path} ...")
    with Path(file_path).open() as reader:
        while True:
            lines = reader.readlines(every_bytes)
            if not lines:
                break
            for line in lines:
                if vector:
                    # split_line => n
                    split_line = line.strip().split()
                    # split_line => n-1
                    key = split_line.pop(0)
                    value = [to(value_type, x) for x in split_line]
                    this_dict[key] = value
                else:
                    key, value = line.strip().split(maxsplit=1)
                    this_dict[key] = to(value_type, value)
    return this_dict


def save_str_first_ark(
    data: Dict[str, Any], file_path: Union[str, Path], vector: bool = False
):
    with file_path.open("w") as f:
        for key, value in sorted(data.items()):
            if vector:
                value = " ".join(map(str, value))
            print(key, value, file=f)


def _class_mapping_segs(data: Dict) -> Dict[str, List[str]]:
    mapping = defaultdict(list)
    for k, v in data.items():
        mapping[v].append(k)
    return mapping


def compute_duration(
    path: Union[str, Path],
) -> Tuple[float, int]:
    """
    Read a audio file.

    path: Path to an audio file or a Kaldi-style pipe.
    @return: (duration, sample_rate)
        float duration of the recording, in seconds.
        sample_rate, int.
    """
    path = str(path)
    if path.strip().endswith("|"):
        if not _KALDI_NATIVE_IO_AVAILABLE:
            raise ValueError(
                "To read Kaldi's data dir where wav.scp has 'pipe' inputs, "
                "please 'pip install kaldi_native_io' first."
            )
        import kaldi_native_io

        wave = kaldi_native_io.read_wave(path)
        assert wave.data.shape[0] == 1, f"Expect 1 channel. Given {wave.data.shape[0]}"

        return wave.duration, wave.sample_freq
    if not torchaudio_info_unfixed:
        info = torchaudio.info(path)
        return info.num_frames / info.sample_rate, info.sample_rate
    else:
        try:
            # Try to parse the file using pysoundfile.
            import soundfile

            info = soundfile.info(path)
            return info.duration, info.samplerate
        except Exception as e:
            raise AudioInfoException(
                f"{e}\n failed get duration wav ({path}, try make reco2dur and reco2sr files yourself.)"
            )


class AudioInfoException(Exception):
    pass
