# -*- coding:utf-8 -*-
# Copyright xmuspeech (Author: Leo 2023-07)

import codecs
import io
import json
import re
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any, Callable, Dict, List, Literal, Optional, Union

import numpy as np
import torch
import torchaudio

from egrecho.data.datasets.audio import KaldiDataset
from egrecho.data.datasets.constants import (
    AUDIO_COLUMN,
    OFFLINE_FEAT_COLUMN,
    SAMPLE_RATE_COLUMN,
)
from egrecho.data.dew import (
    SHARD_COLUMN,
    SHARD_SIZE_COLUM,
    DataclassDew,
    Dew,
    DewSamples,
    DictDew,
)
from egrecho.data.iterable import (
    EgrechoDistSampler,
    IterabelDatasetWrapper,
    Processor,
    processors,
)
from egrecho.utils.common import alt_none, asdict_filt
from egrecho.utils.io import auto_open
from egrecho.utils.logging import get_logger
from egrecho.utils.misc import rich_exception_info

if TYPE_CHECKING:
    from torch.utils.data.dataset import IterableDataset

logger = get_logger(__name__)


@dataclass
class AudioDew(DataclassDew):
    id: str
    audio_path: Optional[Union[str, Path]] = None
    feat_path: Optional[Union[str, Path]] = None
    start: Optional[Union[float, int]] = None
    duration: Optional[Union[float, int]] = None
    speaker: Optional[str] = None
    text: Optional[str] = None
    language: Optional[str] = None
    gender: Optional[str] = None
    extras: Optional[Dict[str, Any]] = None

    def __post__init__(self):
        if self.feat_path is not None and self.audio_path is not None:
            raise ValueError(
                f"{self.id} should have audio or feat, but got both: "
                f"audio_path={self.audio_path}, feat_path={self.feat_path}."
            )
        if self.audio_path:
            self.start = float(alt_none(self.start, 0.0))
            self.duration = float(self.duration) if self.duration else None
        if self.feat_path:
            self.start = int(alt_none(self.start, 0))
            self.duration = int(self.duration) if self.duration else None

    @property
    def end(self):
        return (self.start + self.duration) if self.duration is not None else None

    def to_dict(self) -> dict:
        return asdict_filt(self, filt_type="default")

    @classmethod
    def from_dict(cls, data: dict) -> "AudioDew":
        return AudioDew(**data)

    def __repr__(self):
        return asdict_filt(self)


@rich_exception_info
def decode_audio_dew(sample: AudioDew):
    ret = sample.to_dict()

    if sample.feat_path is not None:
        import kaldi_native_io

        offset, end = sample.start, sample.end
        arr = kaldi_native_io.FloatMatrix.read(sample.feat_path).numpy()
        ret[OFFLINE_FEAT_COLUMN] = torch.from_numpy(arr[offset:end])

    wav = sample.audio_path

    if wav:
        offset, duration = sample.start, sample.duration

        if offset > 0 or duration is not None:
            try:  # torchaudio.info works
                sample_rate = torchaudio.info(wav).sample_rate
                offset = int(offset * sample_rate)
                num_frames = int(duration * sample_rate) if duration is not None else -1
                waveforms, _ = torchaudio.load(
                    wav, num_frames=num_frames, frame_offset=offset
                )
            except Exception:  # load whole first
                waveforms, sample_rate = torchaudio.load(wav)
                offset = int(offset * sample_rate)
                num_frames = (
                    int(duration * sample_rate)
                    if duration is not None
                    else waveforms.shape[1] - offset
                )
                waveforms = waveforms[:, offset : offset + num_frames]
        else:
            waveforms = torchaudio.load(wav)

        ret[AUDIO_COLUMN] = waveforms
        ret[SAMPLE_RATE_COLUMN] = sample_rate

    return ret


@rich_exception_info
def encode_audio_shard_sample(sample: AudioDew) -> Dict[str, io.BytesIO]:
    """
    Encode an audio sample into bytes stream,
    it is useful when you want to store the sample with its data in binary format.
    Returns a dict compose of metadata and actual data.
    """
    ret = {}
    ret["__key__"] = sample.id
    json_stream = io.BytesIO()
    print(
        json.dumps(sample.to_dict()),
        file=codecs.getwriter("utf-8")(json_stream),
    )
    json_stream.seek(0)
    ret["metadata.json"] = json_stream
    wav = sample.audio_path
    feat = sample.feat_path

    if feat is not None:
        import kaldi_native_io

        offset, end = sample.start, sample.end
        arr = kaldi_native_io.FloatMatrix.read(feat).numpy()
        stream = io.BytesIO()
        np.save(stream, arr[offset:end], allow_pickle=False)
        ret[OFFLINE_FEAT_COLUMN + ".npy"] = stream
    elif wav is not None:
        offset, duration = sample.start, sample.duration
        if offset == 0.0 and duration is None:
            with auto_open(wav, "rb") as fin:
                waveforms = fin.read()
                stream = io.BytesIO(waveforms)
        else:
            try:  # torchaudio.info works
                sample_rate = torchaudio.info(wav).sample_rate
                offset = int(offset * sample_rate)
                num_frames = int(duration * sample_rate) if duration is not None else -1
                waveforms, _ = torchaudio.load(
                    wav, num_frames=num_frames, frame_offset=offset, normalize=False
                )
            except Exception:  # load whole first
                waveforms, sample_rate = torchaudio.load(wav, normalize=False)
                offset = int(offset * sample_rate)
                num_frames = (
                    int(duration * sample_rate)
                    if duration is not None
                    else waveforms.shape[1] - offset
                )
                waveforms = waveforms[:, offset : offset + num_frames]
            stream = io.BytesIO()
            torchaudio.backend.soundfile_backend.save(
                stream,
                waveforms,
                sample_rate,
                format="wav",
            )
        ret[AUDIO_COLUMN + ".wav"] = stream
    else:
        logger.warning(f"Neither wav nor feat exists for sample: {sample.to_dict()}.")
    return ret


@rich_exception_info
def decode_audio_shard(sample: Dict) -> Dict:
    r"""
    Decodes bytes data in dict read from webdataset into python data.

    Usually webDataset using the extensions as keys, i.e.,
    any text after the first "." in the filename is used as a key/extension.
    However, in egrecho we write the tar files with id.key.format style, e.g.,
    e39871fd9fd74f55.audio.wav
    e39871fd9fd74f55.metadata.json
    f18b91585c4d3f3e.audio.wav
    f18b91585c4d3f3e.metadata.json
    With `egrecho.data.iterable.processors.webdataset` it aggregates consecutive
    items with the same basename into a single dictionary as:

    - `{"__key__": "e39871fd9fd74f55", "audio.wav": `io.BytesIO`, "metadata.json": `io.BytesIO`}`
    - `{"__key__": "f18b91585c4d3f3e", "audio.wav": `io.BytesIO`, "metadata.json": `io.BytesIO`}`

    Hence, This function aims to parse above dicts formated like:

    - `{'id': "e39871fd9fd74f55", 'audio': Tensor, 'sample_rate': int, 'speaker': str}`
    - `{'id': "f18b91585c4d3f3e", 'audio': Tensor, 'sample_rate': int, 'speaker': str}`

    Example:
        >>> from egrecho.data.iterable.processors import open_files, load_from_tar, webdataset
        >>> data = list(webdataset(load_from_tar(open_files(['shards-validation-00000-of-00001.tar',], mode="b"))))[0]
        >>> data
        {'__key__': 'shards-validation-00000-of-00001.tar/id00039-y7c_8Xn8G-I-00077',
        '.metadata.json': StreamWrapper<(...)>, '.audio.wav': StreamWrapper<(...)>}
        >>> decode_audio_shard(data)
        [{'id': 'id00039-y7c_8Xn8G-I-00077',
        'audio_path': '.../00077.wav',
        'start': 0.0,
        'duration': 4.288,
        'speaker': 'id00039',
        'audio': tensor([[ 0.0000e+00,  ..., -3.0518e-05, -2.7466e-04]]),
        'sample_rate': 16000}]
    """
    from egrecho.utils.patch import is_stream_handle

    ret = {}
    meta = None
    for k, v in sample.items():
        # TODO: better close file handles
        # if is_stream_handle(v):
        #     ds = v
        #     # The behavior of .read can differ between streams (e.g. HTTPResponse), hence this is used instead
        #     v = b"".join(v)
        #     ds.close()
        extension = (re.sub(r".*[.]", "", k)).lower()
        if extension in ["flac", "mp3", "m4a", "ogg", "opus", "wav", "wma"]:
            waveforms, sample_rate = torchaudio.backend.soundfile_backend.load(v)
            ret[AUDIO_COLUMN] = waveforms
            ret[SAMPLE_RATE_COLUMN] = sample_rate
        if extension in ["json", "jsn"]:
            meta = json.loads(v.read().decode("utf-8"))
            meta.pop(AUDIO_COLUMN, None)
            meta.pop(SAMPLE_RATE_COLUMN, None)
            meta.pop(OFFLINE_FEAT_COLUMN, None)
            ret.update(meta)
        if extension == "npy" and "feat" in k:
            ret[OFFLINE_FEAT_COLUMN] = torch.from_numpy(np.load(v))

        if is_stream_handle(v):
            v.close()
    return ret


class ASVSamples(DewSamples):
    """
    `ASVSamples` extends `DewSamples` and represents a class for asv samples.

    Attributes:
        _dew_cls (AudioDew): The class for audio dew.

    Methods:
        export_shard:
            Exports shards (tar files) of samples.
        fixed_clip:
            To get fixed-chunk samples (not implemented).
        build_source_dp:
            Builds a data pipe for the source.
        load_source_dp:
            Loads the data pipe for the source.
    """

    _dew_cls = AudioDew

    def export_shard(
        self,
        out_dir: Union[str, Path],
        encoder_fn: Callable[[Dew], Dict[str, io.BytesIO]] = encode_audio_shard_sample,
        shard_manifest_dir: Optional[Union[str, Path]] = None,
        chunk_size: Optional[int] = None,
        even_chunksize: bool = True,
        split_name: Optional[str] = None,
        shard_prefix: Optional[str] = None,
        nj: int = 1,
    ):
        return super().export_shard(
            out_dir,
            encoder_fn,
            shard_manifest_dir=shard_manifest_dir,
            chunk_size=chunk_size,
            even_chunksize=even_chunksize,
            split_name=split_name,
            shard_prefix=shard_prefix,
            nj=nj,
        )

    def fixed_clip(self):
        """
        TODO: get fixed-chunk samples.
        """
        raise NotImplementedError

    @classmethod
    def build_source_dataset(
        cls,
        path_or_paths: Union[str, List[str]],
        data_type: Literal["raw", "shard", "offline_feat"] = "raw",
        lazyload_source: Optional[bool] = None,
        shuffle: bool = False,
        partition: bool = False,
        decode_fn: Optional[Callable] = None,
        **kwargs,
    ) -> "IterableDataset":
        """
        Build the source data pipe.

        This function mainly consists of three steps:
            - Step 1: load files.
            - Step 2: construct sampler and iterdataset, which related to shuffle, partition, etc.
            - Step 3: lazy-decode samples according to its data type.

        Args:
            path_or_paths:
                The manifest path(s) to the data source.
            data_type (str, optional):
                The type of data. Defaults to 'raw'.
            lazyload_source (bool, optional):
                Whether to lazy-load manifest. Default: False for shard True for others.
            partition (bool):
                Belongs to sampler_kwargs. If True, sharding samples across ddp. Defaults to False.
            shuffle (bool, optional):
                Belongs to sampler_kwargs, Defaults to False.
            sampler_kwargs (dict, optional):
                popped from kwargs and will passe to sampler for dataset.
            loader_kwargs (dict, optional):
                popped from kwargs and will pass to :method::`from_files/load_shard_manifest`.
                where contains loading logic, e.g., rename manifest column, specify 'id' column, etc.
            \**kwargs:
                additional kwargs.

        Returns:
            Union[IterableDataset, Tuple[IterableDataset, DewSamples]]
            The constructed data pipe or a tuple of datapipe with manifest source

        """
        assert data_type in (
            "raw",
            "offline_feat",
            "shard",
        ), f"Invalid data_type:{data_type}."
        if lazyload_source is None:
            lazyload_source = False if data_type == "shard" else True
        loader_kwargs = kwargs.pop("loader_kwargs", {})
        if data_type == "shard":
            if lazyload_source:
                logger.warning(
                    f"Should load Shard data as list, but got (lazyload_source={lazyload_source}), will"
                    " auto change to eager mode.",
                    ranks=[0],
                )
            data_source = cls.load_shard_manifest(
                path_or_paths, lazy=False, **loader_kwargs
            )
            try:
                shard_sizes = [int(d[SHARD_SIZE_COLUM]) for d in data_source]
            except:  # noqa
                shard_sizes = None

        else:
            data_source = cls.from_files(
                path_or_paths, lazy=lazyload_source, **loader_kwargs
            )

        sampler_kwargs = kwargs.pop("sampler_kwargs", {})
        sampler = EgrechoDistSampler(
            data_source, shuffle=shuffle, partition=partition, **sampler_kwargs
        )
        datapipe = IterabelDatasetWrapper(sampler)

        if data_type == "shard":
            datapipe.src_nsamples = shard_sizes
            datapipe = Processor(
                datapipe, processors.maps, lambda sample: sample[SHARD_COLUMN]
            )
            datapipe = Processor(datapipe, processors.open_files, mode="b")
            datapipe = Processor(datapipe, processors.load_from_tar)
            datapipe = Processor(datapipe, processors.webdataset)
            decode_fn = alt_none(decode_fn, decode_audio_shard)
            datapipe = Processor(datapipe, processors.maps, decode_fn)
        else:
            decode_fn = alt_none(decode_fn, decode_audio_dew)
            datapipe = Processor(datapipe, processors.maps, decode_fn)

        return datapipe


class AudioSamples(ASVSamples):
    r"""
    A more generable verision of `ASVSamples` which handles `DictDew`.

    To handle dews with other schemas, you should pass appropriate functions:

        - Parameter `decode_fn` in `def build_source_dataset(cls, ...):`.
        - Parameter `encoder_fn` in `def export_shard(self, ...):`.
    """

    _dew_cls = DictDew


def load_kaldiset_to_asv(kaldi_set: KaldiDataset) -> ASVSamples:
    """
    Convert a KaldiDataset to an ASVSamples instance.

    This function extracts the 'id', 'speaker', 'start', 'duration', and 'feat_path' from the given KaldiDataset
    and constructs an ASVSamples instance.

    Args:
        kaldi_set (KaldiDataset):
            The KaldiDataset instance to convert.

    Returns:
        ASVSamples.
    """
    spk_dict = getattr(kaldi_set, "utt2spk", defaultdict(lambda: None))

    if "feats_scp" not in kaldi_set.loaded_attr:
        wav_dict = kaldi_set.wav_scp
        if "segments" not in kaldi_set.loaded_attr:
            start_dict = defaultdict(lambda: 0.0)
            duration_dict = getattr(kaldi_set, "reco2dur", defaultdict(lambda: None))
            return ASVSamples.from_dews(
                AudioDew(
                    id=utt_id,
                    audio_path=wav_dict[utt_id],
                    start=float(start_dict[utt_id]),
                    duration=float(duration_dict[utt_id]),
                    speaker=spk_dict[utt_id],
                )
                for utt_id in wav_dict
            )
        else:
            segments_dict = kaldi_set.segments
            return ASVSamples.from_dews(
                AudioDew(
                    id=seg_id,
                    audio_path=wav_dict[segments_dict[seg_id][0]],
                    start=float(segments_dict[seg_id][1]),
                    duration=float(segments_dict[seg_id][2])
                    - float(segments_dict[seg_id][1]),
                    speaker=spk_dict[seg_id],
                )
                for seg_id in segments_dict
            )

    else:
        feat_dict = kaldi_set.feats_scp
        start_dict = defaultdict(lambda: 0)
        duration_dict = getattr(kaldi_set, "utt2num_frames", defaultdict(lambda: None))
        return ASVSamples.from_dews(
            AudioDew(
                id=seg_id,
                feat_path=feat_dict[seg_id],
                start=int(start_dict[seg_id]),
                duration=int(duration_dict[seg_id]),
                speaker=spk_dict[seg_id],
            )
            for seg_id in feat_dict
        )
