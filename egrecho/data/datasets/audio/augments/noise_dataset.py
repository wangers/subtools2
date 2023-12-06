# -*- coding:utf-8 -*-

# Copyright xmuspeech (Author: Leo 2023-02-11)

import csv
import functools
import io
import math
import multiprocessing
import pickle
import random
import shutil
import warnings
from concurrent.futures.process import ProcessPoolExecutor
from dataclasses import dataclass, field
from itertools import chain
from pathlib import Path
from typing import Any, ClassVar, List, Optional, Tuple, Union

import lmdb
import numpy as np
import torch
import torchaudio
from tqdm.contrib import tqdm

from egrecho.data.datasets.audio.augments.base import NoiseSet
from egrecho.utils.imports import _H5PY_AVAILABLE
from egrecho.utils.logging import get_logger

if _H5PY_AVAILABLE:
    import h5py

logger = get_logger()


@functools.lru_cache(maxsize=None)
def lru_hdf5_opener(storage_path: str):
    """
    https://github.com/lhotse-speech/lhotse/blob/master/lhotse/features/io.py#lookup_cache_or_open

    Helper internal function used in HDF5 readers.
    It opens the HDF files and keeps their handles open in a global program cache
    to avoid excessive amount of syscalls (frequent use-case).
    """
    if _H5PY_AVAILABLE:
        return h5py.File(storage_path, "r")
    else:
        raise ModuleNotFoundError("use pip to install h5py.")


@functools.lru_cache(maxsize=None)
def lru_lmdb_opener(storage_path: str):
    return lmdb.open(
        str(storage_path),
        readonly=True,
        lock=False,
        readahead=False,
        max_dbs=10,
    )


@dataclass
class NoiseSetConfig:
    db_file: Union[Path, str]
    max_len_cut: Optional[float] = (None,)
    filt_min: float = field(default=0.0, repr=False)


def write_info(file: Path, data: List[List[Any]], head: List = None):
    with open(file, "w", newline="") as csvfile:
        writer = csv.writer(csvfile, delimiter=" ")

        writer.writerow(head)
        for row in data:
            writer.writerow(row)


@dataclass
class Hdf5NoiseSet(NoiseSet):
    """
    This class aims to load noises from hdf5 dataset .
    Just support Mono now, if given multi channel, return the first.

    Arguments
    ---------
    db_file: Path
        Location of database.
    max_len_cut: float
        Limit the chunk, wave longer than it will be cut to save memory.
    filt_min: float
        Filts too short waves.
    """

    db_type: ClassVar[str] = "hdf5"
    suffix: ClassVar[str] = ".h5"

    db_file: Union[str, Path]
    max_len_cut: Optional[float] = field(default=None, repr=False)
    filt_min: float = field(default=0.0, repr=False)

    valid_length: int = field(init=False)
    sample_rate: int = field(init=False)

    def __post_init__(self):
        self.db_file = Path(self.db_file).with_suffix(Hdf5NoiseSet.suffix)
        self.db = lru_hdf5_opener(self.db_file.resolve())
        # If got closed handle, reopen.
        if not self.db:
            lru_hdf5_opener.cache_clear()
            self.db = lru_hdf5_opener(self.db_file.resolve())

        self.sample_rate = self.get_sample_rate()
        self.load_lists = self._get_load_lst()
        self.valid_length = len(self.load_lists)

    def sample(
        self,
        cnts: int = 1,
    ) -> Union[torch.Tensor, List[torch.Tensor]]:
        """
        Random select wav tensors from data base.

        Arguments
        ---------
        cnts: int
            If > 0, sample a batch.

        Returns
        -------
        waveforms: List of wav tensors.
        lengths: Tensor of shape `[batch]`.
        """
        assert (
            0 < cnts <= self.valid_length
        ), f"Sample {cnts} from {self.valid_length} error."

        indices = random.sample(range(self.valid_length), cnts)
        inputs = [self.load_lists[index] for index in indices]
        values = []
        sample_rates = []
        for input in inputs:
            samples, sr = self.load(*input)
            # choose the first channel
            values.append(samples[:1, :])
            sample_rates.append(sr)
        if cnts == 1:
            values = values[0]
            sample_rates = sample_rates[0]
        return values, sample_rates

    def _get_load_lst(
        self,
    ) -> List[Tuple[str, int, int]]:
        loads = []
        for info in self.infos():
            key, duration, sr = info
            if duration < self.filt_min:
                continue
            start = 0
            max_frames = -1
            if self.max_len_cut:
                max_frames = int(self.max_len_cut * sr)
                start = random.randint(0, max(0, int(sr * duration) - max_frames))
            loads.append((key, start, max_frames))
        assert (
            len(loads) > 0
        ), f"No valid waves, check datasets and filt_min param:{self.filt_min}."
        return loads

    def __getitem__(self, key):
        return self.load(key)

    def load(self, key, frame_offset=0, num_frames=-1):
        data = self.db[key]
        bits_per_sample = data.attrs["bits_per_sample"]
        rate = data.attrs["sample_rate"]
        samples = data[()].astype(np.float32) / (1 << (bits_per_sample - 1))
        if num_frames < 0:
            stop = -1
        else:
            stop = frame_offset + num_frames

        return torch.from_numpy(samples)[:, frame_offset:stop], rate

    def keys(self):
        return self.db.keys()

    def values(self):
        for k in self.db:
            yield self[k]

    def items(self):
        for k in self.db:
            yield k, self[k]

    def get_sample_rate(self):
        return next(self.infos())[2]

    def info(self, key):
        return self.db[key].attrs["duration"], self.db[key].attrs["sample_rate"]

    def infos(self):
        for k in self.db:
            yield k, *self.info(k)

    @classmethod
    def create_db(
        cls,
        db_file: Union[str, Path],
        wave_items: Union[List[Tuple[str]], List[Tuple[str, str]]],
        mode: str = "w",
        max_length: Optional[float] = None,
        resample: int = 16000,
        nj: int = 1,
        record_csv: Optional[Union[str, Path]] = None,
    ):
        """
        Create data base.

        Arguments
        ---------
        db_file:
            Location of database.
        wave_items:
            A list of tuples formed as (utt_key, utt_path) or (utt_path).
        mode:
            w    Create file, truncate if exists (default)
            x    Create file, fail if exists
            a    Read/write if exists, create otherwise
        max_length:
            The maximum length in seconds. Waveforms longer
            than this will be cut into pieces.
        resample:
            If not None, store utts in one sample rate.
        nj:
            num_jobs. Multi reader and single writer.
        record_csv:
            If not None, given a path to record the added keys of this operation.

        Returns
        -------
        List of (utt_key, duration, sample_rate) processed.

        """

        db_file = Path(db_file).with_suffix(cls.suffix)
        stem, suffix = db_file.stem, f"{cls.suffix}.csv"
        csv_file = db_file.with_name(stem + suffix)

        if record_csv:
            record_csv_file = Path(record_csv)
            assert (
                not record_csv_file.exists()
            ), f"Record_csv exits {record_csv}, check it."
        read_fn = functools.partial(
            cls._process_utt, max_length=max_length, resample=resample
        )
        db = h5py.File(db_file, mode=mode)
        ok_items = []
        logger.info(
            f"Start creating database, sample rate of waves will be {resample}."
        )
        if nj > 1:
            # torch.set_num_threads(1)
            # torch.set_num_interop_threads(1)
            with ProcessPoolExecutor(
                nj, mp_context=multiprocessing.get_context("spawn")
            ) as ex:
                logger.info(
                    f"Storing utt to create hdf5 database ({db_file}), nj={nj} ..."
                )
                for audio_clip in tqdm(
                    chain.from_iterable(ex.map(read_fn, wave_items)),
                    total=len(wave_items),
                ):
                    id = audio_clip.id
                    data = db.create_dataset(
                        id,
                        data=(audio_clip.samples).numpy(),
                    )
                    data.attrs["sample_rate"] = audio_clip.sample_rate
                    data.attrs["bits_per_sample"] = audio_clip.bits_per_sample
                    data.attrs["duration"] = audio_clip.duration
                    ok_items.append(
                        [str(id), f"{audio_clip.duration:.5f}", audio_clip.sample_rate]
                    )
        else:
            logger.info(
                f"Storing utt to create hdf5 database ({db_file}), num_jobs={nj}"
            )
            for audio_clip in tqdm(
                chain.from_iterable(map(read_fn, wave_items)),
                total=len(wave_items),
            ):
                id = audio_clip.id
                data = db.create_dataset(
                    id,
                    data=(audio_clip.samples).numpy(),
                )
                data.attrs["sample_rate"] = audio_clip.sample_rate
                data.attrs["bits_per_sample"] = audio_clip.bits_per_sample
                data.attrs["duration"] = audio_clip.duration
                ok_items.append(
                    [str(id), f"{audio_clip.duration:.5f}", audio_clip.sample_rate]
                )
        db.close()
        logger.info(f"Create hdf5 database ({db_file}) done, record infos to csv files")
        csv_head_meta = ["id", "duration", "sample_rate"]
        if record_csv:
            write_info(record_csv, ok_items, head=csv_head_meta)

        result_db = Hdf5NoiseSet(db_file=db_file)
        db_infos = []
        for info in result_db.infos():
            db_infos.append(info)
        write_info(csv_file, db_infos, head=csv_head_meta)
        result_db.close()
        logger.info("Record infos done.")
        return ok_items

    @classmethod
    def _process_utt(
        cls,
        wav_item,
        max_length=None,
        resample=16000,
    ):
        results = []
        try:
            results = NoiseSet._load_utts(wav_item, max_length, resample)
        except Exception as e:
            msg = f"{e}\nProcess wav ('{wav_item[1]}') error, pass it."
            warnings.warn(msg)
            pass
        return results

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.db.close()

    def close(self):
        self.db.close()

    def __del__(self):
        self.close()


@dataclass
class LmdbNoiseSet(NoiseSet):
    """
    This class aims to load noises from lmdb dataset .
    Just support Mono now, if given multi channel, return the first.

    Arguments
    ---------
    db_file: Path
        location of database.
    """

    db_type: ClassVar[str] = "lmdb"
    suffix: ClassVar[str] = ".lmdb"

    db_file: Union[str, Path]
    max_len_cut: Optional[float] = field(default=None, repr=False)
    filt_min: float = field(default=0.0, repr=False)

    valid_length: int = field(init=False)
    sample_rate: int = field(init=False)

    def __post_init__(self):
        self.db_file = Path(self.db_file).with_suffix(LmdbNoiseSet.suffix)
        self.db = lru_lmdb_opener(self.db_file.resolve())
        # If got closed handle, reopen.
        try:
            self.db.stats()
        except Exception:
            lru_lmdb_opener.cache_clear()
            self.db = lru_lmdb_opener(self.db_file.resolve())
        self.info_db = self.db.open_db(b"info")
        self.sample_rate = self.get_sample_rate()
        self.load_lists = self._get_load_lst()
        self.valid_length = len(self.load_lists)

    def sample(
        self,
        cnts: int = 1,
    ) -> Union[torch.Tensor, List[torch.Tensor]]:
        """
        Random select wav tensors from data base.

        Arguments
        ---------
        cnts:
            If > 0, sample a batch.

        Returns
        -------
        waveforms: List of wav tensors.
        lengths: Tensor of shape `[batch]`.
        """
        assert (
            0 < cnts <= self.valid_length
        ), f"Sample {cnts} from {self.valid_length} error."

        indices = random.sample(range(self.valid_length), cnts)
        inputs = [self.load_lists[index] for index in indices]
        values = []
        sample_rates = []
        for input in inputs:
            samples, sr = self.load(*input)
            # choose the first channel
            values.append(samples[:1, :])
            sample_rates.append(sr)
        if cnts == 1:
            values = values[0]
            sample_rates = sample_rates[0]
        return values, sample_rates

    def _get_load_lst(
        self,
    ) -> List[Tuple[str, int, int]]:
        loads = []
        for info in self.infos():
            key, duration, sr = info
            if duration < self.filt_min:
                continue
            start = 0
            max_frames = -1
            if self.max_len_cut:
                max_frames = int(self.max_len_cut * sr)
                start = random.randint(0, max(0, int(sr * duration) - max_frames))
            loads.append((key, start, max_frames))
        assert (
            len(loads) > 0
        ), f"No valid waves, check datasets and filt_min param:{self.filt_min}."
        return loads

    def keys(self):
        with self.db.begin(write=False) as txn:
            obj = txn.get(b"__keys__")
            assert obj is not None
            keys = pickle.loads(obj)
            assert isinstance(keys, list)
        return keys

    def __getitem__(self, key):
        return self.load(key)

    def load(self, key, frame_offset=0, num_frames=-1):
        with self.db.begin(write=False) as txn:
            data = txn.get(key.encode())
            data, rate = torchaudio.backend.soundfile_backend.load(
                io.BytesIO(data), frame_offset=frame_offset, num_frames=num_frames
            )
        return data, rate

    def values(self):
        for k in self.keys():
            yield self[k]

    def items(self):
        for k in self.keys():
            yield k, self[k]

    def get_sample_rate(self):
        return next(self.infos())[2]

    def info(self, key):
        with self.db.begin(write=False, db=self.info_db) as txn:
            inf = txn.get(key.encode())
            inf = pickle.loads(inf)
        return inf["duration"], inf["sample_rate"]

    def infos(self):
        for k in self.keys():
            yield k, *self.info(k)

    @classmethod
    def create_db(
        cls,
        db_file: Union[str, Path],
        wave_items: Union[List[Tuple[str]], List[Tuple[str, str]]],
        mode: str = "w",
        max_length: Optional[float] = None,
        resample: int = 16000,
        nj: int = 1,
        record_csv: Optional[Union[str, Path]] = None,
    ):
        """
        Create data base. It do the same thing as `Hdf5NoiseSet.creat_db`.
        """
        assert mode in ["a", "x", "w", "w+"]
        db_file = Path(db_file).with_suffix(cls.suffix)
        stem, suffix = db_file.stem, f"{cls.suffix}.csv"
        csv_file = db_file.with_name(stem + suffix)

        if db_file.exists():
            if mode == "x" and db_file.exists():
                raise ValueError(f'write mode is "x", but db {db_file} exists.')
            if mode == "w":
                shutil.rmtree(db_file)

        if record_csv:
            record_csv_file = Path(record_csv)
            assert (
                not record_csv_file.exists()
            ), f"Record_csv exits {record_csv_file}, check it."

        read_fn = functools.partial(
            cls._process_utt, max_length=max_length, resample=resample
        )
        db = lmdb.open(str(db_file), map_size=int(math.pow(1024, 4)), max_dbs=10)  # 1TB
        info_db = db.open_db(b"info")
        txn = db.begin(write=True)
        ok_items = []
        success = 0
        logger.info(
            f"Start creating database, sample rate of waves will be {resample}."
            f" Storing utt to create lmdb database ({db_file}), num_jobs={nj}"
        )
        if nj > 1:
            # torch.set_num_threads(1)
            # torch.set_num_interop_threads(1)
            with ProcessPoolExecutor(
                nj, mp_context=multiprocessing.get_context("spawn")
            ) as ex:
                for audio_clip in tqdm(
                    chain.from_iterable(ex.map(read_fn, wave_items)),
                    total=len(wave_items),
                ):
                    id, samples, duration, sample_rate = audio_clip
                    info = dict(duration=duration, sample_rate=sample_rate)
                    txn.put(str(id).encode(), pickle.dumps(info), db=info_db)
                    txn.put(str(id).encode(), samples)
                    success += 1
                    ok_items.append([str(id), f"{duration:.5f}", sample_rate])

                    if success % 100 == 0:
                        txn.commit()
                        txn = db.begin(write=True)
        else:
            for audio_clip in tqdm(
                chain.from_iterable(map(read_fn, wave_items)),
                total=len(wave_items),
            ):
                id, samples, duration, sample_rate = audio_clip
                info = dict(duration=duration, sample_rate=sample_rate)
                txn.put(str(id).encode(), pickle.dumps(info), db=info_db)
                txn.put(str(id).encode(), samples)
                success += 1

                if success % 100 == 0:
                    txn.commit()
                    txn = db.begin(write=True)
                ok_items.append([str(id), f"{duration:.5f}", sample_rate])
        txn.commit()
        db_infos = []
        keys = []
        with db.begin(write=False) as txn:
            for id, info in txn.cursor(db=info_db):
                key = id.decode()
                inf = pickle.loads(info)
                db_infos.append([key, inf["duration"], inf["sample_rate"]])
                keys.append(key)
        with db.begin(write=True) as txn:
            txn.put(b"__keys__", pickle.dumps(keys))
        db.sync()
        db.close()

        logger.info(f"Create lmdb database ({db_file}) done, record infos to csv files")
        csv_head_meta = ["id", "duration", "sample_rate"]
        if record_csv:
            write_info(record_csv, ok_items, head=csv_head_meta)

        write_info(csv_file, db_infos, head=csv_head_meta)

        logger.info("Record infos done.")
        return ok_items

    @classmethod
    def _process_utt(
        cls,
        wav_item,
        max_length=None,
        resample=16000,
    ):
        results = []
        try:
            if (
                torchaudio_info_unfixed(wav_item[1])
                or torchaudio.info(wav_item[1]).sample_rate != resample
                or max_length is not None
            ):
                audio_clips = cls._load_utts(wav_item, max_length, resample)
                [
                    results.append(
                        (
                            clip.id,
                            clip.to_bytes().getvalue(),
                            clip.duration,
                            clip.sample_rate,
                        )
                    )
                    for clip in audio_clips
                ]
            else:
                id = wav_item[0]
                sample_rate = torchaudio.info(wav_item[1]).sample_rate
                duration = torchaudio.info(wav_item[1]).num_frames / sample_rate
                with open(wav_item[1], "rb") as fin:
                    data = fin.read()
                    results.append((id, data, duration, sample_rate))

        except Exception as e:
            msg = f"{e}\nProcess wav ('{wav_item[1]}') error, pass it."
            warnings.warn(msg)
            pass
        return results

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()

    def close(self):
        self.db.close()

    def __del__(self):
        self.close()


def torchaudio_info_unfixed(path_or_fileobj: Union[Path, str, io.BytesIO]):
    is_mp3 = isinstance(path_or_fileobj, (str, Path)) and str(path_or_fileobj).endswith(
        ".mp3"
    )
    is_fileobj = isinstance(path_or_fileobj, io.BytesIO)
    return is_mp3 or is_fileobj
