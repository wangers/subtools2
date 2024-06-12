# -*- coding:utf-8 -*-
# Copyright xmuspeech (Author: Leo 2023-07)

import io
import random
import shutil
from abc import ABC, abstractmethod
from dataclasses import dataclass
from itertools import chain, islice, repeat
from pathlib import Path
from typing import Any, Callable, Dict, Iterable, List, Optional, Tuple, Type, Union

import numpy as np
from tqdm.contrib import tqdm

from egrecho.data.dew.lazy import LazyChainIterable, LazyDictReader, LazyShuffler
from egrecho.utils.common import ObjectDict, Timer, alt_none, asdict_filt
from egrecho.utils.data_utils import (
    ichunk_size,
    iflatmap_unordered,
    split_sequence,
    try_length,
)
from egrecho.utils.logging import get_logger

logger = get_logger()


SHARD_COLUMN = "shard_path"
SHARD_SIZE_COLUM = "shard_size"


class Dew:
    # to be implemented in subclass.
    from_dict: Callable[[Dict], "Dew"]
    to_dict: Callable[["Dew"], Dict]


@dataclass
class DataclassDew(Dew, ABC):
    id: str
    extras: Optional[Dict[str, Any]] = None

    @classmethod
    @abstractmethod
    def from_dict(data: dict) -> "DataclassDew":
        ...

    @abstractmethod
    def to_dict(self) -> dict:
        ...

    def __setattr__(self, key: str, value: Any) -> None:
        if key in self.__dataclass_fields__:
            super().__setattr__(key, value)
        else:
            extras = alt_none(self.extras, {})
            if value is None:
                extras.pop(key, None)
            else:
                extras[key] = value
            if extras:
                self.extras = extras

    def __getattr__(self, name: str) -> Any:
        try:
            return self.extras[name]
        except (ArithmeticError, KeyError):
            raise AttributeError(f"No such attr: {name}.")

    def __getitem__(self, name: str):
        return getattr(self, name)

    __setitem__ = __setattr__


class DictDew(ObjectDict, Dew):
    """
    It is more flexible compared with `DataclassDew` as its field is variable, however you should manually convert dataformats.
    """

    def __init__(self, id=None, **kwargs):
        """
        Maybe an 'id' key.
        """
        super().__init__(id=id, **kwargs)

    @staticmethod
    def from_dict(data: dict):
        """
        Recommend to override it in subclass.
        """
        return DictDew(**data)

    def to_dict(self) -> dict:
        """
        Recommend to override it in subclass.
        """
        data = asdict_filt(data)
        if id_ := data.pop('id', None) is not None:
            data['id'] = id_
        return data


class DewSamples:
    """
    Base class for
    """

    _dew_cls: Type[Dew] = DictDew

    def __init__(self, dews: Optional[Iterable[Dew]] = None):
        self.dews = alt_none(dews, [])

    @classmethod
    def from_dicts(cls, data: Iterable[dict]) -> "DewSamples":
        dew_cls = cls._dew_cls
        return cls.from_dews(dew_cls.from_dict(dew) for dew in data)

    def to_dicts(self) -> Iterable[dict]:
        return (dew.to_dict() for dew in self)

    @classmethod
    def from_dews(cls, dews: Iterable[Dew]) -> "DewSamples":
        dews = iter(dews)
        try:
            first = next(dews)
        except StopIteration:
            return cls()  # empty
        if not isinstance(first, cls._dew_cls):
            raise ValueError(
                f"Fast failed, the first item is not an instance of {cls._dew_cls}"
            )
        dews = chain([first], dews)
        return cls(dews=list(dews))

    @classmethod
    def from_files(
        cls,
        path_or_paths: Union[str, Path, List[Union[str, Path]]],
        lazy: bool = True,
        rename_col_map: Dict[str, str] = None,
        concat_col_id: Union[List[str], Tuple[str, ...], None] = None,
        easy_check: bool = True,
        random_id_ifneed: bool = False,
        **kwargs,
    ) -> "DewSamples":
        """
        Read file(s) from jsonl/json/csv. Note lazy json is invalid.

        Note that json is a dict so it not really lazy and will load it into memory for all
        even in lazy mode.

        Args:
            path_or_paths (Union[str, Path, List[Union[str, Path]]]):
                The file path(s) to read data from.
            lazy (bool):
                If False, will eagerly loading.  Defaults is True.
            rename_col_map (Dict[str, str], optional):
                A dictionary that maps old column names to new names. Default is None.
            concat_col_id (Union[List[str], Tuple[str, ...], None], optional):
                A list of columns to concatenate into a new column with the key 'id'. Default is None.
            easy_check (bool, optional):
                If True, assume that all data samples share the same structure and
                only check the first one while loading. Default is True.
            random_id_ifneed (bool, optional):
                If True, generate a random 'id' using Python's `uuid` if it is missing. Default is False.
            **kwargs:
                Additional keyword arguments.
        """
        if lazy:
            return cls(
                LazyDewIterator(
                    path_or_paths,
                    dew_cls=cls._dew_cls,
                    rename_col_map=rename_col_map,
                    concat_col_id=concat_col_id,
                    easy_check=easy_check,
                    random_id_ifneed=random_id_ifneed,
                    **kwargs,
                )
            )
        else:
            return cls(
                LazyDewIterator(
                    path_or_paths,
                    dew_cls=cls._dew_cls,
                    rename_col_map=rename_col_map,
                    concat_col_id=concat_col_id,
                    easy_check=easy_check,
                    random_id_ifneed=random_id_ifneed,
                    **kwargs,
                )
            ).to_eager()

    @classmethod
    def open_writer(cls, path: Union[str, Path], overwrite: bool = True, **kwargs):
        """
        Get file handler of jsonl file to write line by line.
        """
        from egrecho.utils.io.writer import SequentialDewWriter

        return SequentialDewWriter(path, overwrite=overwrite, **kwargs)

    def to_file(
        self, path: Union[str, Path], overwrite: bool = False, **kwargs
    ) -> "DewSamples":
        logger.info(f"Save data to {path} ...")
        with self.open_writer(path, overwrite=overwrite, **kwargs) as w:
            for dew in tqdm(self, desc="Save", dynamic_ncols=True, leave=False):
                w.write(dew)
        return self.from_files(path)

    def split_to_files(
        self,
        out_dir: Union[str, Path],
        chunk_size: Optional[int] = None,
        split_num: Optional[int] = None,
        split_name: Optional[str] = None,
        even_chunksize: bool = False,
        prefix: Optional[str] = None,
        gzip: bool = False,
    ) -> List[Union[str, Path]]:
        """
        Splits self into a list of subset. As a lazy way, it requires an
        extra placeholder `out_dir`, which is useful to dump huge data file.

        NOTE: First infer the chunk sizes of splits, then split self according
        to these number. The final file defaults to e.g., "egs-00001-of-00002.jsonl".
        when the reult only have one file, file name changes to "egs.jsonl".
        There exists cases as follows:
            - Case 1: both `chunk_size` and `split_num` is None, it means
                export a single manifest.
            - Case 2: the loaded can get total length, infer split chunk_sizes
                by given `split_num` or `chunk_size`, for the edge case total length less than
                a given `split_num`, it will only results `total length` number files which is
                discouraged.
            - Case 3: unknown total length, `chunk_size` must be provided.

        Args:
            out_dir:
                output manifest dir.
            chunk_size:
                Number of samples per split, may infected by `enven_chunksize=True`.
            split_num:
                Total numbers of splits, conflict with `chunk_size`.
            enven_chunksize (default: False):
                If True, the max num differ between splits is 1.
            prefix (defaults to 'egs'):
                prefix of exported manifest file name.
            split_name (defaults to ''):
                String concated after `prefix`.

        Returns:
            List of exported file paths.
        """
        if chunk_size is not None and split_num is not None:
            raise ValueError("Can't set both chunk_size and split_num.")

        total_lens = try_length(self)

        if chunk_size is None and split_num is None:
            chunk_sizes = [float("inf")]

        elif total_lens:
            if split_num and split_num > total_lens:
                logger.warning(
                    f"Do you really mean to split file with length ({total_lens}) to a greater "
                    f" number of ({split_num}) splits, this will result a less number ({total_lens}) of split? "
                    f" Make sure this case is under your control."
                )
            chunk_sizes = ichunk_size(
                total_lens,
                split_num=split_num,
                chunk_size=chunk_size,
                even=even_chunksize,
            )
        else:
            assert (
                chunk_size > 0
            ), f"Failed to get total length of data, need provide chunk_size, but got chunk_size={chunk_size}."
            chunk_sizes = repeat(chunk_size)

        out_dir = Path(out_dir)
        out_dir.mkdir(parents=True, exist_ok=True)
        split_name = "" if (not split_name) else f"-{split_name}"
        prefix = alt_none(prefix, "egs")
        SHARD_SUFFIX = "SSSSS-of-NNNNN"

        # TODO: find how to rename the inner filename of jsonl.gz formmat file.
        suffix = ".jsonl.gz" if gzip else ".jsonl"
        fpath = (out_dir / f"{prefix}{split_name}-{SHARD_SUFFIX}").with_suffix(suffix)

        data = iter(self)
        done_files = []

        timer = Timer()

        logger.info(f"Generating split files {fpath.name} to {out_dir} ...")
        pbar = tqdm(
            unit=" examples",
            total=total_lens,
            leave=False,
        )

        with pbar:
            num_examples_progress_update = 0
            for i, chunk_size in enumerate(chunk_sizes):
                try:
                    written = 0
                    split_fpath = str(fpath).replace("SSSSS", f"{i:05d}")
                    with self.open_writer(split_fpath) as w:
                        while written < chunk_size:
                            dew = next(data)
                            written += w.write(dew)
                            num_examples_progress_update += 1
                            if timer.elapse() > 0.005:
                                pbar.update(num_examples_progress_update)
                                num_examples_progress_update = 0
                                timer.reset()

                except StopIteration:
                    break
                finally:
                    done_files.append(split_fpath)
            pbar.update(num_examples_progress_update)
        total_splits = len(done_files)
        split_paths = []
        for done_file in done_files:
            if total_splits == 1:
                done_fpath = str(fpath).replace(f"-{SHARD_SUFFIX}", "")
            else:
                done_fpath = str(done_file).replace("NNNNN", f"{total_splits:05d}")
            Path(done_file).rename(done_fpath)
            split_paths.append(done_fpath)
        return split_paths

    def export_shard(
        self,
        out_dir: Union[str, Path],
        encoder_fn: Callable[[Dew], Dict[str, io.BytesIO]],
        shard_manifest_dir: Optional[Union[str, Path]] = None,
        chunk_size: Optional[int] = None,
        even_chunksize: bool = False,
        split_name: Optional[str] = None,
        shard_prefix: Optional[str] = None,
        nj: int = 1,
    ):
        """
        Exports samples to shards (similar to webdataset).

        The process consists of three steps:
            Step 1: execute `split_to_files` to get manifest of splits.
            Step 2: exports shard for each split by `_export_shard_single`,
                it can be multiproccsing.
            Step 3: exports a manifest (e.g., egs-shards.jsonl) have shards'
                info, i.e., `shard_size`, `id`, `shard_path`.
        The final shards file defaults to e.g., "shards-00001-of-00002.tar",
        with a manifest (egs-shards.jsonl, maybe.) points to all shards.

        Args:
            out_dir:
                output shards dir.
            encoder_fn:
                Function decide how to shard raw data.
            shard_manifest_dir (defaults to None):
                If provided, will copy the manifest of shards to it.
            chunk_size:
                Number of samples per shard, may infected by `enven_chunksize=True`.
            enven_chunksize (default: False):
                If True, the max num differ between shards is 1.
            shard_prefix (defaults to 'shards'):
                prefix of exported shard file name.
            split_name (defaults to ''):
                String concated after `prefix`.
            nj:
                num of jobs.
        """
        out_dir = Path(out_dir)
        split_paths = self.split_to_files(
            out_dir=out_dir / "shards_metadata",
            chunk_size=chunk_size,
            even_chunksize=even_chunksize,
            split_name=split_name,
        )
        splits = [self.from_files(f) for f in split_paths]
        num_shards = len(splits)
        total_lens = None
        try:
            total_lens = sum([len(split) for split in splits])
        except Exception:
            pass
        shard_prefix = alt_none(shard_prefix, "shards")
        split_name = "" if (not split_name) else f"-{split_name}"
        SHARD_SUFFIX = f"SSSSS-of-{num_shards:05d}"
        fpath = (out_dir / f"{shard_prefix}{split_name}-{SHARD_SUFFIX}").with_suffix(
            ".tar"
        )
        kwargs_per_split = [
            {
                "split": splits[indice],
                "encoder_fn": encoder_fn,
                "fpath": str(fpath).replace("SSSSS", f"{indice:05d}"),
                "split_id": indice,
            }
            for indice in range(num_shards)
        ]

        nj = int(min(nj, num_shards))
        export_done = [[] for _ in range(num_shards)]
        shards_done = 0
        if nj == 1:
            with tqdm(
                total=total_lens,
                unit=" examples",
                leave=False,
                desc=f"SHARDING, nj={nj}, {shards_done}/{num_shards} shards.",
            ) as pbar:
                for split_id, done, stats in chain.from_iterable(
                    self._export_shard_single(**kwargs) for kwargs in kwargs_per_split
                ):
                    if done:
                        export_done[split_id] = stats
                        shards_done += 1
                        pbar.set_description(
                            f"SHARDING, nj={nj}, {shards_done}/{num_shards} shards."
                        )
                    else:
                        pbar.update(stats)
        else:
            with tqdm(
                total=total_lens,
                unit=" examples",
                leave=False,
                desc=f"SHARDING, nj={nj}, {shards_done}/{num_shards} shards.",
            ) as pbar:
                for split_id, done, stats in iflatmap_unordered(
                    nj=nj, fn=self._export_shard_single, kwargs_iter=kwargs_per_split
                ):
                    if done:
                        export_done[split_id] = stats
                        shards_done += 1
                        pbar.set_description(
                            f"SHARDING, nj={nj}, {shards_done}/{num_shards} shards."
                        )
                    else:
                        pbar.update(stats)

        shard_manifest = (out_dir / f"egs-{shard_prefix}{split_name}").with_suffix(
            ".jsonl"
        )

        dews = []
        for idx, export in enumerate(export_done):
            num_samples, fpath = export
            dew = dict(id=f"{str(idx)}-{shard_prefix}{split_name}")
            dew[SHARD_SIZE_COLUM] = num_samples
            dew[SHARD_COLUMN] = fpath
            dews.append(dew)
        shard_dews = DewSamples.from_dicts(dews)
        shard_dews.to_file(shard_manifest)

        alias_msg = ""
        if shard_manifest_dir is not None:
            shard_manifest_dir = Path(shard_manifest_dir)
            shard_manifest_dir.mkdir(parents=True, exist_ok=True)
            shutil.copy(shard_manifest, shard_manifest_dir)
            alias_msg = f" alias into {str(shard_manifest_dir)}"
        logger.info(
            f"Sharding Done, The shard manifest is {shard_manifest}{alias_msg}."
        )

    @staticmethod
    def _export_shard_single(
        split: "DewSamples",
        encoder_fn: Callable[[Dew], Dict[str, io.BytesIO]],
        fpath: Union[str, Path],
        split_id: Optional[int],
    ):
        """
        Export a single shard, see `export_shard`.
        """
        from egrecho.utils.io.writer import ShardWriter

        timer = Timer()
        num_examples_progress_update = 0
        written = 0

        with ShardWriter(pattern=fpath) as writer:
            for dew in split:
                data = encoder_fn(dew)
                try:
                    written += writer.write(data)
                except Exception as e:
                    logger.warning(e)
                    continue
                num_examples_progress_update += 1
                if timer.elapse() > 0.05:
                    yield split_id, False, num_examples_progress_update
                    num_examples_progress_update = 0
                    timer.reset()
        yield split_id, False, num_examples_progress_update
        yield split_id, True, (written, fpath)

    @staticmethod
    def load_shard_manifest(
        path_or_paths: Union[str, Path, List[Union[str, Path]]],
        lazy: bool = False,
        **kwargs,
    ):
        random_id_ifneed = kwargs.pop('random_id_ifneed', True)
        if lazy:
            return DewSamples(
                LazyDewIterator(
                    path_or_paths, random_id_ifneed=random_id_ifneed, **kwargs
                )
            )
        else:
            return DewSamples(
                LazyDewIterator(
                    path_or_paths, random_id_ifneed=random_id_ifneed, **kwargs
                )
            ).to_eager()

    @property
    def data(self) -> List[Dew]:
        return self.dews

    @property
    def ids(self) -> Iterable[str]:
        return (c.id for c in self.data)

    @property
    def is_lazy(self) -> bool:
        """
        Indicates whether this object is lazy or not.
        """
        return not isinstance(self.data, (dict, list, tuple))

    def to_eager(self):
        """
        Convert lazy mode to a dict set in memory.
        """
        if not self.is_lazy:
            return self
        cls = type(self)
        return cls.from_dews(self)

    def head(self, n: int = 5):
        return list(self.take(n))

    def take(self, n: int):
        return islice(iter(self), n)

    def sample(self, n: int = 1) -> Union[Dew, "DewSamples"]:
        """
        Randomly sample.
        When ``n`` is 1, will return a single Dew; otherwise will return a ``DewSamples``.
        """
        assert n > 0
        indices = random.sample(range(len(self)), min(n, len(self)))
        dews = [self[idx] for idx in indices]
        if n == 1:
            return dews[0]
        cls_ = type(self)
        return cls_(dews)

    def map(self, fn: Callable):
        """TODO
        map fn is useful to transform to desirable data format.
        """
        ...

    def shuffle(
        self,
        seed: Optional[int] = 42,
        rng: Optional[np.random.Generator] = None,
        buffer_size: int = 20_000,
    ):
        """
        Shuffles the elements and returns a shuffled variant of self.
        If data is lazy, it applys a buffer shuffle.

        Args:
            buffer_size: int
            seed: int
            rng: np.random.Generator
                the real generator controls shuffle, if specified, seed param is invalid.
            buffer_size: int
                for lazy shuffle buffer.

        Returns:
            A new DewSamples.
        """
        if rng is not None and not isinstance(rng, np.random.Generator):
            raise ValueError(
                "The provided rng must be an instance of numpy.random.Generator"
            )
        if rng is None:
            if seed is None:
                _, seed, pos, *_ = np.random.get_state()

                seed = seed[pos] if pos < 624 else seed[0]
                _ = np.random.random()  # imitate 1 step

            rng = np.random.default_rng(seed)
        cls = type(self)

        if self.is_lazy:
            return cls(LazyShuffler(self.dews, buffer_size=buffer_size, rng=rng))
        else:
            new: List = self.data.copy()
            rng.shuffle(new)
            return cls(new)

    def split(
        self, split_num: int, shuffle: bool = False, drop_last: bool = False
    ) -> List["DewSamples"]:
        """
        Split into `split_num` list of subset. As an eager way, it may require significant memory storage.

        Args:
            num_splits : int.
                Split num.
            shuffle:
                If true, shuffle input sequence before split it.
            drop_last:
                If true, drop last items when `len(seq)` is not divisible by `num_splits`.

        Returns:
            List of smaller squences.
        """
        cls_ = type(self)
        return [
            cls_(subset)
            for subset in split_sequence(
                self, split_num=split_num, shuffle=shuffle, drop_last=drop_last
            )
        ]

    def __repr__(self) -> str:
        try:
            len_val = len(self)
        except Exception:
            len_val = "<unknown>"
        return f"<class {type(self).__name__}> (len={len_val}) [data type: {type(self.data).__name__}]\n"
        #    f"Head: {self.head()}."

    def __contains__(self, other: Union[str, Dew]) -> bool:
        if isinstance(other, str):
            return any(other == item.id for item in self)
        else:
            return any((other.id == item.id and other == item) for item in self)

    def __getitem__(self, key: Union[int, str]) -> "Dew":
        try:
            return self.dews[key]  # int passed, eager manifest, fast
        except TypeError:
            # either lazy manifest or str passed, both are slow
            if self.is_lazy:
                return next(item for idx, item in enumerate(self) if idx == key)
            else:
                # string id passed, support just for backward compatibility, not recommended
                return next(item for item in self if item.id == key)

    def __iter__(self) -> Iterable[Dew]:
        yield from self.data

    def __len__(self) -> int:
        return len(self.data)

    def __add__(self, other):
        cls = type(self)
        return cls(LazyChainIterable(self.data, other.data))

    def __eq__(self, other: "DewSamples") -> bool:
        return self.data == other.data


# FIXME: add random id -> add index id.
class LazyDewIterator(LazyDictReader):
    """
    Read data file(s) lazily into dews.

    This class is a simple wrapper that loads jsonl/json/csv files lazily.
    It allows you to rename column names and checks for the existence of a key: 'id'.

    Note that json is a dict so it not really lazy and will load it into memory for all.

    Args:
        path_or_paths (Union[str, Path, List[Union[str, Path]]]):
            The file path(s) to read data from.
        dew_cls:
            A type of `egrecho.data.dew.Dew` which has :method::`from_dict`.
        rename_col_map (Dict[str, str], optional):
            A dictionary that maps old column names to new names. Default is None.
        concat_col_id (Union[List[str], Tuple[str, ...], None], optional):
            A list of columns to concatenate into a new column with the key 'id'. Default is None.
        easy_check (bool, optional):
            If True, assume that all data samples share the same structure and only check the first one.
            Default is True.
        random_id_ifneed (bool, optional):
            If True, generate a random 'id' using Python's `uuid` if it is missing. Default is False.
        **kwargs:
            Additional keyword arguments.

    Returns:
        Generator: A generator that yields data items as dictionaries.
    """

    def __init__(
        self,
        path_or_paths: Union[str, Path, List[Union[str, Path]]],
        dew_cls: Type[Dew] = DictDew,
        rename_col_map: Dict[str, str] = None,
        concat_col_id: Union[List[str], Tuple[str, ...], None] = None,
        easy_check: bool = True,
        random_id_ifneed: bool = False,
        **kwargs,
    ) -> None:
        super().__init__(
            path_or_paths=path_or_paths,
            rename_col_map=rename_col_map,
            concat_col_id=concat_col_id,
            random_id_ifneed=random_id_ifneed,
            easy_check=easy_check,
            **kwargs,
        )
        self.dew_cls = dew_cls

    def __iter__(self):
        yield from map(self.dew_cls.from_dict, self.iterator)


def split_raw_file(
    datafile: Union[str, Path],
    split_num: Optional[int] = None,
    chunk_size: Optional[int] = None,
    even_chunksize: bool = False,
    out_dir: Optional[Union[str, Path]] = None,
    sub_dir: str = "",
    split_name: Optional[str] = None,
    prefix: Optional[str] = None,
    gzip: bool = False,
) -> List[Union[str, Path]]:
    """
    Splits file into a list of jsonl files. split files saved to:
    ``out_dir/sub_dir``, where `out_dir` defaults to the dir of datafile with name `"split_jsonl"` and
    `sub_dir` defaults to ``""``.
    This method is useful for splitting datafile (e.g., one possible circumstance is multiprocessing preprocess.)

    NOTE: First infer the chunk sizes of splits, then split self according
    to these number. The final file defaults to e.g., "splits-00001-of-00002.jsonl".
    when the reult only   have one file, file name changes to "egs.jsonl".
    There exists cases as follows:
        - Case 1: both `chunk_size` and `split_num` is None, it means
            export a single manifest.
        - Case 2: the loaded can get total length, infer split chunk_sizes
            by given `split_num` or `chunk_size`, for the edge case total length less than
            a given `split_num`, it will only results `total length` number files which is
            discouraged.
        - Case 3: unknown total length, `chunk_size` must be provided.

    Args:
        datafile:
            datafile supported by :function::`egrecho.io.utils.get_lazy_iterable` to be splitted.
        split_num:
            Total numbers of splits, conflict with `chunk_size`.
        chunk_size:
            Number of samples per split, may infected by `enven_chunksize=True`.
        out_dir:
            savedir for splits. Default to the same level of datafile with name `"splits_jsonl"`
        sub_dir:
            user may want to group split files here.
        enven_chunksize (default: False):
            If True, the max num differ between splits is 1.
        prefix (defaults to 'splits'):
            prefix of exported manifest file name.
        split_name (defaults to ''):
            String concated after `prefix`.

    Returns:
        List of exported file paths.
    """
    from egrecho.utils.io import SequentialDewWriter, get_lazy_iterable

    if chunk_size is not None and split_num is not None:
        raise ValueError("Can't set both chunk_size and split_num.")
    data = get_lazy_iterable(datafile)
    total_lens = try_length(data)

    if chunk_size is None and split_num is None:
        chunk_sizes = [float("inf")]

    elif total_lens:
        if split_num and split_num > total_lens:
            logger.warning(
                f"Do you really mean to split file with length ({total_lens}) to a greater "
                f" number of ({split_num}) splits, this will result a less number ({total_lens}) of split? "
                f" Make sure this case is under your control."
            )
        chunk_sizes = ichunk_size(
            total_lens,
            split_num=split_num,
            chunk_size=chunk_size,
            even=even_chunksize,
        )
    else:
        assert (
            chunk_size > 0
        ), f"Failed to get total length of data, need provide chunk_size, but got chunk_size={chunk_size}."
        chunk_sizes = repeat(chunk_size)
    if out_dir is None:
        out_dir = Path(datafile).parent / "splits_jsonl"
    save_dir = Path(out_dir) / sub_dir
    save_dir.mkdir(parents=True, exist_ok=True)

    split_name = "" if (not split_name) else f"-{split_name}"
    prefix = alt_none(prefix, "splits")
    SHARD_SUFFIX = "SSSSS-of-NNNNN"

    # TODO: find how to rename the inner filename of jsonl.gz formmat file.
    suffix = ".jsonl.gz" if gzip else ".jsonl"
    fpath = (save_dir / f"{prefix}{split_name}-{SHARD_SUFFIX}").with_suffix(suffix)

    data_iter = iter(data)
    done_files = []

    timer = Timer()

    logger.info(f"Generating split files {fpath.name} to {save_dir} ...")
    pbar = tqdm(
        unit=" examples",
        total=total_lens,
        leave=False,
    )

    with pbar:
        num_examples_progress_update = 0
        for i, chunk_size in enumerate(chunk_sizes):
            try:
                written = 0
                split_fpath = str(fpath).replace("SSSSS", f"{i:05d}")
                with SequentialDewWriter(split_fpath) as w:
                    while written < chunk_size:
                        dew = next(data_iter)
                        written += w.write(dew)
                        num_examples_progress_update += 1
                        if timer.elapse() > 0.05:
                            pbar.update(num_examples_progress_update)
                            num_examples_progress_update = 0
                            timer.reset()

            except StopIteration:
                break
            finally:
                done_files.append(split_fpath)
        pbar.update(num_examples_progress_update)
    total_splits = len(done_files)
    split_paths = []
    for done_file in done_files:
        if total_splits == 1:
            done_fpath = str(fpath).replace(f"-{SHARD_SUFFIX}", "")
        else:
            done_fpath = str(done_file).replace("NNNNN", f"{total_splits:05d}")
        Path(done_file).rename(done_fpath)
        split_paths.append(done_fpath)
    return split_paths
