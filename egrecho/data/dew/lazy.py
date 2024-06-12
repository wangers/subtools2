# -*- coding:utf-8 -*-
# Copyright xmuspeech (Author: Leo 2023-07)

import uuid
import warnings
from itertools import chain
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple, Union

import numpy as np

from egrecho.data.processors import concat_columns_id, rename_columns
from egrecho.utils.data_utils import Dillable, buffer_shuffle
from egrecho.utils.io.reader import get_lazy_iterable


class LazyDict(Dillable):
    def __iter__(self):
        raise NotImplementedError

    def values(self):
        yield from self

    def keys(self):
        return (item.id for item in self)

    def items(self):
        return ((item.id, item) for item in self)


class LazyChainIterable(Dillable):
    """
    Iterates all underlying iterables sequentially.

    NOTE: if any of the input iterables is a dict, we'll iterate only its values.
    """

    def __init__(self, *iterables: Iterable) -> None:
        self.iterables = []
        for it in iterables:
            # flatten
            if isinstance(it, LazyChainIterable):
                for sub_it in it.iterables:
                    self.iterables.append(sub_it)
            else:
                self.iterables.append(it)

    def __iter__(self):
        for it in self.iterables:
            if isinstance(it, dict):
                it = it.values()
            yield from it

    def __len__(self) -> int:
        return sum(len(it) for it in self.iterables)

    def __add__(self, other) -> "LazyChainIterable":
        return LazyChainIterable(self, other)


class LazyShuffler(Dillable):
    """
    Shuffle in lazy mode.

    Data items first fills a buffer with `buffer_size`, then randomly samples elements from this buffer,
    replacing the selected elements with new elements.

    Args:
        iterator : Iterable
        buffer_size : int.
        rng : np.random.Generator

    Returns:
        Generator yields data item.
    """

    def __init__(
        self,
        iterable: Iterable,
        buffer_size: int = 10000,
        rng: Optional[np.random.Generator] = None,
    ) -> None:
        self.iterable = iterable
        self.buffer_size = buffer_size
        self.rng = rng

    def __iter__(self):
        return iter(
            buffer_shuffle(
                iter(self.iterable),
                buffer_size=self.buffer_size,
                rng=self.rng,
            )
        )

    def __len__(self) -> int:
        return len(self.iterable)

    def __add__(self, other) -> "LazyChainIterable":
        return LazyChainIterable(self, other)


# FIXME: add random id -> add index id.
class LazyDictReader(Dillable):
    """
    Read data file(s) lazily into dicts.

    This class is a simple wrapper that loads jsonl/json/csv files lazily into dictionaries.
    Note that json is a dict so it not really lazy and will load it into memory for all.
    It allows you to rename column names or format a key: 'id'.

    Args:
        path_or_paths (Union[str, Path, List[Union[str, Path]]]):
            The file path(s) to read data from.
        rename_col_map (Dict[str, str], optional):
            A dictionary that maps old column names to new names. Default is None.
        concat_col_id (Union[List[str], Tuple[str, ...], None], optional):
            A list of columns to concatenate into a new column with the key 'id'. Default is None.
        join_str (str):
            join str for concat col.
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
        rename_col_map: Dict[str, str] = None,
        concat_col_id: Union[List[str], Tuple[str, ...], None] = None,
        easy_check: bool = True,
        random_id_ifneed: bool = False,
        **kwargs,
    ) -> None:
        if not isinstance(path_or_paths, (list, tuple)):
            path_or_paths = [path_or_paths]
        self.files = path_or_paths
        csv_read_kwargs = kwargs.get("csv_read_kwargs", None)
        iterable = LazyChainIterable(
            *[
                get_lazy_iterable(path, csv_read_kwargs=csv_read_kwargs)
                for path in self.files
            ]
        )
        if rename_col_map:
            iterable = rename_columns(iterable, rename_col_map, easy_check=easy_check)

        if concat_col_id:
            join_str = kwargs.pop("join_str", "_")
            iterable = concat_columns_id(
                iterable, concat_col_id, easy_check=easy_check, join_str=join_str
            )
        self.iterator = iterable
        self.random_id_ifneed = random_id_ifneed

    def __iter__(self):
        has_id = True
        iterator = iter(self.iterator)
        try:
            first = next(iterator)
            _ = first["id"]
        except (AttributeError, KeyError):
            has_id = False
            warn_msg = (
                f"It seems the samples lack an 'id' key in {self.files}, "
                f"the picked one is {first} in file:{self.files[0]}. If not intend to ignore this: "
                "HINT: Provide it in manifest or or choose strategy. "
                "via `rename_col_map`/`concat_col_id`/`random_id_ifneed`."
            )

            if self.random_id_ifneed:
                warnings.warn(
                    f"{warn_msg}\n It seems you seem use random uuuid to generate ids, please notice your seed."
                )
            else:
                warnings.warn(f"{warn_msg}\n It seems you ignore this.")
        except StopIteration:
            return
        for item in chain([first], iterator):
            if not has_id and self.random_id_ifneed:
                item["id"] = str(uuid.uuid4())
            yield item

    def __len__(self) -> int:
        return len(self.iterator)

    def __add__(self, other) -> "LazyChainIterable":
        return LazyChainIterable(self, other)
