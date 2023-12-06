# -*- coding:utf-8 -*-
# Copyright xmuspeech (Author: Leo 2023-03)

from pathlib import Path
from typing import Optional, Union

from egrecho.utils.io.utils import (
    InvalidPathExtension,
    buf_count_newlines,
    extension_contains,
    load_csv_lazy,
    load_json,
    load_jsonl_lazy,
)


class JsonlIterable:
    """
    Get example iterator from json lines file.
    """

    def __init__(self, path: Union[str, Path], **kwargs) -> None:
        self.path = path
        self._len = None

    def __iter__(self):
        yield from load_jsonl_lazy(self.path)

    def __len__(self) -> int:
        if self._len is None:
            self._len = buf_count_newlines(self.path)
        return self._len


class JsonIterable:
    """
    Get example iterator from json file.
    """

    def __init__(self, path: Union[str, Path], **kwargs) -> None:
        self.path = path
        self._data = load_json(self.path)
        self._len = len(self._data)

    def __iter__(self):
        yield from self._data

    def __len__(self) -> int:
        return self._len


class CsvIterable:
    """
    Get example iterator from csv file.
    """

    def __init__(self, path: Union[str, Path], **csv_reader_kwargs) -> None:
        self.path = path
        self.fmtparams = csv_reader_kwargs
        self._len = None

    def __iter__(self):
        yield from load_csv_lazy(self.path, self.fmtparams)

    def __len__(self) -> int:
        if self._len is None:
            if self.fmtparams.get("fieldnames", None) is None:
                len = buf_count_newlines(self.path) - 1
            else:
                len = buf_count_newlines(self.path)
            self._len = len
        return self._len


def get_lazy_iterable(
    path: Union[str, Path],
    csv_read_kwargs: Optional[dict] = None,
):
    csv_read_kwargs = {} if csv_read_kwargs is None else csv_read_kwargs
    if not Path(path).is_file():
        raise FileNotFoundError(f"{path} not exist.")
    if extension_contains(".jsonl", path):
        return JsonlIterable(path)
    if extension_contains(".json", path):  # note: not really lazy for json.
        return JsonIterable(path)
    elif extension_contains(".csv", path):
        return CsvIterable(path, **csv_read_kwargs)
    else:
        raise InvalidPathExtension(f"Support jsonl, json or csv now, but got {path}.")


def check_input_dataformat(path):
    supported = (".jsonl", ".json", ".csv")
    if not Path(path).is_file():
        raise FileNotFoundError(f"{path} not exist.")

    if not any(extension_contains(sffx, path) for sffx in supported):
        raise InvalidPathExtension(f"Support jsonl, json or csv now, but got {path}.")
