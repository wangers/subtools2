# -*- coding:utf-8 -*-
# Copyright xmuspeech (Author: Leo 2023-05)

import collections
from typing import Iterable, Optional

from egrecho.utils.patch import StreamWrapper

__all__ = ["open_files"]


def open_files(
    data,
    mode: str = "r",
    encoding: Optional[str] = None,
):
    """
    Given pathnames, opens files and yield pathname and file stream
    in a tuple (functional name: ``open_files``).

    Args:
        data:
            Input filenames iterator to be processed.
        mode:
            An optional string that specifies the mode in which
            the file is opened by ``open()``. It defaults to ``r``, other options are
            ``b`` for reading in binary mode and ``t`` for text mode.
        encoding:
            An optional string that specifies the encoding of the
            underlying file. It defaults to ``None`` to match the default encoding of ``open``.

    Note:
        The opened file handles will be closed by Python's GC periodically. Users can choose
        to close them explicitly.

    Example:
        >>> test_file_path = "test_file.txt"
        >>> test_data = [test_file_path]
        >>> file_generator = open_files(test_data)
        >>> list(file_generator)
        [('test_file.txt', StreamWrapper<<_io.TextIOWrapper name='test_file.txt' mode='r' encoding='UTF-8'>>)]
    """

    if mode not in ("b", "t", "rb", "rt", "r"):
        raise ValueError("Invalid mode {}".format(mode))

    if "b" in mode and encoding is not None:
        raise ValueError("binary mode doesn't take an encoding argument")

    yield from gen_file_binaries(data, mode, encoding)


def gen_file_binaries(pathnames: Iterable, mode: str, encoding: Optional[str] = None):
    if not isinstance(pathnames, collections.abc.Iterable):
        pathnames = [
            pathnames,
        ]

    if mode in ("b", "t"):
        mode = "r" + mode

    for pathname in pathnames:
        if not isinstance(pathname, str):
            raise TypeError(
                "Expected string type for pathname, but got {}".format(type(pathname))
            )
        yield pathname, StreamWrapper(open(pathname, mode, encoding=encoding))
