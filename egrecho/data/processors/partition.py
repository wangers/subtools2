# -*- coding:utf-8 -*-
# Copyright xmuspeech (Author: Leo 2023-05)

from typing import Generator, Iterable


def partition_one(
    data: Iterable,
    index: int,
    n_shards: int,
) -> Generator:
    """
    Partition data to desired index piece.

    Args:
        data:
            Input data iterator to be processed.
        index:
            target id of total shards.
        n_shards:
            num of total shards.

    Example:
        >>> iterable = range(10)
        >>> list(partition_one(iterable, 1, 3))
    """
    for i, sample in enumerate(data):
        if i % n_shards == index:
            yield sample
