# -*- coding:utf-8 -*-
# Copyright xmuspeech (Author: Leo 2023-05)

from typing import Any, Iterator, List, TypeVar

__all__ = [
    "batch",
    "unbatch",
]

T_co = TypeVar("T_co", covariant=True)


def batch(data, batch_size: int, drop_last: bool = False) -> Iterator[List[T_co]]:
    r"""
    Creates mini-batches of data. An outer dimension will be added as
    ``batch_size`` if ``drop_last`` is set to ``True``, or ``length % batch_size``
    for the last batch if ``drop_last`` is set to ``False``.

    Args:
        data:
            Input data iterator to be processed.
        batch_size:
            The size of each batch
        drop_last:
            Option to drop the last batch if it's not full

    Example:
        >>> # xdoctest: +SKIP
        >>> data = (range(10))
        >>> list(batch(data, batch_size=3, drop_last=True))
        [[0, 1, 2], [3, 4, 5], [6, 7, 8]]
    """
    assert batch_size > 0, "Batch size is required to be larger than 0!"
    batch = []
    for sample in data:
        batch.append(sample)
        if len(batch) == batch_size:
            yield batch
            batch = []
    if len(batch) > 0 and not drop_last:
        yield batch


def unbatch(data, unbatch_level: int = 1) -> Iterator[Any]:
    r"""
    Undoes batching of data, i.e., flattens the data up to the specified level.

    Args:
        data:
            Iterable data being un-batched
        unbatch_level:
            Defaults to ``1`` (only flattening the top level). If set to ``2``,
            it will flatten the top two levels, and ``-1`` will flatten the entire levels.

    Example:
        >>> src_data = [[[0, 1], [2]], [[3, 4], [5]], [[6]]]
        >>> list(unbatch(src_data))
        [[0, 1], [2], [3, 4], [5], [6]]
        >>> list(unbatch(src_data, unbatch_level=2))
        [0, 1, 2, 3, 4, 5, 6]
    """
    for element in data:
        yield from _dive(element, unbatch_level=unbatch_level)


def _dive(element, unbatch_level: int) -> Iterator[Any]:
    if unbatch_level < -1:
        raise ValueError("unbatch_level must be -1 or >= 0")
    if unbatch_level == -1:
        if isinstance(element, list):
            for item in element:
                yield from _dive(item, unbatch_level=-1)
        else:
            yield element
    elif unbatch_level == 0:
        yield element
    else:
        if isinstance(element, list):
            for item in element:
                yield from _dive(item, unbatch_level=unbatch_level - 1)
        else:
            raise IndexError(
                f"unbatch_level {unbatch_level} exceeds the depth of the iterator"
            )
