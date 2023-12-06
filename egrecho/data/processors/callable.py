# -*- coding:utf-8 -*-
# Copyright xmuspeech (Author: Leo 2023-05)

from typing import Callable, Generator, Optional

from torch.utils.data._utils.collate import default_collate

from egrecho.utils.misc import rich_exception_info
from egrecho.utils.patch import StreamWrapper, validate_input_col

__all__ = ["filters", "maps"]


@rich_exception_info
def maps(
    data,
    fn: Callable,
    input_col=None,
    output_col=None,
    **fn_kwds,
) -> Generator:
    r"""Comes from `torch.utils.data.datapipes.MapperIterDataPipe`.

    Applies a function over each item from the source.
    The function can be any regular Python function or partial object. Lambda
    function is not recommended as it is not supported by pickle.

    Args:
        data:
            Input data iterator to be processed.
        fn:
            Function being applied over each item
        input_col:
            Index or indices of data which ``fn`` is applied, such as:
                - ``None`` as default to apply ``fn`` to the data directly.
                - Integer(s) is used for list/tuple.
                - Key(s) is used for dict.

        output_col:
            Index of data where result of ``fn`` is placed. ``output_col`` can be specified
            only when ``input_col`` is not ``None``
                - ``None`` as default to replace the index that ``input_col`` specified; For ``input_col`` with
                multiple indices, the left-most one is used, and other indices will be removed.
                - Integer is used for list/tuple. ``-1`` represents to append result at the end.
                - Key is used for dict. New key is acceptable.
        fn_kwds:
            kwargs for map function.

    Example:
        >>> # xdoctest: +SKIP
        >>> def add_one(x):
        ...     return x + 1
        >>> data = (range(10))
        >>> list(maps(data, add_one))
        [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
        >>> # We discourage the usage of `lambda` functions as they are not serializable with `pickle`
        >>> # Use `functools.partial` or explicitly define the function instead
        >>> mapper2 = maps(data, lambda x: x + 1)
        >>> list(mapper2)
        [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
    """

    if input_col is None and output_col is not None:
        raise ValueError("`output_col` must be None when `input_col` is None.")
    if isinstance(output_col, (list, tuple)):
        if len(output_col) > 1:
            raise ValueError("`output_col` must be a single-element list or tuple")
        output_col = output_col[0]
    validate_input_col(fn, input_col)
    for sample in data:
        yield _apply_fn(sample, fn, input_col, output_col, **fn_kwds)


def _apply_fn(data, fn: Callable, input_col=None, output_col=None, **fn_kwds):
    if input_col is None and output_col is None:
        return fn(data, **fn_kwds)

    if input_col is None:
        res = fn(data, **fn_kwds)
    elif isinstance(input_col, (list, tuple)):
        args = tuple(data[col] for col in input_col)
        res = fn(*args, **fn_kwds)
    else:
        res = fn(data[input_col], **fn_kwds)

    # Copy tuple to list and run in-place modification because tuple is immutable.
    if isinstance(data, tuple):
        t_flag = True
        data = list(data)
    else:
        t_flag = False

    if output_col is None:
        if isinstance(input_col, (list, tuple)):
            data[input_col[0]] = res
            for idx in sorted(input_col[1:], reverse=True):
                del data[idx]
        else:
            data[input_col] = res
    else:
        if output_col == -1:
            data.append(res)
        else:
            data[output_col] = res

    # Convert list back to tuple
    return tuple(data) if t_flag else data


@rich_exception_info
def filters(
    data,
    filter_fn: Callable,
    input_col=None,
    **filter_fn_kwds,
) -> Generator:
    """
    Filters the data based on the given filter function.

    Args:
        data:
            Input data iterator to be processed.
        filter_fn:
            Filter function applied to each item.
        input_col:
            Index or indices of data which `filter_fn` is applied, such as:
            - `None` as default to apply `filter_fn` to the data directly.
            - Integer(s) is used for list/tuple.
            - Key(s) is used for dict.
        output_col:
            Not used in the filter function.
        **filter_fn_kwds:
            Keyword arguments for the `filter_fn` function.

    Yields:
        The filtered data items that satisfy the condition specified by the filter function.

    Raises:
        ValueError: If the output of `filter_fn` is not a boolean value.

    Example:
        >>> def is_even(x):
        ...     return x % 2 == 0
        >>> data = (range(10))
        >>> list(filters(data, is_even))
        [0, 2, 4, 6, 8]
    """
    validate_input_col(filter_fn, input_col)
    for sample in data:
        condition = _apply_filter_fn(sample, filter_fn, input_col, **filter_fn_kwds)
        if not isinstance(condition, bool):
            raise ValueError(
                "Boolean output is required for `filter_fn` of FilterIterDataPipe, got",
                type(condition),
            )
        if condition:
            yield sample
        else:
            StreamWrapper.close_streams(sample)


def _apply_filter_fn(data, fn: Callable, input_col=None, **fn_kwds) -> bool:
    if input_col is None:
        return fn(data, **fn_kwds)
    elif isinstance(input_col, (list, tuple)):
        args = tuple(data[col] for col in input_col)
        return fn(*args, **fn_kwds)
    else:
        return fn(data[input_col], **fn_kwds)


def collate(data, collate_fn: Optional[Callable] = default_collate):
    r"""
    Casts data into tensor in batch, default `collate_fn` see pytorch
    ``torch.utils.data._utils.collate.default_collate``.
    """
    return maps(data, collate_fn)
