# Adapted and modified from Lightning-AI/utilities:
# https://github.com/Lightning-AI/utilities/blob/main/src/lightning_utilities/core/apply_func.py
#
# Copyright The PyTorch Lightning team.
# Licensed under the Apache License, Version 2.0 (the "License");
#     http://www.apache.org/licenses/LICENSE-2.0

import collections
import copy
import dataclasses
from typing import Any, Callable, Optional, Tuple, Union


def apply_to_collection(
    data: Any,
    dtype: Union[type, Any, Tuple[Union[type, Any]]],
    function: Callable,
    *args: Any,
    wrong_dtype: Optional[Union[type, Tuple[type, ...]]] = None,
    include_none: bool = True,
    allow_frozen: bool = False,
    **kwargs: Any,
) -> Any:
    """Recursively applies a function to all elements of a certain dtype.

    Copied from `Lighting-AI`:
        https://github.com/Lightning-AI/utilities/blob/main/src/lightning_utilities/core/apply_func.py

    Args:
        data: the collection to apply the function to
        dtype: the given function will be applied to all elements of this dtype
        function: the function to apply
        *args: positional arguments (will be forwarded to calls of ``function``)
        wrong_dtype: the given function won't be applied if this type is specified and the given collections
            is of the ``wrong_dtype`` even if it is of type ``dtype``
        include_none: Whether to include an element if the output of ``function`` is ``None``.
        allow_frozen: Whether not to error upon encountering a frozen dataclass instance.
        **kwargs: keyword arguments (will be forwarded to calls of ``function``)

    Returns:
        The resulting collection
    """

    def _is_namedtuple(obj: object) -> bool:
        """Check if object is type nametuple."""
        # https://github.com/pytorch/pytorch/blob/v1.8.1/torch/nn/parallel/scatter_gather.py#L4-L8
        return (
            isinstance(obj, tuple)
            and hasattr(obj, "_asdict")
            and hasattr(obj, "_fields")
        )

    # Breaking condition
    if isinstance(data, dtype) and (
        wrong_dtype is None or not isinstance(data, wrong_dtype)
    ):
        return function(data, *args, **kwargs)

    elem_type = type(data)

    # Recursively apply to collection items
    if isinstance(data, collections.abc.Mapping):
        out = []
        for k, v in data.items():
            v = apply_to_collection(
                v,
                dtype,
                function,
                *args,
                wrong_dtype=wrong_dtype,
                include_none=include_none,
                allow_frozen=allow_frozen,
                **kwargs,
            )
            if include_none or v is not None:
                out.append((k, v))
        if isinstance(data, collections.defaultdict):
            return elem_type(data.default_factory, collections.OrderedDict(out))
        return elem_type(collections.OrderedDict(out))

    is_namedtuple_ = _is_namedtuple(data)
    is_sequence = isinstance(data, collections.abc.Sequence) and not isinstance(
        data, str
    )
    if is_namedtuple_ or is_sequence:
        out = []
        for d in data:
            v = apply_to_collection(
                d,
                dtype,
                function,
                *args,
                wrong_dtype=wrong_dtype,
                include_none=include_none,
                allow_frozen=allow_frozen,
                **kwargs,
            )
            if include_none or v is not None:
                out.append(v)
        return elem_type(*out) if is_namedtuple_ else elem_type(out)

    if dataclasses.is_dataclass(data) and not isinstance(data, type):
        # make a deepcopy of the data,
        # but do not deepcopy mapped fields since the computation would
        # be wasted on values that likely get immediately overwritten
        fields = {}
        memo = {}
        for field in dataclasses.fields(data):
            field_value = getattr(data, field.name)
            fields[field.name] = (field_value, field.init)
            memo[id(field_value)] = field_value
        result = copy.deepcopy(data, memo=memo)
        # apply function to each field
        for field_name, (field_value, field_init) in fields.items():
            v = None
            if field_init:
                v = apply_to_collection(
                    field_value,
                    dtype,
                    function,
                    *args,
                    wrong_dtype=wrong_dtype,
                    include_none=include_none,
                    allow_frozen=allow_frozen,
                    **kwargs,
                )
            if not field_init or (not include_none and v is None):  # retain old value
                v = getattr(data, field_name)
            try:
                setattr(result, field_name, v)
            except dataclasses.FrozenInstanceError as e:
                if allow_frozen:
                    # Quit early if we encounter a frozen data class; return `result` as is.
                    break
                raise ValueError(
                    "A frozen dataclass was passed to `apply_to_collection` but this is not allowed."
                ) from e
        return result

    # data is neither of dtype, nor a collection
    return data
