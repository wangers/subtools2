# -*- coding:utf-8 -*-
# Copyright xmuspeech (Author: Leo 2022-12-22)

import copy
import dataclasses
import functools
import os
import random
import re
import time
import warnings
from dataclasses import dataclass, fields, is_dataclass
from enum import Enum
from itertools import chain, islice
from os import PathLike
from typing import Any, Dict, List, Optional, Sequence, Tuple, Union

import numpy as np
import torch
from torch.nn.utils.rnn import pad_sequence


def set_all_seed(seed=None, deterministic=True):
    """This is refered to https://github.com/lonePatient/lookahead_pytorch/blob/master/tools.py."""
    if seed is not None:
        random.seed(seed)
        os.environ['PYTHONHASHSEED'] = str(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        # some cudnn methods can be random even after fixing the seed
        # unless you tell it to be deterministic
        torch.backends.cudnn.deterministic = deterministic


def alt_none(item: Optional[Any], alt_item: Any) -> Any:
    """
    Replace None with  ``alt_item``.
    """
    return alt_item if item is None else item


def is_in_range(
    val, max_val: Optional[Any] = None, min_val: Optional[Any] = None
) -> bool:
    """
    Value in range judging.

    Range is close interval (e.g., [1, 2]), If the boundary is None, skip that condition.

    Args:
        val (_type_)
        max_val (Optional[Any], optional): Defaults to None.
        min_val (Optional[Any], optional): Defaults to None.

    Returns:
        bool
    """
    if min_val is not None and max_val is not None:
        return min_val <= val <= max_val
    elif min_val is not None:
        return val >= min_val
    elif max_val is not None:
        return val <= max_val
    else:
        return True


# Inspired by tornado
# https://www.tornadoweb.org/en/stable/_modules/tornado/util.html#ObjectDict
_ObjectDictBase = Dict[str, Any]


class ObjectDict(_ObjectDictBase):
    """
    Make a dictionary behave like an object, with attribute-style access.

    Here are some examples of how it can be used:

    o = ObjectDict(my_dict)
    # or like this:
    o = ObjectDict(samples=samples, sample_rate=sample_rate)

    # Attribute-style access
    samples = o.samples

    # Dict-style access
    samples = o["samples"]
    """

    def __getattr__(self, name):
        # type: (str) -> Any
        try:
            return self[name]
        except KeyError:
            raise AttributeError(name)

    def __setattr__(self, name, value):
        # type: (str, Any) -> None
        self[name] = value

    def __dir__(self):
        # for auto-completion in a REPL (e.g. Jupyter notebook)
        return list(self.keys()) + list(super().__dir__())


@dataclass
class RandomValue:
    """
    Generate a uniform distribution in the range [start, end].
    """

    end: Union[int, float]
    start: Union[int, float] = 0

    def __post_init__(self):
        assert self.start <= self.end

    def sample(
        self, shape: Union[Sequence[int], int] = [], device: str = "cpu"
    ) -> torch.Tensor:
        rad = torch.rand(shape, device=device)
        return (self.end - self.start) * rad + self.start

    def sample_int(
        self, shape: Union[Sequence[int], int] = [], device: str = "cpu"
    ) -> torch.Tensor:
        assert self.start <= self.end
        start = int(self.start)
        end = int(self.end)
        if isinstance(shape, int):
            shape = tuple([shape])
        return torch.randint(low=start, high=end + 1, size=shape, device=device)


class Timer(object):
    def __init__(self) -> None:
        super().__init__()
        self.start_time = time.time()

    def reset(self):
        self.start_time = time.time()

    def elapse(self):
        return time.time() - self.start_time


def asdict_filt(
    obj, *, dict_factory=dict, filt_type='default', init_field_only=False
) -> Dict[str, Any]:
    """
    Recursively convert a dataclass/dict object into a dict.
    To be used in place of dataclasses.asdict().

    filt_type:
        - 'default': filt default value in dataclass obj.
        - 'none': filt None value in dataclass/dict obj.
        - 'orig': original dataclasses.asdict() without filter.

    init_field_only: if True, obj of `dataclasses` only consider the fields
        with `init == True`.

    """
    support_flit = ['default', 'none', 'orig']
    if filt_type:
        assert (
            filt_type in support_flit
        ), f'Unsupport filt_type: {filt_type}. choose from {support_flit}'

    def _is_dataclass_instance(obj):
        # https://docs.python.org/3/library/dataclasses.html#dataclasses.is_dataclass
        return is_dataclass(obj) and not isinstance(obj, type)

    def _asdict_inner(obj, dict_factory):
        if _is_dataclass_instance(obj):
            result = []

            for f in fields(obj):
                if init_field_only and not f.init:
                    continue
                value = _asdict_inner(getattr(obj, f.name), dict_factory)
                append_flag = False
                if filt_type:
                    if filt_type == 'default':
                        append_flag = (
                            not f.init
                            or value != f.default
                            or f.metadata.get("include_default", False)
                        )
                    elif filt_type == 'none':
                        append_flag = value is not None
                    elif filt_type == 'orig':
                        append_flag = True
                    else:
                        raise ValueError(f'Unsupport filt_type: {filt_type}.')
                else:
                    append_flag = True

                if append_flag:
                    result.append((f.name, value))
            return dict_factory(result)
        elif isinstance(obj, tuple) and hasattr(obj, "_fields"):
            # obj is a namedtuple
            return type(obj)(*[_asdict_inner(v, dict_factory) for v in obj])
        elif isinstance(obj, (list, tuple)):
            # Assume we can create an object of this type by passing in a
            # generator (which is not true for namedtuples, handled
            # above).
            return type(obj)(_asdict_inner(v, dict_factory) for v in obj)
        elif isinstance(obj, dict):
            return type(obj)(
                (_asdict_inner(k, dict_factory), _asdict_inner(v, dict_factory))
                for k, v in obj.items()
                if (filt_type != 'none' or v is not None)
            )
        else:
            return copy.deepcopy(obj)

    if not isinstance(obj, dict) and not _is_dataclass_instance(obj):
        raise TypeError(f"{obj} is not a dict or a dataclass")

    return _asdict_inner(obj, dict_factory)


def fields_init_var(class_or_instance):
    """Return a tuple describing the `InitVar` fields of this dataclass.

    Modified from:
        https://docs.python.org/3/library/dataclasses.html#dataclasses.fields

    Accepts a dataclass or an instance of one. Tuple elements are of
    type Field.
    """
    # Might it be worth caching this, per class?
    try:
        fields = getattr(class_or_instance, dataclasses._FIELDS)
    except AttributeError:
        raise TypeError('must be called with a dataclass type or instance')
    # Exclude pseudo-fields.  Note that fields is sorted by insertion
    # order, so the order of the tuple is as the fields were defined.
    return tuple(
        f for f in fields.values() if f._field_type is dataclasses._FIELD_INITVAR
    )


def string2dict(s: str) -> dict:
    """
    Parse a string to a dict, it is useful for parsing cli args.

    comma_separated:
        - "a='test_a', b='test_b'"  -> \ndict: {a='test_a', b='test_b'}
        - "(a='test_a', (b='test_b'))"  -> \ndict: {a='test_a', b='test_b'}
        - "a='test_a', b=(b='test_b', c='test_c')" -> \ndict: {a='test_a', b={b='test_b', c='test_c'}}
    json_stye:
        -'{a:'test_a', b:'test_b'}' -> \ndict: {a='test_a', b='test_b'}

    Example
    -------

    """

    def _parse_comma_separated(s):
        ret = {}
        for mapping in s.replace(" ", "").split(","):
            key, value = mapping.split("=")
            ret[key] = value
        return ret

    return _parse_comma_separated(s)


def is_tensor(x):
    from torch import Tensor

    return isinstance(x, Tensor) or isinstance(x, np.ndarray)


def audio_collate_fn(
    waveforms: List[torch.Tensor],
) -> Tuple[torch.Tensor, torch.Tensor]:
    lengths = torch.tensor(
        [waveform.size(-1) for waveform in waveforms], dtype=torch.int32
    )

    # Move the last time dimension to first.
    waveforms = pad_sequence(
        [waveform.transpose(0, -1) for waveform in waveforms], batch_first=True
    )
    return waveforms.transpose(1, -1), lengths


def batched(iterable, n):
    "Batch data into tuples of length n. The last batch may be shorter."
    # batched('ABCDEFG', 3) --> ABC DEF G
    if n < 1:
        raise ValueError('n must be at least one')
    it = iter(iterable)
    while batch := tuple(islice(it, n)):
        yield batch


def rich_exception_info(fn):
    @functools.wraps(fn)
    def wrapper(*args, **kwargs):
        try:
            return fn(*args, **kwargs)
        except Exception as e:
            raise type(e)(
                f"{e}\n[extra info] When calling: {fn.__qualname__}(args={args} kwargs={kwargs})"
            )

    return wrapper


_uppercase_uppercase_re = re.compile(r"([A-Z]+)([A-Z][a-z])")
_lowercase_uppercase_re = re.compile(r"([a-z\d])([A-Z])")

_single_underscore_re = re.compile(r"(?<!_)_(?!_)")
_multiple_underscores_re = re.compile(r"(_{2,})")


def camelcase_to_snakecase(name):
    """Convert camel-case string to snake-case."""
    name = _uppercase_uppercase_re.sub(r"\1_\2", name)
    name = _lowercase_uppercase_re.sub(r"\1_\2", name)
    return name.lower()


def snakecase_to_camelcase(name):
    """Convert snake-case string to camel-case string."""
    name = _single_underscore_re.split(name)
    name = [_multiple_underscores_re.split(n) for n in name]
    return "".join(n.capitalize() for n in chain.from_iterable(name) if n != "")


# https://github.com/speechbrain/speechbrain/blob/develop/speechbrain/utils/data_utils.py
def batch_pad_right(tensors: list, mode="constant", value=0, val_index=-1):
    """Given a list of torch tensors it batches them together by padding to the right
    on each dimension in order to get same length for all.

    Parameters
    ----------
    tensors : list
        List of tensor we wish to pad together.
    mode : str
        Padding mode see torch.nn.functional.pad documentation.
    value : float
        Padding value see torch.nn.functional.pad documentation.

    Returns
    -------
    tensor : torch.Tensor
        Padded tensor.
    valid_vals : list
        List containing proportion for each dimension of original, non-padded values.

    """

    if not len(tensors):
        raise IndexError("Tensors list must not be empty")
    # tensors = list(map(list,tensors))

    if len(tensors) == 1:
        # if there is only one tensor in the batch we simply unsqueeze it.
        return tensors[0].unsqueeze(0), torch.tensor([1.0])

    if not (any([tensors[i].ndim == tensors[0].ndim for i in range(1, len(tensors))])):
        raise IndexError("All tensors must have same number of dimensions")

    # FIXME we limit the support here: we allow padding of only the last dimension
    # need to remove this when feat extraction is updated to handle multichannel.
    max_shape = []
    for dim in range(tensors[0].ndim):
        if dim != (tensors[0].ndim - 1):
            if not all([x.shape[dim] == tensors[0].shape[dim] for x in tensors[1:]]):
                raise EnvironmentError(
                    "Tensors should have same dimensions except for last one"
                )
        max_shape.append(max([x.shape[dim] for x in tensors]))

    batched = []
    valid = []
    for t in tensors:
        # for each tensor we apply pad_right_to
        padded, valid_percent = pad_right_to(t, max_shape, mode=mode, value=value)
        batched.append(padded)
        valid.append(valid_percent[val_index])

    batched = torch.stack(batched)

    return batched, torch.tensor(valid)


def pad_right_to(
    tensor: torch.Tensor,
    target_shape: (list, tuple),
    mode="constant",
    value=0,
):
    """
    This function takes a torch tensor of arbitrary shape and pads it to target
    shape by appending values on the right.

    Parameters
    ----------
    tensor : input torch tensor
        Input tensor whose dimension we need to pad.
    target_shape : (list, tuple)
        Target shape we want for the target tensor its len must be equal to tensor.ndim
    mode : str
        Pad mode, please refer to torch.nn.functional.pad documentation.
    value : float
        Pad value, please refer to torch.nn.functional.pad documentation.

    Returns
    -------
    tensor : torch.Tensor
        Padded tensor.
    valid_vals : list
        List containing proportion for each dimension of original, non-padded values.
    """
    assert len(target_shape) == tensor.ndim
    # this contains the abs length of the padding for each dimension.
    pads = []
    valid_vals = []  # thic contains the relative lengths for each dimension.
    i = len(target_shape) - 1  # iterating over target_shape ndims
    j = 0
    while i >= 0:
        assert (
            target_shape[i] >= tensor.shape[i]
        ), "Target shape must be >= original shape for every dim"
        pads.extend([0, target_shape[i] - tensor.shape[i]])
        valid_vals.append(tensor.shape[j] / target_shape[j])
        i -= 1
        j += 1
    tensor = torch.nn.functional.pad(tensor, pads, mode=mode, value=value)

    return tensor, valid_vals


def get_diff_dict(src_dict: Dict, curr_dict: Dict) -> Dict:
    """
    Compare two dicts, returns a dict contains the different part.
    """
    diff_dict = {}

    # search through init model opts
    for k, v in curr_dict.items():
        if k not in src_dict or v != src_dict[k]:
            diff_dict[k] = v

    return diff_dict


def add_start_docstrings(*docstr):
    def docstring_decorator(fn):
        fn.__doc__ = "".join(docstr) + (fn.__doc__ if fn.__doc__ is not None else "")
        return fn

    return docstring_decorator
