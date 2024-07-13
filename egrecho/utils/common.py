# -*- coding:utf-8 -*-
# Copyright xmuspeech (Author: Leo 2023-03)

from __future__ import annotations

import dataclasses
import functools
import re
import time
from collections import Counter, OrderedDict
from copy import deepcopy
from dataclasses import Field
from itertools import chain, islice
from typing import Any, ClassVar, Dict, Iterable, Literal, Optional, TypeVar, Union

from egrecho.utils.apply import apply_to_collection
from egrecho.utils.imports import _HYDRA_AVAILABLE, _OMEGACONF_AVAILABLE
from egrecho.utils.logging import get_logger
from egrecho.utils.patch import asdict_filt, from_dict, register_decoding_fn
from egrecho.utils.types import FilterType

if _HYDRA_AVAILABLE:
    import hydra

import traceback

logger = get_logger(__name__)

D = TypeVar("D", bound="DataclassSerialMixin")
G = TypeVar("G", bound="GenericSerialMixin")
T = TypeVar("T")
K = TypeVar("K")
V = TypeVar("V")


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
        val: value to be judged.
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


class ObjectDict(dict):
    """
    Makes a dictionary behave like an object, with attribute-style access.

    Here are some examples of how it can be used::

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


class Timer(object):
    def __init__(self) -> None:
        super().__init__()
        self.start_time = time.time()

    def reset(self):
        self.start_time = time.time()

    def elapse(self):
        return time.time() - self.start_time


class DataclassSerialMixin:
    """From/to dict mixin of dataclass.

    Example:
        >>> from dataclasses import dataclass
        >>> @dataclass
        ... class Config(DataclassSerialMixin):
        ...   a: int = 123
        ...   b: str = "456"
        ...
        >>> config = Config(filt_type='orig')
        >>> config
        Config(a=123, b='456')
        >>> config.to_dict()
        {'a': 123, 'b': '456'}
        >>> config_ = Config.from_dict({"a": 123, "b": 456})
        >>> config_
        Config(a=123, b='456')
        >>> assert config == config_
    """

    subclasses: ClassVar[list] = []
    decode_into_subclasses: ClassVar[bool] = False
    filt_type: ClassVar[Union[str, FilterType]] = FilterType.NONE

    def __init_subclass__(cls, decode_into_subclasses: Optional[bool] = None):
        logger.debug(f"Registering a new Serializable subclass: {cls}")
        super().__init_subclass__()
        if decode_into_subclasses is None:
            # if decode_into_subclasses is None, we will use the value of the
            # parent class, if it is also a subclass of Serializable.
            # Skip the class itself as well as object.
            parents = cls.mro()[1:-1]
            logger.debug(f"parents: {parents}")

            for parent in parents:
                if (
                    parent in DataclassSerialMixin.subclasses
                    and parent is not DataclassSerialMixin
                ):
                    decode_into_subclasses = parent.decode_into_subclasses
                    logger.debug(
                        f"Parent class {parent} has decode_into_subclasses = {decode_into_subclasses}"
                    )
                    break

        cls.decode_into_subclasses = decode_into_subclasses or False
        if cls not in DataclassSerialMixin.subclasses:
            DataclassSerialMixin.subclasses.append(cls)

        register_decoding_fn(cls, cls.from_dict)

    def to_dict(
        self,
        dict_factory=dict,
        filt_type: Optional[Literal["default", "none", "orig"]] = None,
        init_field_only=True,
        save_dc_types=False,
    ) -> dict:
        """
        Serializes this dataclass to a dict.
        """
        filt_type = filt_type or self.filt_type
        return asdict_filt(
            self,
            dict_factory=dict_factory,
            filt_type=filt_type,
            init_field_only=init_field_only,
            save_dc_types=save_dc_types,
        )

    @classmethod
    def from_dict(
        cls: type[D], obj: dict, drop_extra_fields: Optional[bool] = None
    ) -> D:
        """Parses an instance of ``cls`` from the given dict.

        NOTE: If the ``decode_into_subclasses`` class attribute is set to True (or
        if ``decode_into_subclasses=True`` was passed in the class definition),
        then if there are keys in the dict that aren't fields of the dataclass,
        this will decode the dict into an instance the first subclass of `cls`
        which has all required field names present in the dictionary.

        - Passing ``drop_extra_fields=None`` (default) will use the class attribute
            described above.
        - Passing ``drop_extra_fields=True`` will decode the dict into an instance
            of ``cls`` and drop the extra keys in the dict.
            Passing ``drop_extra_fields=False`` forces the above-mentioned behaviour.
        """
        if _OMEGACONF_AVAILABLE:
            obj = omegaconf2container(obj)
        return from_dict(cls, obj, drop_extra_fields=drop_extra_fields)


class GenericSerialMixin:
    """Serialization mixin of common class with config attribute."""

    CONFIG_CLS = None

    def to_dict(self, **kwargs) -> dict:
        """Returns object's configuration to config dictionary"""

        if hasattr(self, "config") and self.config is not None:
            if (to_dict := getattr(self.config, "to_dict", None)) and callable(to_dict):
                return self.config.to_dict(**kwargs)
            if _OMEGACONF_AVAILABLE:
                config = omegaconf2container(config)
            filt_type = kwargs.pop('filt_type', FilterType.NONE)
            init_field_only = kwargs.pop('init_field_only', True)
            return asdict_filt(
                config, filt_type=filt_type, init_field_only=init_field_only, **kwargs
            )
        else:
            raise NotImplementedError(
                "to_dict() can currently only return object.config but current object does not have it."
            )

    @classmethod
    def from_dict(cls: type[G], config) -> G:
        """Instantiates object using DictConfig-based configuration"""

        if _OMEGACONF_AVAILABLE:
            # Resolves interpolations of DictConfig
            config = omegaconf2container(config)

        out_desc = []
        imported_cls = None
        if target_cls := config.get('_target_', None):
            from egrecho.core.loads import load_module_class

            try:
                # try to import the target class
                imported_cls = load_module_class(target_cls, cls)
            except Exception as e:
                tb = traceback.format_exc()
                prev_error = (
                    f"Import target class failed!\nTarget class:\t{target_cls}"
                    f"\nError(s):\t{e}\n{tb}"
                )
                out_desc.append(prev_error)
                logger.debug(
                    prev_error + f"\nFalling back to use curent `cls` =>{cls}."
                )

        cur_cls = imported_cls or cls

        # shortpath
        if (
            (cfg_cls := cur_cls.CONFIG_CLS) is not None
            and hasattr(cfg_cls, "from_dict")
            and callable(cfg_cls.from_dict)
        ):
            config.pop('_target_', None)
            config = cfg_cls.from_dict(config)
            return cur_cls(config=config)

        # Hydra 1.x API
        if imported_cls and _HYDRA_AVAILABLE:
            # hydra-based instantiation
            try:
                instance = hydra.utils.instantiate(config=config)
            except Exception as e:
                tb = traceback.format_exc()
                prev_error = (
                    f"Hydra model instantiation failed!\nTarget class:\t{target_cls}"
                    f"\nError(s):\t{e}\n{tb}"
                )
                out_desc.append(prev_error)
                logger.debug(
                    prev_error + f"\nFalling back to cur_cls(config) =>{cur_cls}."
                )
        else:
            try:
                # target class resolution was unsuccessful, fall back to current `cls`
                instance = cur_cls(config=config)
            except Exception as e:
                # report saved errors, if any, and raise
                if out_desc:
                    logger.error('\n'.join(out_desc))
                raise e

        return instance


def fields_init_var(class_or_instance):
    """Return a tuple describing the ``InitVar`` fields of this dataclass.

    Modified from:
    https://docs.python.org/3/library/dataclasses.html#dataclasses.fields

    Accepts a dataclass or an instance of one. Tuple elements are of
    type Field.
    """
    # Might it be worth caching this, per class?
    try:
        all_fields = getattr(class_or_instance, dataclasses._FIELDS)
    except AttributeError:
        raise TypeError("must be called with a dataclass type or instance")
    # Exclude pseudo-fields.  Note that fields is sorted by insertion
    # order, so the order of the tuple is as the fields were defined.
    return tuple(
        f for f in all_fields.values() if f._field_type is dataclasses._FIELD_INITVAR
    )


def field_dict(dataclass, init_field_only=True) -> Dict[str, Field]:
    result: dict[str, Field] = OrderedDict()

    for field in dataclasses.fields(dataclass):
        if init_field_only and not field.init:
            continue
        result[field.name] = field
    return result


def batched(iterable, n):
    "Batch data into tuples of length n. The last batch may be shorter."
    # batched('ABCDEFG', 3) --> ABC DEF G
    if n < 1:
        raise ValueError("n must be at least one")
    it = iter(iterable)
    while batch := tuple(islice(it, n)):
        yield batch


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


def get_diff_dict(curr_dict: Dict, src_dict: Dict) -> Dict:
    """
    Compare two dictionaries and return a new dictionary containing the differing key-value pairs.

    Args:
        curr_dict (Dict): The current dictionary to compare.
        src_dict (Dict): The source dictionary to compare against.

    Returns:
        Dict: A dictionary containing the key-value pairs that differ between ``src_dict`` and ``curr_dict``.

    Example:
        >>> src_dict = {"name": "John", "age": 30}
        >>> curr_dict = {"name": "John", "age": 35, "city": "New York"}
        >>> diff_dict = get_diff_dict(curr_dict, src_dict)
        {'age': 35, 'city': 'New York'}

    """
    diff_dict = {}

    for k, v in curr_dict.items():
        if k not in src_dict or v != src_dict[k]:
            diff_dict[k] = v
    return diff_dict


def del_default_dict(data: Dict, defaults: Dict, recurse: bool = False):
    """
    Removes key-value pairs from a dictionary that match default values.

    Args:
        data (Dict): The dictionary to remove default values from.
        defaults (Dict): The dictionary containing default values to compare against.
        recurse: recursively processes. Defaults to False.

    Returns:
        Dict: A dictionary with default values removed.

    Example:
        >>> data = {"name": "John", "age": 30, "address": {"city": "New York", "zip": 10001}}
        >>> defaults = {"name": "John", "age": 30, "address": {"city": "Unknown", "zip": None}}
        >>> cleaned_data = del_default_dict(data, defaults)
        >>> cleaned_data
        {'address': {'city': 'New York', 'zip': 10001}}

    NOTE:
        This function modifies the `data` dictionary in place and also returns it. If
        ``recurse=True`` it recursively processes nested dictionaries.
    """
    for k in list(data.keys()):
        default_val = defaults.get(k)
        val = data[k]
        if default_val and default_val == val:
            del data[k]
        elif isinstance(val, dict) and isinstance(default_val, dict) and recurse:
            del_default_dict(val, default_val)
    return data


def dict_union(
    *dicts: dict,
    recurse: bool = True,
    sort_keys=False,
    dict_factory=dict,
) -> dict:
    """
    Combine multiple dictionaries into the first one.

    Args:
        *dicts (dict): One or more dictionaries to combine.
        recurse (bool, optional): If True, also recursively combine nested dictionaries.
        sort_keys (bool, optional): If True, sort the keys alphabetically in the resulting dictionary.
        dict_factory (callable, optional): A callable that creates the output dictionary (default is `dict`).

    Returns:
        dict: A new dictionary containing the union of all input dictionaries.

    Example:
        >>> from collections import OrderedDict
        >>> a = OrderedDict(a=1, b=2, c=3)
        >>> b = OrderedDict(c=5, d=6, e=7)
        >>> dict_union(a, b, dict_factory=OrderedDict)
        OrderedDict([('a', 1), ('b', 2), ('c', 5), ('d', 6), ('e', 7)])
        >>> a = OrderedDict(a=1, b=OrderedDict(c=2, d=3))
        >>> b = OrderedDict(a=2, b=OrderedDict(c=3, e=6))
        >>> dict_union(a, b, dict_factory=OrderedDict)
        OrderedDict([('a', 2), ('b', OrderedDict([('c', 3), ('d', 3), ('e', 6)]))])
    """
    result: dict = dict_factory()
    if not dicts:
        return result
    assert len(dicts) >= 1
    if sort_keys:
        all_keys: set[str] = set()
        all_keys.update(*dicts)
        all_keys = sorted(all_keys)
    else:
        counter = Counter(chain(*dicts))
        all_keys = list(counter.keys())
    # Create a neat generator of generators, to save some memory.
    all_values: Iterable[tuple[V, Iterable[K]]] = (
        (k, (d[k] for d in dicts if k in d)) for k in all_keys
    )
    for k, values in all_values:
        sub_dicts: list[dict] = []
        new_value: V = None
        n_values = 0
        for v in values:
            if isinstance(v, dict) and recurse:
                sub_dicts.append(v)
            else:
                # Overwrite the new value for that key.
                new_value = v
            n_values += 1

        if len(sub_dicts) == n_values and recurse:
            # We only get here if all values for key `k` were dictionaries,
            # and if recurse was True.
            new_value = dict_union(*sub_dicts, recurse=True, dict_factory=dict_factory)

        result[k] = new_value
    return result


def list2tuple(func):
    """
    Transfer the list in input parameter to hashable tuple, it is useful for ``lru_cache``.

    NOTE: Don't support nest structure.

    Example:
        >>> def get_input(*args, **kwargs):
        ...     return args, kwargs

        >>> @list2tuple
        >>> def get_input_wrapper(*args, **kwargs):
        ...     return args, kwargs
        >>> arg_input = ([1,2,3], 'foo')
        >>> kwargs_input = {'bar': True, 'action': ["go", "leave", "break"], 'action1':['buy', ['sell', 'argue']]}
        >>> get_input(*arg_input, **kwargs_input)
        (([1, 2, 3], 'foo'),
        {'bar': True,
        'action': ['go', 'leave', 'break'],
        'action1': ['buy', ['sell', 'argue']]})
        >>> get_input_wrapper(*arg_input, **kwargs_input)
        (((1, 2, 3), 'foo'),
        {'bar': True,
        'action': ('go', 'leave', 'break'),
        'action1': ('buy', ['sell', 'argue'])})
    """

    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        args = list(args)
        for idx, arg in enumerate(args):
            if isinstance(arg, list):
                args[idx] = tuple(arg)
        for key, arg in kwargs.items():
            if isinstance(arg, list):
                kwargs[key] = tuple(arg)
        return func(*args, **kwargs)

    return wrapper


def omegaconf2container(config):
    from omegaconf import OmegaConf
    from omegaconf.dictconfig import DictConfig

    config = deepcopy(config)
    # Resolves interpolations of DictConfig
    config = apply_to_collection(
        config, DictConfig, OmegaConf.to_container, resolve=True, enum_to_str=True
    )
    return config
