# -*- coding:utf-8 -*-
# Copyright xmuspeech (Author: Leo 2023-08)


import warnings
from collections import OrderedDict
from dataclasses import fields
from enum import Enum
from typing import Any, List, Literal, Optional, Tuple

TRAIN_DATALOADERS = Any  # any iterable or collection of iterables
EVAL_DATALOADERS = Any  # any iterable or collection of iterables


class ModelOutput(OrderedDict):
    """
    Base class for all model outputs as dataclass. Copied from `huggingface modelout
    <https://github.com/huggingface/transformers/blob/main/src/transformers/utils/generic.py#ModelOutput>`_.

    Has a :meth:`__getitem__` that allows indexing by integer or slice (like a
    tuple) or strings (like a dictionary) that will ignore the `None` attributes. Otherwise behaves like a regular
    python dictionary.

    .. tip::

        You can't unpack a `ModelOutput` directly. Use the [`ModelOutput.to_tuple`] method to convert it to a tuple
        before.
    """

    def __post_init__(self):
        class_fields = fields(self)

        # Safety and consistency checks
        if not len(class_fields):
            raise ValueError(f"{self.__class__.__name__} has no fields.")
        if not all(field.default is None for field in class_fields[1:]):
            raise ValueError(
                f"{self.__class__.__name__} should not have more than one required field."
            )

        first_field = getattr(self, class_fields[0].name)
        other_fields_are_none = all(
            getattr(self, field.name) is None for field in class_fields[1:]
        )

        if other_fields_are_none and not is_tensor(first_field):
            if isinstance(first_field, dict):
                iterator = first_field.items()
                first_field_iterator = True
            else:
                try:
                    iterator = iter(first_field)
                    first_field_iterator = True
                except TypeError:
                    first_field_iterator = False

            # if we provided an iterator as first field and the iterator is a (key, value) iterator
            # set the associated fields
            if first_field_iterator:
                for idx, element in enumerate(iterator):
                    if (
                        not isinstance(element, (list, tuple))
                        or not len(element) == 2
                        or not isinstance(element[0], str)
                    ):
                        if idx == 0:
                            # If we do not have an iterator of key/values, set it as attribute
                            self[class_fields[0].name] = first_field
                        else:
                            # If we have a mixed iterator, raise an error
                            raise ValueError(
                                f"Cannot set key/value for {element}. It needs to be a tuple (key, value)."
                            )
                        break
                    setattr(self, element[0], element[1])
                    if element[1] is not None:
                        self[element[0]] = element[1]
            elif first_field is not None:
                self[class_fields[0].name] = first_field
        else:
            for field in class_fields:
                v = getattr(self, field.name)
                if v is not None:
                    self[field.name] = v

    def __delitem__(self, *args, **kwargs):
        raise Exception(
            f"You cannot use ``__delitem__`` on a {self.__class__.__name__} instance."
        )

    def setdefault(self, *args, **kwargs):
        """remove this action"""
        raise Exception(
            f"You cannot use ``setdefault`` on a {self.__class__.__name__} instance."
        )

    def pop(self, *args, **kwargs):
        """remove this action"""
        raise Exception(
            f"You cannot use ``pop`` on a {self.__class__.__name__} instance."
        )

    def update(self, *args, **kwargs):
        """remove this action"""
        raise Exception(
            f"You cannot use ``update`` on a {self.__class__.__name__} instance."
        )

    def __getitem__(self, k):
        if isinstance(k, str):
            inner_dict = {k: v for (k, v) in self.items()}
            return inner_dict[k]
        else:
            return self.to_tuple()[k]

    def __setattr__(self, name, value):
        if name in self.keys() and value is not None:
            # Don't call self.__setitem__ to avoid recursion errors
            super().__setitem__(name, value)
        super().__setattr__(name, value)

    def __setitem__(self, key, value):
        # Will raise a KeyException if needed
        super().__setitem__(key, value)
        # Don't call self.__setattr__ to avoid recursion errors
        super().__setattr__(key, value)

    def to_tuple(self) -> Tuple[Any]:
        """
        Convert self to a tuple containing all the attributes/keys that are not ``None``.
        """
        return tuple(self[k] for k in self.keys())


class PipeOutput(ModelOutput):
    ...


class SingletonMeta(type):
    """
    A metaclass for creating singleton classes.
    """

    _instances = {}

    def __call__(cls, *args, **kwargs):
        # If an instance of this class already exists, return it
        if cls in cls._instances:
            assert (
                len(args) == 0 and len(kwargs) == 0
            ), f"{cls.__name__} is a singleton class and a instance has been created."
            return cls._instances[cls]
        instance = super(SingletonMeta, cls).__call__(*args, **kwargs)
        cls._instances[cls] = instance
        return instance


# https://github.com/Lightning-AI/utilities/blob/main/src/lightning_utilities/core/enums.py
class StrEnum(str, Enum):
    """Type of any enumerator with allowed comparison to string invariant to cases.

    >>> class MySE(StrEnum):
    ...     t1 = "T-1"
    ...     t2 = "T-2"
    >>> MySE("T-1") == MySE.t1
    True
    >>> MySE.from_str("t-2", source="value") == MySE.t2
    True
    >>> MySE.from_str("t-2", source="value")
    <MySE.t2: 'T-2'>
    >>> MySE.from_str("t-3", source="any")
    Traceback (most recent call last):
      ...
    ValueError: Invalid match: expected one of ['t1', 't2', 'T-1', 'T-2'], but got t-3.
    """

    @staticmethod
    def _generate_next_value_(name, start, count, last_values):
        """
        Return the lower-cased version of the member name.
        """
        return name.lower()

    @classmethod
    def from_str(
        cls, value: str, source: Literal["key", "value", "any"] = "key"
    ) -> "StrEnum":
        """Create ``StrEnum`` from a string matching the key or value.

        Args:
            value: matching string
            source: compare with:

                - ``"key"``: validates only from the enum keys, typical alphanumeric with "_"
                - ``"value"``: validates only from the values, could be any string
                - ``"any"``: validates with any key or value, but key has priority

        Raises:
            ValueError:
                if requested string does not match any option based on selected source.
        """
        if source in ("key", "any"):
            for enum_key in cls.__members__:
                if enum_key.lower() == value.lower():
                    return cls[enum_key]
        if source in ("value", "any"):
            for enum_key, enum_val in cls.__members__.items():
                if enum_val == value:
                    return cls[enum_key]
        raise ValueError(
            f"Invalid match: expected one of {cls._allowed_matches(source)}, but got {value}."
        )

    @classmethod
    def try_from_str(
        cls, value: str, source: Literal["key", "value", "any"] = "key"
    ) -> Optional["StrEnum"]:
        """Try to create emun and if it does not match any, return `None`."""
        try:
            return cls.from_str(value, source)
        except ValueError:
            warnings.warn(  # noqa: B028
                UserWarning(
                    f"Invalid string: expected one of {cls._allowed_matches(source)}, but got {value}."
                )
            )
        return None

    @classmethod
    def make_from_keys(cls_name: str, s_list: List[str]):
        if not isinstance(s_list, (tuple, list)) and any(
            not isinstance(s, str) for s in s_list
        ):
            raise ValueError(f"Expect a list of string to make enum, but got {s_list}")
        if len(set(s_list)) < len(s_list):
            raise ValueError(
                f"Expect a list of string which has unique value to make enum, but got {s_list}"
            )
        return StrEnum(cls_name, s_list)

    @classmethod
    def _allowed_matches(cls, source: str) -> List[str]:
        keys, vals = [], []
        for enum_key, enum_val in cls.__members__.items():
            keys.append(enum_key)
            vals.append(enum_val.value)
        if source == "key":
            return keys
        if source == "value":
            return vals
        return keys + vals

    def __eq__(self, other: object) -> bool:
        """Compare two instances."""
        if isinstance(other, Enum):
            other = other.value
        return self.value.lower() == str(other).lower()

    def __hash__(self) -> int:
        """Return unique hash."""
        # re-enable hashtable, so it can be used as a dict key or in a set
        # example: set(LightningEnum)
        return hash(self.value.lower())

    def __str__(self) -> str:
        return self.value


class Split(StrEnum):
    """
    Contains Enums of split.
    """

    TRAIN = "train"
    VALIDATION = "val"
    TEST = "test"
    ALL = "all"

    @staticmethod
    def names():
        return list(Split._member_map_.values())


def is_tensor(x):
    try:
        import numpy as np
        from torch import Tensor

        return isinstance(x, Tensor) or isinstance(x, np.ndarray)
    except ImportError:
        return False
