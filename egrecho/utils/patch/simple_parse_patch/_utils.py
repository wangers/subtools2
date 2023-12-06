"""Utility functions used in various parts of the simple_parsing package."""
from __future__ import annotations

import argparse
import dataclasses
import enum
import inspect
import sys
import types
import typing
from collections import abc as c_abc
from dataclasses import Field
from enum import Enum
from logging import getLogger
from typing import Any, ClassVar, ForwardRef, Mapping, TypeVar, Union

from typing_extensions import Literal, Protocol, TypeGuard, get_args, get_origin

# There are cases where typing.Literal doesn't match typing_extensions.Literal:
# https://github.com/python/typing_extensions/pull/148
try:
    from typing import Literal as LiteralAlt
except ImportError:
    LiteralAlt = Literal  # type: ignore


# NOTE: Copied from typing_inspect.
def is_typevar(t) -> bool:
    return type(t) is TypeVar


def get_bound(t):
    if is_typevar(t):
        return getattr(t, "__bound__", None)
    else:
        raise TypeError(f"type is not a `TypeVar`: {t}")


def is_forward_ref(t) -> TypeGuard[typing.ForwardRef]:
    return isinstance(t, typing.ForwardRef)


def get_forward_arg(fr: ForwardRef) -> str:
    return getattr(fr, "__forward_arg__")


logger = getLogger(__name__)


T = TypeVar("T")


class Dataclass(Protocol):
    __dataclass_fields__: ClassVar[dict[str, Field]]


def is_dataclass_instance(obj: Any) -> TypeGuard[Dataclass]:
    return dataclasses.is_dataclass(obj) and dataclasses.is_dataclass(type(obj))


def is_dataclass_type(obj: Any) -> TypeGuard[type[Dataclass]]:
    return inspect.isclass(obj) and dataclasses.is_dataclass(obj)


DataclassT = TypeVar("DataclassT", bound=Dataclass)


TRUE_STRINGS: list[str] = ["yes", "true", "t", "y", "1"]
FALSE_STRINGS: list[str] = ["no", "false", "f", "n", "0"]


def str2bool(raw_value: str | bool) -> bool:
    """
    Taken from https://stackoverflow.com/questions/15008758/parsing-boolean-values-with-argparse
    """
    if isinstance(raw_value, bool):
        return raw_value
    v = raw_value.strip().lower()
    if v in TRUE_STRINGS:
        return True
    elif v in FALSE_STRINGS:
        return False
    else:
        raise argparse.ArgumentTypeError(
            f"Boolean value expected for argument, received '{raw_value}'"
        )


def _mro(t: type) -> list[type]:
    # TODO: This is mostly used in 'is_tuple' and such, and should be replaced with
    # either the built-in 'get_origin' from typing, or from typing-inspect.
    if t is None:
        return []
    if hasattr(t, "__mro__"):
        return t.__mro__
    elif get_origin(t) is type:
        return []
    elif hasattr(t, "mro") and callable(t.mro):
        return t.mro()
    return []


def is_literal(t: type) -> bool:
    """Returns True with `t` is a Literal type.

    >>> from typing_extensions import Literal
    >>> from typing import *
    >>> is_literal(list)
    False
    >>> is_literal("foo")
    False
    >>> is_literal(Literal[True, False])
    True
    >>> is_literal(Literal[1,2,3])
    True
    >>> is_literal(Literal["foo", "bar"])
    True
    >>> is_literal(Optional[Literal[1,2]])
    False
    """
    return get_origin(t) in (Literal, LiteralAlt)


def is_list(t: type) -> bool:
    """returns True when `t` is a List type.

    Args:
        t (Type): a type.

    Returns:
        bool: True if `t` is list or a subclass of list.

    >>> from typing import *
    >>> is_list(list)
    True
    >>> is_list(tuple)
    False
    >>> is_list(List)
    True
    >>> is_list(List[int])
    True
    >>> is_list(List[Tuple[int, str, None]])
    True
    >>> is_list(Optional[List[int]])
    False
    >>> class foo(List[int]):
    ...   pass
    ...
    >>> is_list(foo)
    True
    """
    return list in _mro(t)


def is_tuple(t: type) -> bool:
    """returns True when `t` is a tuple type.

    Args:
        t (Type): a type.

    Returns:
        bool: True if `t` is tuple or a subclass of tuple.

    >>> from typing import *
    >>> is_tuple(list)
    False
    >>> is_tuple(tuple)
    True
    >>> is_tuple(Tuple)
    True
    >>> is_tuple(Tuple[int])
    True
    >>> is_tuple(Tuple[int, str, None])
    True
    >>> class foo(tuple):
    ...   pass
    ...
    >>> is_tuple(foo)
    True
    >>> is_tuple(List[int])
    False
    """
    return tuple in _mro(t)


def is_dict(t: type) -> bool:
    """returns True when `t` is a dict type or annotation.

    Args:
        t (Type): a type.

    Returns:
        bool: True if `t` is dict or a subclass of dict.

    >>> from typing import *
    >>> from collections import OrderedDict
    >>> is_dict(dict)
    True
    >>> is_dict(OrderedDict)
    True
    >>> is_dict(tuple)
    False
    >>> is_dict(Dict)
    True
    >>> is_dict(Dict[int, float])
    True
    >>> is_dict(Dict[Any, Dict])
    True
    >>> is_dict(Optional[Dict])
    False
    >>> is_dict(Mapping[str, int])
    True
    >>> class foo(Dict):
    ...   pass
    ...
    >>> is_dict(foo)
    True
    """
    mro = _mro(t)
    return dict in mro or Mapping in mro or c_abc.Mapping in mro


def is_set(t: type) -> bool:
    """returns True when `t` is a set type or annotation.

    Args:
        t (Type): a type.

    Returns:
        bool: True if `t` is set or a subclass of set.

    >>> from typing import *
    >>> is_set(set)
    True
    >>> is_set(Set)
    True
    >>> is_set(tuple)
    False
    >>> is_set(Dict)
    False
    >>> is_set(Set[int])
    True
    >>> is_set(Set["something"])
    True
    >>> is_set(Optional[Set])
    False
    >>> class foo(Set):
    ...   pass
    ...
    >>> is_set(foo)
    True
    """
    return set in _mro(t)


def is_enum(t: type) -> bool:
    if inspect.isclass(t):
        return issubclass(t, enum.Enum)
    return Enum in _mro(t)


def is_union(t: type) -> bool:
    """Returns whether or not the given Type annotation is a variant (or subclass) of typing.Union

    Args:
        t (Type): some type annotation

    Returns:
        bool: Whether this type represents a Union type.

    >>> from typing import *
    >>> is_union(Union[int, str])
    True
    >>> is_union(Union[int, str, float])
    True
    >>> is_union(Tuple[int, str])
    False
    """
    if sys.version_info[:2] >= (3, 10) and isinstance(t, types.UnionType):
        return True
    return getattr(t, "__origin__", "") == Union


def get_type_arguments(container_type: type) -> tuple[type, ...]:
    # return getattr(container_type, "__args__", ())
    return get_args(container_type)


def all_subclasses(t: type[T]) -> set[type[T]]:
    immediate_subclasses = t.__subclasses__()
    return set(immediate_subclasses).union(
        *[all_subclasses(s) for s in immediate_subclasses]
    )


if __name__ == "__main__":
    import doctest

    doctest.testmod()
