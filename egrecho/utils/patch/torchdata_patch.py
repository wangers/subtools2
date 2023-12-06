# -*- coding:utf-8 -*-
# Copyright xmuspeech (Author: Leo 2023-03)

import functools
import inspect
import io
from io import IOBase
from typing import (
    Any,
    Callable,
    Dict,
    Generic,
    Iterator,
    Optional,
    Tuple,
    TypeVar,
    Union,
)

T = TypeVar("T")


def validate_input_col(fn: Callable, input_col: Optional[Union[int, tuple, list]]):
    """
    Copid from `torch.utils.data.datapipes.utils.common`.

    Checks that function used in a callable datapipe works with the input column

    This simply ensures that the number of positional arguments matches the size
    of the input column. The function must not contain any non-default
    keyword-only arguments.

    Examples:
        >>> # xdoctest: +SKIP("Failing on some CI machines")
        >>> def f(a, b, *, c=1):
        >>>     return a + b + c
        >>> def f_def(a, b=1, *, c=1):
        >>>     return a + b + c
        >>> validate_input_col(f, [1, 2])
        >>> validate_input_col(f_def, 1)
        >>> validate_input_col(f_def, [1, 2])

    Notes:
        If the function contains variable positional (`inspect.VAR_POSITIONAL`) arguments,
        for example, f(a, *args), the validator will accept any size of input column
        greater than or equal to the number of positional arguments.
        (in this case, 1).

    Args:
        fn: The function to check.
        input_col: The input column to check.

    Raises:
        ValueError: If the function is not compatible with the input column.
    """
    try:
        sig = inspect.signature(fn)
    except (
        ValueError
    ):  # Signature cannot be inspected, likely it is a built-in fn or written in C
        return
    if isinstance(input_col, (list, tuple)):
        input_col_size = len(input_col)
    else:
        input_col_size = 1

    pos = []
    var_positional = False
    non_default_kw_only = []

    for p in sig.parameters.values():
        if p.kind in (
            inspect.Parameter.POSITIONAL_ONLY,
            inspect.Parameter.POSITIONAL_OR_KEYWORD,
        ):
            pos.append(p)
        elif p.kind is inspect.Parameter.VAR_POSITIONAL:
            var_positional = True
        elif p.kind is inspect.Parameter.KEYWORD_ONLY:
            if p.default is p.empty:
                non_default_kw_only.append(p)
        else:
            continue

    if isinstance(fn, functools.partial):
        fn_name = getattr(fn.func, "__name__", repr(fn.func))
    else:
        fn_name = getattr(fn, "__name__", repr(fn))

    if len(non_default_kw_only) > 0:
        raise ValueError(
            f"The function {fn_name} takes {len(non_default_kw_only)} "
            f"non-default keyword-only parameters, which is not allowed."
        )

    if len(sig.parameters) < input_col_size:
        if not var_positional:
            raise ValueError(
                f"The function {fn_name} takes {len(sig.parameters)} "
                f"parameters, but {input_col_size} are required."
            )
    else:
        if len(pos) > input_col_size:
            if any(p.default is p.empty for p in pos[input_col_size:]):
                raise ValueError(
                    f"The function {fn_name} takes {len(pos)} "
                    f"positional parameters, but {input_col_size} are required."
                )
        elif len(pos) < input_col_size:
            if not var_positional:
                raise ValueError(
                    f"The function {fn_name} takes {len(pos)} "
                    f"positional parameters, but {input_col_size} are required."
                )


class StreamWrapper:
    """
    StreamWrapper is introduced to wrap file handler generated by
    DataPipe operation like `FileOpener`. StreamWrapper would guarantee
    the wrapped file handler is closed when it's out of scope.
    """

    session_streams: Dict[Any, int] = {}
    debug_unclosed_streams: bool = False

    def __init__(self, file_obj, parent_stream=None, name=None):
        self.file_obj = file_obj
        self.child_counter = 0
        self.parent_stream = parent_stream
        self.close_on_last_child = False
        self.name = name
        self.closed = False
        if parent_stream is not None:
            if not isinstance(parent_stream, StreamWrapper):
                raise RuntimeError(
                    "Parent stream should be StreamWrapper, {} was given".format(
                        type(parent_stream)
                    )
                )
            parent_stream.child_counter += 1
            self.parent_stream = parent_stream
        if StreamWrapper.debug_unclosed_streams:
            StreamWrapper.session_streams[self] = 1

    @classmethod
    def close_streams(cls, v, depth=0):
        """
        Traverse structure and attempts to close all found StreamWrappers on best effort basis.
        """
        if depth > 10:
            return
        if isinstance(v, StreamWrapper):
            v.close()
        else:
            # Traverse only simple structures
            if isinstance(v, dict):
                for kk, vv in v.items():
                    cls.close_streams(vv, depth=depth + 1)
            elif isinstance(v, (list, tuple)):
                for vv in v:
                    cls.close_streams(vv, depth=depth + 1)

    def __getattr__(self, name):
        file_obj = self.__dict__["file_obj"]
        return getattr(file_obj, name)

    def close(self, *args, **kwargs):
        if self.closed:
            return
        if StreamWrapper.debug_unclosed_streams:
            del StreamWrapper.session_streams[self]
        if hasattr(self, "parent_stream") and self.parent_stream is not None:
            self.parent_stream.child_counter -= 1
            if (
                not self.parent_stream.child_counter
                and self.parent_stream.close_on_last_child
            ):
                self.parent_stream.close()
        try:
            self.file_obj.close(*args, **kwargs)
        except AttributeError:
            pass
        self.closed = True

    def autoclose(self):
        """
        Close steam if there is no children, or make it to be automatically closed as soon as
        all child streams are closed.
        """
        self.close_on_last_child = True
        if self.child_counter == 0:
            self.close()

    def __dir__(self):
        attrs = list(self.__dict__.keys()) + list(StreamWrapper.__dict__.keys())
        attrs += dir(self.file_obj)
        return list(set(attrs))

    def __del__(self):
        if not self.closed:
            self.close()

    def __iter__(self):
        yield from self.file_obj

    def __next__(self):
        return next(self.file_obj)

    def __repr__(self):
        if self.name is None:
            return f"StreamWrapper<{self.file_obj!r}>"
        else:
            return f"StreamWrapper<{self.name},{self.file_obj!r}>"

    def __getstate__(self):
        return self.file_obj

    def __setstate__(self, obj):
        self.file_obj = obj


def is_stream_handle(data):
    obj_to_check = data.file_obj if isinstance(data, StreamWrapper) else data
    return isinstance(obj_to_check, (io.BufferedIOBase, io.RawIOBase))


def validate_pathname_binary_tuple(data: Tuple[str, IOBase]):
    if not isinstance(data, tuple):
        raise TypeError(
            f"pathname binary data should be tuple type, but it is type {type(data)}"
        )
    if len(data) != 2:
        raise TypeError(
            f"pathname binary stream tuple length should be 2, but got {len(data)}"
        )
    if not isinstance(data[0], str):
        raise TypeError(
            f"pathname within the tuple should have string type pathname, but it is type {type(data[0])}"
        )
    if not isinstance(data[1], IOBase) and not isinstance(data[1], StreamWrapper):
        raise TypeError(
            f"binary stream within the tuple should have IOBase or"
            f"its subclasses as type, but it is type {type(data[1])}"
        )


class DataChunk(list, Generic[T]):
    def __init__(self, items):
        super().__init__(items)
        self.items = items

    def as_str(self, indent=""):
        res = indent + "[" + ", ".join(str(i) for i in iter(self)) + "]"
        return res

    def __iter__(self) -> Iterator[T]:
        yield from super().__iter__()

    def raw_iterator(self) -> T:  # type: ignore[misc]
        yield from self.items
