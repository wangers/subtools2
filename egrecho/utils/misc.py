# -*- coding:utf-8 -*-
# Copyright xmuspeech (Author: Leo 2023-06)

import functools
import importlib
import textwrap
import types
import warnings
from typing import Any, Literal, Optional

from egrecho.utils.logging import get_logger

logger = get_logger()


class ConfigurationException(Exception):
    pass


class NoneDataException(Exception):
    pass


def rich_exception_info(fn):
    @functools.wraps(fn)
    def wrapper(*args, **kwargs):
        try:
            return fn(*args, **kwargs)
        except Exception as e:
            msg = f"{e}\n[extra info] When calling: {fn.__qualname__}(args={args} kwargs={kwargs})"
            raise type(e)(msg)

    return wrapper


def format_exc_msg(message):
    message = textwrap.fill(textwrap.dedent(message), 110).strip()
    message = "\n" + textwrap.indent(message, "    ") + "\n"
    return message


def deprecated(deprecated_info: Optional[str] = None):
    def decorator(fn_or_cls):
        if not callable(fn_or_cls):
            raise TypeError(
                f"You can only deprecate a callable function or class, but got: {fn_or_cls}"
            )

        @functools.wraps(fn_or_cls)
        def wrapper(*args, **kwargs):
            name = (
                fn_or_cls.__name__
                if hasattr(fn_or_cls, "__name__")
                else fn_or_cls.__class__.__name__
            )
            warning_msg = (
                f"{name} is deprecated. {deprecated_info}"
                if deprecated_info
                else f"{name} is deprecated."
            )
            warnings.warn(warning_msg, DeprecationWarning)
            return fn_or_cls(*args, **kwargs)

        return wrapper

    return decorator


def parse_bytes(size) -> float:
    "parse `size` from bytes to the largest possible unit"
    for x in ["bytes", "KB", "MB", "GB", "TB"]:
        if size < 1024.0:
            return f"{round(size, 2)} {x}"
        size /= 1024.0

    return f"{round(size, 2)}"


def parse_size(size: int, unit_type: Literal["abbrev", "multi"] = "abbrev"):
    """parse `size` to the largest possible unit."""
    if unit_type == "multi":
        units = ["", "×10³", "×10⁶", "×10⁹", "×10¹²"]
    elif unit_type == "abbrev":
        units = ["", "K", "M", "G", "T"]
    else:
        raise ValueError("unknown unit_type")
    for x in units:
        if size < 1000.0:
            return f"{round(size, 2)}{x}"
        size /= 1000.0

    return f"{round(size, 2)}"


def add_start_docstrings(*docstr):
    def docstring_decorator(fn):
        fn.__doc__ = "".join(docstr) + (fn.__doc__ if fn.__doc__ is not None else "")
        return fn

    return docstring_decorator


def add_end_docstrings(*docstr):
    def docstring_decorator(fn):
        fn.__doc__ = (fn.__doc__ if fn.__doc__ is not None else "") + "".join(docstr)
        return fn

    return docstring_decorator


def kill_name_proc(grep_str: str, force_kill: bool = False):
    """
    Kills processes based on a grep string.

    Args:
        grep_str (str):
            The grep string to search for processes.
        force_kill (bool):
            If True, uses ``SIGKILL`` (-9) to forcefully terminate processes.

    The function uses ``ps aux | grep`` to search for processes that match the given ``grep_str``.
    It then requests user confirmation to kill these processes.
    If the user agrees, it terminates the processes.
    """
    import subprocess

    grep_command = f"ps aux | grep '{grep_str}'"
    process = subprocess.Popen(grep_command, shell=True, stdout=subprocess.PIPE)
    grep_output, _ = process.communicate()
    print(grep_output.decode())
    print("### Matching processes as above. Are you sure you want kill them?")
    if if_continue():
        pid_list = []
        for line in grep_output.decode().split("\n"):
            if line.strip():
                fields = line.split()
                if len(fields) > 1:
                    pid = fields[1]
                    pid_list.append(int(pid))
        for pid in pid_list:
            _kill_process_tree(pid, force_kill)
    else:
        print("Pass.")


def _kill_process_tree(pid, kill: bool = False):
    """kill pstree. ``kill=True`` means ``kill -9``."""
    try:
        import psutil
    except ImportError as e:
        raise e
    try:
        parent = psutil.Process(pid)
        children = parent.children(recursive=True)
        for child in children:
            child.kill() if kill else child.terminate()
        psutil.wait_procs(children, timeout=5)
        parent.kill() if kill else parent.terminate()
        parent.wait(5)
    except psutil.NoSuchProcess:
        pass
    except psutil.AccessDenied:
        print(f"Permission denied while kill tree: {pid}.")


TRUES = ("yes", "true", "t", "y", "1")
FALSES = ("no", "false", "f", "n", "0")


def if_continue() -> bool:
    print(f"### Waiting for confirmation: y:{TRUES} / n:{FALSES}")

    # Read user input
    user_input = input()

    # Check if user wants to continue or not
    if user_input.lower() in TRUES:
        return True
    elif user_input.lower() in FALSES:
        return False
    else:
        raise ValueError(
            f"Truthy value expected: got {user_input} but expected one of yes/no, true/false, t/f, y/n, 1/0 (case insensitive)."
        )


def is_picklable(obj: object) -> bool:
    """Tests if an object can be pickled."""
    import pickle

    try:
        pickle.dumps(obj)
        return True
    except (pickle.PickleError, AttributeError, RuntimeError, TypeError):
        return False


def valid_import_clspath(name: str):
    """Import path must be str with dot pattern (``'calendar.Calendar'``)."""
    if not isinstance(name, str) or "." not in name:
        raise ValueError(f"Expected a dot import path string: {name}")
    if not all(x.isidentifier() for x in name.split(".")):
        raise ValueError(f"Unexpected import path format: {name}")


def class2str(value):
    """
    Extract path from class type.

    Example::

        from calendar import Calendar
        assert str(Calendar) == "<class 'calendar.Calendar'>"
        assert class2str(Calendar) == "calendar.Calendar"
    """
    s = str(value)
    s = s[s.find("'") + 1 : s.rfind("'")]  # pull out import path
    return s


# https://github.com/omni-us/jsonargparse/blob/main/jsonargparse/_util.py#get_import_path
def get_import_path(value: Any) -> Optional[str]:
    """Returns the shortest dot import path for the given object."""
    import inspect

    path = None
    module_path = getattr(value, "__module__", None)
    qualname = getattr(value, "__qualname__", "")

    if module_path is None:
        raise ValueError(f"Failed get __module__ attribute from obj: {value}.")

    if (not qualname and not inspect.isclass(value)) or (
        inspect.ismethod(value) and not inspect.isclass(value.__self__)
    ):
        path = get_module_var_path(module_path, value)
    elif qualname:
        path = module_path + "." + qualname

    if not path:
        raise ValueError(
            f"Not possible to determine the import path for object {value}."
        )

    if qualname and module_path and "." in module_path:
        module_parts = module_path.split(".")
        for num in range(len(module_parts)):
            module_path = ".".join(module_parts[: num + 1])
            module = importlib.import_module(module_path)
            if "." in qualname:  # inner case
                obj_name, attr = qualname.rsplit(".", 1)
                obj = getattr(module, obj_name, None)
                if getattr(obj, attr, None) is value:
                    path = module_path + "." + qualname
                    break
            elif getattr(module, qualname, None) is value:
                path = module_path + "." + qualname
                break
    return path


def get_module_var_path(module_path: str, value: Any) -> Optional[str]:
    module = importlib.import_module(module_path)
    for name, var in vars(module).items():
        if var is value:
            return module_path + "." + name
    return None


def locate_(path: str):
    """
    COPIED FROM `Hydra
    <https://github.com/facebookresearch/hydra/blob/f8940600d0ab5c695961ad83abd042ffe9458caf/hydra/_internal/utils.py#L614>`_.

    Locate an object by name or dotted path, importing as necessary.
    This is similar to the pydoc function ``locate``, except that it checks for
    the module from the given path from back to front.

    Behaviours like::

        path = "calendar.Calendar"
        m, c = path.rsplit('.', 1)
        mo = importlib.import_module(m)
        cl = getattr(mo, c)
    """
    if path == "":
        raise ImportError("Empty path")

    parts = [part for part in path.split(".")]
    for part in parts:
        if not len(part):
            raise ValueError(
                f"Error loading '{path}': invalid dotstring."
                + "\nRelative imports are not supported."
            )
    assert len(parts) > 0
    part0 = parts[0]
    try:
        obj = importlib.import_module(part0)
    except Exception as exc_import:
        raise ImportError(
            f"Error loading '{path}':\n{repr(exc_import)}"
            + f"\nAre you sure that module '{part0}' is installed?"
        ) from exc_import
    for m in range(1, len(parts)):
        part = parts[m]
        try:
            obj = getattr(obj, part)
        except AttributeError as exc_attr:
            parent_dotpath = ".".join(parts[:m])
            if isinstance(obj, types.ModuleType):
                mod = ".".join(parts[: m + 1])
                try:
                    obj = importlib.import_module(mod)
                    continue
                except ModuleNotFoundError as exc_import:
                    raise ImportError(
                        f"Error loading '{path}':\n{repr(exc_import)}"
                        + f"\nAre you sure that '{part}' is importable from module '{parent_dotpath}'?"
                    ) from exc_import
                except Exception as exc_import:
                    raise ImportError(
                        f"Error loading '{path}':\n{repr(exc_import)}"
                    ) from exc_import
            raise ImportError(
                f"Error loading '{path}':\n{repr(exc_attr)}"
                + f"\nAre you sure that '{part}' is an attribute of '{parent_dotpath}'?"
            ) from exc_attr
    return obj


def pprint2str(object: object, **kwargs):
    import io
    from pprint import pprint

    stream = io.StringIO()
    pprint(object, stream=stream, **kwargs)
    return stream.getvalue()
