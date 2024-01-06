# -*- coding:utf-8 -*-
# Copyright xmuspeech (Author: Leo 2023-07)

from __future__ import annotations

import copy
import warnings
from collections.abc import Mapping
from dataclasses import _MISSING_TYPE, MISSING, Field, fields, is_dataclass
from enum import Enum
from importlib import import_module
from itertools import chain
from logging import getLogger
from pathlib import Path
from types import ModuleType
from typing import Any, Literal, Optional

from ._utils import DataclassT, all_subclasses
from .decoding import decode_field

logger = getLogger(__name__)


DC_TYPE_KEY = "_type_"


def default_value(field: Field) -> Any | _MISSING_TYPE:
    """Returns the default value of a field in a dataclass, if available.
    When not available, returns ``dataclasses.MISSING``.

    Args:
        field (dataclasses.Field): The dataclasses.Field to get the default value of.

    Returns:
        Union[T, _MISSING_TYPE]: The default value for that field, if present, or None otherwise.
    """
    if field.default is not MISSING:
        return field.default
    elif field.default_factory is not MISSING:  # type: ignore
        constructor = field.default_factory  # type: ignore
        return constructor()
    else:
        return MISSING


def asdict_filt(
    obj,
    *,
    dict_factory=dict,
    filt_type: Literal["default", "none", "orig"] = "default",
    init_field_only=False,
    save_dc_types=False,
) -> dict:
    """
    Recursively converts a dataclass/dict object into a filtered dictionary representation.

    Args:
        obj: The dataclass or dictionary object to convert.
        dict_factory: The factory function to create the resulting dictionary (default: dict).
        filt_type: The type of filtering to apply (default: 'default').

            - 'default': Filters out default values in the dataclass object.
            - 'none': Filters out None values in the dataclass/dict object.
            - 'orig': Original dataclasses.asdict() behavior without filtering.

        init_field_only: If True, considers only fields with ``init == True`` in dataclasses (default: False).
        save_dc_types: If True, saves the type information of dataclasses in the serialized
            dictionary (default: False).

    Returns:
        A filtered dictionary representation of the input object.

    Raises:
        TypeError: If the input object is not a dictionary or a dataclass.

    Note:
        -   This function is intended to be used as a replacement for the ``dataclasses.asdict()`` function.
        -   The ``init_field_only`` parameter is only applicable when the input object is a dataclass,
            and it controls whether only fields with ``init == True`` are considered.
        -   The ``save_dc_types`` parameter is used to include the type information of dataclasses
            in the serialized dictionary.
    """

    support_flit = ["default", "none", "orig"]
    if filt_type:
        assert (
            filt_type in support_flit
        ), f"Unsupport filt_type: {filt_type}. choose from {support_flit}"

    def _is_dataclass_instance(obj):
        return is_dataclass(obj) and not isinstance(obj, type)

    def _asdict_inner(obj, dict_factory):
        if _is_dataclass_instance(obj):
            result = []

            for f in fields(obj):
                if init_field_only and not f.init:
                    continue
                if not f.metadata.get("to_dict", True) and filt_type != "orig":
                    continue
                value = getattr(obj, f.name)

                append_flag = False
                if filt_type:
                    if filt_type == "default":
                        append_flag = (
                            not f.init
                            or value != default_value(f)
                            or f.metadata.get("include_default", False)
                        )
                    elif filt_type == "none":
                        append_flag = value is not None
                    elif filt_type == "orig":
                        append_flag = True
                    else:
                        raise ValueError(f"Unsupport filt_type: {filt_type}.")
                else:
                    append_flag = True

                if append_flag:
                    # encode before added to result
                    encoding_fn = f.metadata.get("encoding_fn")
                    if encoding_fn is not None:
                        encoded = encoding_fn(value)
                    else:
                        # if filt_type == 'default':
                        #     default_v = default_value(f)
                        #     if isinstance(default_v, Mapping) and isinstance(
                        #         value, Mapping
                        #     ):
                        #         value = {
                        #             k: v
                        #             for k, v in value.items()
                        #             if k not in default_v or v != default_v[k]
                        #         }

                        encoded = _asdict_inner(value, dict_factory)
                    result.append((f.name, encoded))

            if save_dc_types:
                class_name = obj.__class__.__qualname__
                module = type(obj).__module__
                if "<locals>" in class_name:
                    # Don't save the type of function-scoped dataclasses.
                    warnings.warn(
                        RuntimeWarning(
                            f"Dataclass type {type(obj)} is defined in a function scope, which might cause "
                            f"issues when deserializing the containing dataclass. Refusing to save the "
                            f"type of this dataclass in the serialized dictionary."
                        )
                    )
                else:
                    result.append(("_type_", module + "." + class_name))
            return dict_factory(result)
        elif isinstance(obj, tuple) and hasattr(obj, "_fields"):
            return type(obj)(*[_asdict_inner(v, dict_factory) for v in obj])
        elif isinstance(obj, (list, tuple)):
            return type(obj)(_asdict_inner(v, dict_factory) for v in obj)
        elif isinstance(obj, Mapping):
            return type(obj)(
                (
                    _asdict_inner(k, dict_factory),
                    _asdict_inner(v, dict_factory),
                )
                for k, v in obj.items()
                if (filt_type != "none" or v is not None)
            )
        elif isinstance(obj, Path):
            return obj.__fspath__()
        elif isinstance(obj, Enum):
            return obj.name
        else:
            return copy.deepcopy(obj)

    if not isinstance(obj, dict) and not _is_dataclass_instance(obj):
        raise TypeError(f"{obj} is not a dict or a dataclass")

    return _asdict_inner(obj, dict_factory)


def from_dict(
    cls: type[DataclassT], d: dict[str, Any], drop_extra_fields: Optional[bool] = None
) -> DataclassT:
    """Parses an instance of the dataclass ``cls`` from the dict ``d``.

    Args:
        cls (Type[Dataclass]): A ``dataclass`` type.
        d (Dict[str, Any]): A dictionary of raw values, obtained for example
            when deserializing a json file into an instance of class ``cls``.
        drop_extra_fields (bool, optional): Whether or not to drop extra
            dictionary keys (dataclass fields) when encountered. There are three
            options:

                - True: The extra keys are dropped, and this function returns an
                    instance of ``cls``.
                - False: The extra keys (if any) are kept, and we search through the
                    subclasses of ``cls`` for the first dataclass which has all the
                    required fields.
                - None (default): ``drop_extra_fields = not cls.decode_into_subclasses``.

    Raises:
        RuntimeError: If an error is encountered while instantiating the class.

    Returns:
        Dataclass: An instance of the dataclass ``cls``.
    """
    if d is None:
        return None

    obj_dict: dict[str, Any] = d.copy()

    init_args: dict[str, Any] = {}
    non_init_args: dict[str, Any] = {}

    if DC_TYPE_KEY in obj_dict:
        target = obj_dict.pop(DC_TYPE_KEY)
        # module, dc_type = target.rsplit(".", 1)
        live_dc_type = _locate(target)
        # live_module = importlib.import_module(module)
        # live_dc_type = getattr(live_module, dc_type)
        return from_dict(live_dc_type, obj_dict, drop_extra_fields=drop_extra_fields)

    if drop_extra_fields is None:
        drop_extra_fields = not getattr(cls, "decode_into_subclasses", False)
        logger.debug("drop_extra_fields is None. Using cls attribute.")

    logger.debug(f"from_dict for {cls}, drop extra fields: {drop_extra_fields}")
    for field in fields(cls) if is_dataclass(cls) else []:
        name = field.name
        if name not in obj_dict:
            if (
                field.metadata.get("to_dict", True)
                and field.default is MISSING
                and field.default_factory is MISSING
            ):
                logger.warning(
                    f"Couldn't find the field '{name}' in the dict with keys "
                    f"{list(d.keys())}"
                )
            continue

        raw_value = obj_dict.pop(name)
        field_value = decode_field(
            field,
            raw_value,
            containing_dataclass=cls,
            drop_extra_fields=drop_extra_fields,
        )

        if field.init:
            init_args[name] = field_value
        else:
            non_init_args[name] = field_value

    extra_args = obj_dict

    # If there are arguments left over in the dict after taking all fields.
    if extra_args:
        if drop_extra_fields:
            logger.warning(f"Dropping extra args {extra_args}")
            extra_args.clear()

        else:
            # Use the first Serializable derived class that has all the required
            # fields.
            logger.debug(f"Missing field names: {extra_args.keys()}")

            # Find all the "registered" subclasses of `cls`. (from Serializable)
            derived_classes: list[type[DataclassT]] = []

            for subclass in all_subclasses(cls):
                if subclass is not cls:
                    derived_classes.append(subclass)
            logger.debug(f"All derived classes of {cls} available: {derived_classes}")

            # All the arguments that the dataclass should be able to accept in
            # its 'init'.
            req_init_field_names = set(chain(extra_args, init_args))

            # Sort the derived classes by their number of init fields, so that
            # we choose the first one with all the required fields.
            derived_classes.sort(key=lambda dc: len(get_init_fields(dc)))

            for child_class in derived_classes:
                logger.debug(
                    f"child class: {child_class.__name__}, mro: {child_class.mro()}"
                )
                child_init_fields: dict[str, Field] = get_init_fields(child_class)
                child_init_field_names = set(child_init_fields.keys())

                if child_init_field_names >= req_init_field_names:
                    # `child_class` is the first class with all required fields.
                    logger.debug(f"Using class {child_class} instead of {cls}")
                    return from_dict(child_class, d, drop_extra_fields=False)

    init_args.update(extra_args)
    try:
        instance = cls(**init_args)  # type: ignore
    except TypeError as e:
        # raise RuntimeError(f"Couldn't instantiate class {cls} using init args {init_args}.")
        raise RuntimeError(
            f"Couldn't instantiate class {cls} using init args {init_args.keys()}: {e}"
        )

    for name, value in non_init_args.items():
        logger.debug(f"Setting non-init field '{name}' on the instance.")
        setattr(instance, name, value)
    return instance


def _locate(path: str) -> Any:
    """
    COPIED FROM Hydra:
    https://github.com/facebookresearch/hydra/blob/f8940600d0ab5c695961ad83abd042ffe9458caf/hydra/_internal/utils.py#L614

    Locate an object by name or dotted path, importing as necessary.
    This is similar to the pydoc function `locate`, except that it checks for
    the module from the given path from back to front.
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
        obj = import_module(part0)
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
            if isinstance(obj, ModuleType):
                mod = ".".join(parts[: m + 1])
                try:
                    obj = import_module(mod)
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


def get_init_fields(dataclass: type) -> dict[str, Field]:
    result: dict[str, Field] = {}
    for field in fields(dataclass):
        if field.init:
            result[field.name] = field
    return result
