# -*- coding:utf-8 -*-
# Copyright xmuspeech (Author: Leo 2023-10)

import importlib
import inspect
import warnings
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Literal, Optional, Set, Type, Union

from egrecho.core.feature_extractor import BaseFeature
from egrecho.core.module import TopVirtualModel
from egrecho.utils import constants
from egrecho.utils.io import SerializationFn, is_remote_url, resolve_ckpt
from egrecho.utils.misc import (
    add_start_docstrings,
    class2str,
    get_import_path,
    locate_,
    rich_exception_info,
    valid_import_clspath,
)


@rich_exception_info
def save_ckpt_conf_dir(
    ckptdir: str,
    model_conf: Optional[Dict[str, Any]] = None,
    extractor: Optional[Union[Dict[str, Any], BaseFeature]] = None,
    model_type: Optional[Union[Type, str]] = None,
    feature_extractor_type: Optional[Union[Type, str]] = None,
    **kwargs,
):
    """
    Makes it convenient to load from pretrained, save extractor, model_type, etc.. to a dir.

    Construct a dir like::

        ./ckptdir
        └── ./config
            ├── model_config.yaml
            ├── feature_config.yaml
            └── types.yaml

    Args:
        ckptdir:
            the parent of savedir, it will create a ``config`` subdir as a placeholder of files.
        model_conf:
            a dict of model config.
        extractor:
            extractor can be either a dict or a instance of
            :class:`~egrecho.core.feature_extractor.BaseFeature`.
        model_type:
            model class type or class import path.
        feature_extractor_type:
           feature extractor class type or class import path.
    """

    if model_conf:
        if not isinstance(model_conf, dict):
            raise TypeError(
                f"The provided model config should be a dict, but got {type(model_conf)}."
            )

    if extractor:
        if isinstance(extractor, BaseFeature):
            feature_cls = extractor.__class__
            if feature_extractor_type:
                warnings.warn(
                    f"Passed a extractor instance of {feature_cls}, but also got param: feature_extractor_type={feature_extractor_type}, "
                    f"auto use type({feature_cls}) of this instance. Cancel param of feature_extractor_type to avoid ambiguity."
                )
            extractor = extractor.to_dict()
            feature_extractor_type = feature_cls
        elif isinstance(extractor, dict):
            pass
        else:
            raise TypeError(
                "The provided extractor shuold be a dict or an instance "
                f"of {BaseFeature!r}, but got {type(extractor)}."
            )

    types_dict = {}
    if model_type:
        types_dict[constants.MODEL_TYPE_KEY] = get_class_path(model_type)
    if feature_extractor_type:
        types_dict[constants.EXTRACTOR_KEY] = get_class_path(feature_extractor_type)

    if model_conf or extractor or types_dict:
        savedir = Path(ckptdir / constants.CHECKPOINT_CONFIG_DIRNAME)
        savedir.mkdir(parents=True, exist_ok=True)
        save_files = (
            constants.DEFAULT_MODEL_FILENAME,
            constants.DEFAULT_EXTRACTOR_FILENAME,
            constants.TYPE_FILENAME,
        )
        for item, file in zip((model_conf, extractor, types_dict), save_files):
            if item is not None:
                SerializationFn.save_file(item, savedir / file)


@dataclass
class ResolveModelResult:
    """Resolved opts.

    Args:
        checkpoint (str):
            ckpt weight path
        model_type (str):
            model type string.
        feature_config:
            loaded dict of feature extractor config.
    """

    checkpoint: Optional[str] = None
    model_type: Optional[str] = None
    feature_config: Optional[Dict[str, Any]] = None


def resolve_pretrained_model(
    checkpoint: str = "last.ckpt",
    dirpath: Optional[str] = None,
    best_k_mode: Literal["max", "min"] = "min",
    version: str = "version",
    extractor_fname: str = constants.DEFAULT_EXTRACTOR_FILENAME,
    **resolve_ckpt_kwargs,
) -> ResolveModelResult:
    """Resolves ``checkpoint``, ``model_type``, ``feats_config``.

    Checkpoint resolving see :func:`~egrecho.utils.io.resolve_ckpt.resolve_ckpt` for details.
    Auto resolve local dir like::

        ./dirpath/version_1
                └── checkpoints
                    ├── best_k_models.yaml
                    ├── last.ckpt
                    ├── abc.ckpt
                    └── ./config
                        ├── model_config.yaml
                        ├── feature_config.yaml
                        └── types.yaml

    Args:
        checkpoint (str, optional):
            The file name of checkpoint to resolve, local file needs a suffix like ``".ckpt" / ".pt"``,
            While ``checkpoint="best"`` is a preseved key means it will find ``best_k_fname`` which is
            a file contains Dict[BEST_K_MODEL_PATH, BEST_K_SCORE], and sort by its score to
            match a best ckpt. Defaults to "last.ckpt".
        dirpath (Path or str, optional):
            The root path. Defaults to None, which means the current directory.
        version (str, optional):
            The versioned subdir name. Conmmonly subdir is named as "version_0/version_1", if you specify
            the version name with a version num, it will search that version dir, otherwise choose the max number
            of version (above "version_1"). Defaults to "version".
        best_k_mode (Literal["max", "min"], optional):
            The mode for selecting the best_k checkpoint. Defaults to "min".
        extractor_fname (str):
            feature extractor file name, defaults to ``"feature_config.yaml"``, search in ``config/`` subdir.
        resolve_ckpt_kwargs (dict):
            additional kwargs to :func:`~egrecho.utils.io.resolve_ckpt.resolve_ckpt`.
    """
    if is_remote_url(checkpoint):
        warnings.warn("To be implemented, download and form a local directory.")
        ckpt_dir = dirpath or "./"
    else:
        checkpoint = resolve_ckpt(
            checkpoint=checkpoint,
            dirpath=dirpath,
            best_k_mode=best_k_mode,
            version=version,
            **resolve_ckpt_kwargs,
        )
        ckpt_dir = Path(checkpoint).resolve().parent

    # somepath/config/feature_config.yaml
    extractor_file = (
        Path(ckpt_dir) / constants.CHECKPOINT_CONFIG_DIRNAME / extractor_fname
    )
    type_file = (
        Path(ckpt_dir) / constants.CHECKPOINT_CONFIG_DIRNAME / constants.TYPE_FILENAME
    )
    feature_config = (
        SerializationFn.load_file(extractor_file) if extractor_file.is_file() else None
    )

    model_type = (
        (SerializationFn.load_file(type_file)).get("model_type")
        if type_file.is_file()
        else None
    )

    return ResolveModelResult(
        checkpoint, model_type=model_type, feature_config=feature_config
    )


def get_class_path(
    cls_or_clsname: Union[Type, str], base_cls: Optional[Type] = None
) -> str:
    if not isinstance(cls_or_clsname, (Type, str)):
        raise TypeError(f"Invalid type {type(cls_or_clsname)}.")
    if isinstance(cls_or_clsname, Type):
        if base_cls and not issubclass(cls_or_clsname, base_cls):
            raise ValueError(f"class {cls_or_clsname} is not subclass of {base_cls}.")

        return class2str(cls_or_clsname)
    else:
        return resolve_class_path_by_name(cls_or_clsname, base_cls)


@rich_exception_info
def load_module_class(
    module_path: str, base_module_type: Optional[Type] = None
) -> Type:
    """
    Given a import path which contains class and returns the class type.

    If import path is full format, it should be dot import format and the last part is the
    class name.

    If only provide model class name (without dot "."), it will resolve the subclasses of ``base_module_type``
    which have been registered via `imports` in python file and match the model name in the last part.
    if one name matches more than one model class, it'will failed and you need provide the full path
    to elimiate ambiguity.

    Args:
        module_path (str):
            The import path containing the module class. For the case only provide class name,
            that class should be registered by ``import`` in your python.
        base_module_type (Type, optional):
            The base class type to check against.

    Returns:
        Type: The class type loaded from the module path.
    """
    if base_module_type is not None:
        assert inspect.isclass(base_module_type)
    try:
        module_path = resolve_class_path_by_name(module_path, base_module_type)

        module_name, class_name = module_path.rsplit(".", 1)

        my_module = importlib.import_module(module_name)
        model_class = getattr(my_module, class_name)
    except (ImportError, ValueError) as exc:
        raise type(exc)(
            f"Failed import class from module_path=({module_path}), check the class is {base_module_type}, "
            f"if passing single name, check that class is registered by import or use full path:\n{repr(exc)}"
        )
    return model_class


@add_start_docstrings(load_module_class.__doc__)
def load_model_type(
    module_path: str, base_module_type: Type = TopVirtualModel
) -> Type[TopVirtualModel]:
    return load_module_class(module_path, base_module_type)


@add_start_docstrings(load_module_class.__doc__)
def load_extractor_type(
    module_path: str, base_module_type: Type = BaseFeature
) -> Type[BaseFeature]:
    return load_module_class(module_path, base_module_type)


def resolve_class_path_by_name(name: str, cls: Optional[Type] = None) -> str:
    class_path = name
    if cls and "." not in class_path:
        subclass_dict = defaultdict(list)
        for subclass in get_subclass_paths(cls):
            subclass_name = subclass.rsplit(".", 1)[1]
            subclass_dict[subclass_name].append(subclass)
        if name in subclass_dict:
            name_subclasses = subclass_dict[name]
            if len(name_subclasses) > 1:
                raise ValueError(
                    f"Multiple subclasses with name {name}. Give the full class path to "
                    f'avoid ambiguity: {", ".join(name_subclasses)}.'
                )
            class_path = name_subclasses[0]
    else:
        try:
            valid_import_clspath(class_path)
            imported_cls = locate_(name)
            if cls and not issubclass(imported_cls, cls):
                raise ValueError(f"Imported class of {name} is not subclass of {cls}.")
            class_path = get_import_path(imported_cls)
        except (ImportError, ValueError) as exc:
            raise exc
    return class_path


def get_subclass_paths(cls: Type) -> Set[str]:
    subclass_list = []

    def is_local(cl):
        return ".<locals>." in getattr(cl, "__qualname__", ".<locals>.")

    def is_private(class_path):
        return "._" in class_path

    def add_subclasses(cl):
        class_path = class2str(cl)
        if is_local(cl):
            return
        if not (inspect.isabstract(cl) or is_private(class_path)):
            subclass_list.append(class_path)
        for subclass in cl.__subclasses__() if hasattr(cl, "__subclasses__") else []:
            add_subclasses(subclass)

    add_subclasses(cls)

    return set(subclass_list)
