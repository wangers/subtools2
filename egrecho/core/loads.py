# -*- coding:utf-8 -*-
# Copyright xmuspeech (Author: Leo 2023-10)

import collections
import importlib
import inspect
import shutil
import tempfile
import warnings
from argparse import Namespace
from collections import defaultdict
from copy import deepcopy
from dataclasses import dataclass
from pathlib import Path
from typing import (
    Any,
    Dict,
    Iterable,
    Literal,
    MutableMapping,
    Optional,
    Set,
    Tuple,
    Type,
    Union,
)

import torch

from egrecho.core.feature_extractor import BaseFeature
from egrecho.core.module import TopVirtualModel
from egrecho.core.tokenizer import BaseTokenizer
from egrecho.utils import constants
from egrecho.utils.common import (
    _OMEGACONF_AVAILABLE,
    ObjectDict,
    SaveLoadMixin,
    asdict_filt,
    dict_union,
    omegaconf2container,
    omegaconf_handler,
)
from egrecho.utils.cuda_utils import release_memory
from egrecho.utils.dist import is_global_rank_zero
from egrecho.utils.io import SerializationFn, is_remote_url, resolve_ckpt, resolve_file
from egrecho.utils.logging import get_logger
from egrecho.utils.misc import (
    ConfigurationException,
    add_end_docstrings,
    add_start_docstrings,
    class2str,
    get_import_path,
    locate_,
    pprint2str,
    rich_exception_info,
    valid_import_clspath,
)
from egrecho.utils.types import _INIT_WEIGHT, InitWeightType, StrEnum

logger = get_logger()


class LoadArgs(ObjectDict):
    _cls_: str

    def __init__(self, _cls_=None, **kwargs):
        super().__init__(_cls_=_cls_, **kwargs)


class HLoads(ObjectDict):
    """
    Structs load parameters as::

        model:
            _cls_: egrecho.models.ecapa.model.EcapaModel
            # override_init_model_cfg
            config: {}
            # other kwargs placeholder
            ...
        feature_extractor:
            _cls_: egrecho.data.features.feature_extractor_audio.KaldiFeatureExtractor
            # kwargs passing to _cls_.fetch_from
            ...
    """

    model: Optional[Union[Dict, LoadArgs]]
    feature_extractor: Optional[Union[Dict, LoadArgs]]
    tokenizer: Optional[Union[Dict, LoadArgs]]

    def gather_types_dict(self) -> Dict:
        types_dict = {
            k: v.get("_cls_", None)
            for k, v in self.items()
            if isinstance(v, collections.abc.Mapping)
        }
        return asdict_filt(types_dict, filt_type="none")

    def merge_types_dict(self, types_dict: Dict) -> Dict:

        for t_k, t_v in types_dict.items():
            if self.get(t_k) and isinstance(self[t_k], collections.abc.MutableMapping):
                self[t_k]["_cls_"] = t_v
            elif self.get(t_k):
                raise TypeError(
                    f"Invalid update type={type(self[t_k])} for the key={t_k}, value={self[t_k]}"
                )
            else:

                self[t_k] = LoadArgs(t_v)
        return self.gather_types_dict()

    @classmethod
    def from_config(
        cls,
        config: Optional["HL"] = None,
        **kwargs,
    ) -> "HLoads":
        """
        Creates hloads from config.

        Input ``config`` can be an instance of ``dict|str|Path|HLoads|MutableMapping``, the
        \**kwargs will be merged recursely into ``config``.

            Normalize dict -> Merge -> (maybe) Omegaconf resolve -> Instantiate -> Output

        Args:
            config: The configuration.
            \**kwargs: Override kwargs

        Returns:
            HLoads:
                The new hloads instance.
        """

        config = config or {}

        hloads = normalize_hloads(config)
        hloads = dict_union(hloads, kwargs)
        if _OMEGACONF_AVAILABLE:
            hloads = omegaconf_handler(hloads)
        return cls.from_dict(hloads)

    @classmethod
    def from_dict(cls, data: dict) -> "HLoads":
        if _OMEGACONF_AVAILABLE:
            data = omegaconf2container(data)
        for k, v in data.items():
            if isinstance(v, LoadArgs):
                continue
            if isinstance(v, dict):
                data[k] = ObjectDict(v)

        return cls(**data)

    def to_dict(self, filt_type="none") -> dict:
        return asdict_filt(self, filt_type=filt_type)

    def to_cfg_file(
        self, path: Union[Path, str], file_type: Optional[str] = None, **kwargs
    ):
        """
        Saves hloads.

        Args:
            path: path to config file.
        """
        filt_type = kwargs.pop("filt_type", "none")
        d = self.to_dict(filt_type=filt_type)
        SerializationFn.save_file(d, path=path, file_type=file_type, **kwargs)

    @classmethod
    def from_cfg_file(
        cls, path: Union[Path, str], file_type: Optional[str] = None, **kwargs
    ) -> "HLoads":
        """
        Get hloads from file.

        Args:
            path: path to config file.

        Returns:

        """
        data = cls.load_cfg_file(path, file_type, **kwargs)
        return cls.from_dict(data)

    @staticmethod
    def load_cfg_file(
        path: Union[Path, str], file_type: Optional[str] = None, **kwargs
    ) -> Dict:
        # resloves later in from_dict
        omegaconf_resolve = kwargs.pop("omegaconf_resolve", False)
        config = SerializationFn.load_file(path, file_type=file_type, **kwargs)
        if _OMEGACONF_AVAILABLE:
            return omegaconf_handler(config, omegaconf_resolve=omegaconf_resolve)
        else:
            return config


_ALLOWED_HLOADS_TYPES = (HLoads, MutableMapping, str, Path, Namespace)
HL = Union[HLoads, MutableMapping, str, Path, Namespace, None]


_MODEL_LOADED = Union[
    Tuple[torch.nn.Module, Dict[str, Any]],
    Tuple[Type, Dict[str, Any], Dict[str, Any]],
]
_MODEL_COMPONENTS = Tuple[Optional[_MODEL_LOADED], Dict[str, Any], HLoads]


class HResults(ObjectDict):
    """
    Structs loaded result of :class:`SaveLoadHelper.fetch_from`.
    """

    model: Union[torch.nn.Module, Tuple[Type, Dict[str, Any]]]
    feature_extractor: Any
    tokenizer: Any
    hloads: HLoads

    @property
    def is_lazy_model(self) -> bool:
        if self.get("model"):
            return isinstance(self.model, (tuple, list)) and isinstance(
                self.model[0], Type
            )
        return False


def normalize_hloads(hl: HL) -> dict:
    if isinstance(hl, Namespace):
        hl = vars(hl)
    if not isinstance(hl, _ALLOWED_HLOADS_TYPES):
        raise ValueError(
            f"Unsupported config type of {type(hl)}. Should be one of {_ALLOWED_HLOADS_TYPES}"
        )
    if isinstance(hl, (str, Path)):
        hl = HLoads.load_cfg_file(hl, omegaconf_resolve=True)
    if not isinstance(hl, dict):  # MutableMapping
        hl = {k: v for k, v in hl.items()}
    return asdict_filt(hl, filt_type="orig")


SAVELOAD_EXAMPLE_DOCSTRING = r"""
        Example:

        .. code-block::

            from egrecho.core.loads import SaveLoadHelper
            from egrecho.models.ecapa.model import EcapaModel
            from egrecho.data.features.feature_extractor_audio import KaldiFeatureExtractor
            sl_helper = SaveLoadHelper()
            extractor = KaldiFeatureExtractor()
            model = EcapaModel()
            dirpath = 'testdir/ecapa'
            sl_helper.save_to(dirpath,model_or_state=model,components=extractor)

        .. code-block::

            $ tree testdir/ecapa
            testdir/ecapa/
            ├── config
            │   ├── feature_config.yaml
            │   ├── model_config.yaml
            │   └── types.yaml
            └── model_weight.ckpt

        .. code-block::

            hresults = sl_helper.fetch_from(dirpath)
            assert isinstance(hresults.model,EcapaModel)
            assert isinstance(hresults.feature_extractor, KaldiFeatureExtractor)
            # hloads control random init
            hloads = {'model': {'init_weight': 'random'}}
            hresults = sl_helper.fetch_from(dirpath, hloads=hloads)
            # kwargs overrides to pretrained again
            hresults = sl_helper.fetch_from(dirpath, hloads=hloads, model={'init_weight': 'pretrained'})

            # now remove types.yaml
            # rm -f testdir/ecapa/config/types.yaml
            hresults = sl_helper.fetch_from(dirpath, single_key='model')
            # raise ConfigurationException: Failed request model type
            # Let's complete the model type
            model_cls = 'egrecho.models.ecapa.model.EcapaModel'
            hresults = sl_helper.fetch_from(dirpath, single_key='model', model={'_cls_': model_cls})
            assert isinstance(hresults.model,EcapaModel)
            # Type is ok
            model_cls = EcapaModel
            hresults = sl_helper.fetch_from(dirpath, single_key='model', model={'_cls_': model_cls})
            assert isinstance(hresults.model,EcapaModel)
            # classname string is ok as EcapaModel is already imported
            model_cls = 'EcapaModel'
            hresults = sl_helper.fetch_from(dirpath, single_key='model', model={'_cls_': model_cls})
            assert isinstance(hresults.model,EcapaModel)
            model_cls = 'Valle'
            # Error as 'Valle' is not registed.
            hresults = sl_helper.fetch_from(
                dirpath,
                single_key="model",
                kwargs_recurse_override=False,
                model={"_cls_": model_cls, "init_weight": "random", "config": None},
            )  # only load model without weight and eliminate the influences of Ecapa model directory
            from egrecho.models.valle.model import Valle
            # Try again.
            hresults = sl_helper.fetch_from(
                dirpath,
                single_key="model",
                kwargs_recurse_override=False,
                model={"_cls_": model_cls, "init_weight": "random", "config": None},
            )
            assert isinstance(hresults.model, Valle)
"""

SAVELOAD_EXAMPLE_DOCSTRING_CLS = SAVELOAD_EXAMPLE_DOCSTRING.replace(
    "\n        ",
    "\n    ",
)


@add_end_docstrings(SAVELOAD_EXAMPLE_DOCSTRING_CLS)
class SaveLoadHelper:
    """
    Save/load model in a directory, overwrite this for any special manners.
    """

    model_weight_ckpt_name = constants.MODEL_WEIGHTS_FNAME

    @add_end_docstrings(SAVELOAD_EXAMPLE_DOCSTRING)
    def save_to(
        self,
        savedir,
        model_or_state: Optional[Union[TopVirtualModel, Dict[str, Any]]] = None,
        components: Optional[Iterable[Any]] = None,
        **kwargs,
    ):
        """Save model after pretraining.

        Exports a pretrained model with its subcompnents (configs, tokenizer, etc ...)
        outdir like::

            ./savedir
            ├── model_weight.ckpt
            └── ./config
                ├── model_config.yaml
                ├── feature_config.yaml
                └── types.yaml

        Args:
            savedir: local directory.
            model_or_state: TopVirtualModel object or model state dict to be saved.
            components: obj of tokenizer, feature extractor etc..
        """
        assert savedir is not None
        components = components or []
        if not isinstance(components, (list, tuple)):
            components = [components]
        savedir = Path(savedir)

        if is_global_rank_zero():
            savedir.mkdir(parents=True, exist_ok=True)
            with tempfile.TemporaryDirectory(suffix="_tmp", dir=str(savedir)) as tmpdir:
                tmpdir = Path(tmpdir)

                tmp_ckpt_path = tmpdir / self.model_weight_ckpt_name
                tmp_cfg_dir = tmpdir / constants.CHECKPOINT_CONFIG_DIRNAME
                tmp_cfg_dir.mkdir(exist_ok=True)
                types_dict = {}
                if model_or_state is not None:
                    logger.info(f"About to save model state in {tmpdir}.")
                    if isinstance(model_or_state, TopVirtualModel):
                        types_dict[constants.MODEL_KEY] = get_class_path(
                            model_or_state.__class__
                        )
                        model_or_state.to_cfg_file(
                            tmp_cfg_dir / constants.DEFAULT_MODEL_FILENAME
                        )
                        self._save_state_dict_to_disk(
                            model_or_state.state_dict(), tmp_ckpt_path
                        )
                    elif isinstance(model_or_state, dict):
                        state_dict = model_or_state.get("state_dict", model_or_state)
                        self._save_state_dict_to_disk(state_dict, tmp_ckpt_path)
                    shutil.move(tmp_ckpt_path, savedir / self.model_weight_ckpt_name)
                logger.info(f"About to save config dir in {tmp_cfg_dir}.")
                self.update_conf_dir(tmp_cfg_dir, *components, types_dict=types_dict)
                self.copy_conf_dir(
                    tmp_cfg_dir, savedir / constants.CHECKPOINT_CONFIG_DIRNAME
                )
                logger.info(f"Save to {str(savedir)} done.")

    @add_end_docstrings(SAVELOAD_EXAMPLE_DOCSTRING)
    def fetch_from(
        self,
        srcdir: Union[str, Path],
        hloads: Optional[Union[str, Path, Dict[str, Any]]] = None,
        base_model_cls: Optional[Union[str, Type]] = None,
        skip_keys: "SkipType" = None,
        single_key: Optional[str] = None,
        return_hloads: bool = False,
        kwargs_recurse_override: bool = True,
        **kwargs,
    ) -> HResults:
        """
        Load module class in Hloads. Return HResults dict contains (MODEL, FEATURE_EXTRACTOR, ...), MODEL could be:

        - A instance of model.
        - ``None`` when skip_keys apply on model.

        NOTE:
            Workflow is defined as a sequence of the following operations:

            1. Reslove hloads.
                User could use config file or passing kwargs to control behaviour.
            2. Load available class types via :meth:`load_types_dict`.
                Note that passing classname as type is available if that class
                is **imported** in current namespace and **is a subclass of some base module**, support
                (:class:`TopVirtualModel`, :class:`BaseFeaature`, :class:`BaseTokenizer`) now. E.g., instead of
                passing a full class path: ``'egrecho.models.ecapa.model.EcapaModel'``, user can first import
                that class in python module act as a register manner, then the class name ``'EcapaModel'`` is
                available. This mechianism could simplify parameter control.
            3. Instantiate classes :meth:`instantiate_classes` according to typtes dict.
                Specially, the model is loaded lazily, i.e., a tuple of (MODEL_CLS, INIT_MODEL_CFG, LEFT_MODEL_CFG)
                resloved by ``instantiate_model(lazy_model=True)``.
            4. Instantiate model :meth:`_instantiate_model`.
                User might overwrite this method in subclasses.
            5. Load model weight.

        Args:
            srcdir:
                Model directory like::

                    ./srcdir
                    ├── model_weight.ckpt
                    └── ./config
                        ├── model_config.yaml
                        ├── feature_config.yaml
                        └── types.yaml

            hloads: Path|str|Dict, optional
                Hparam dict/file with hierarchical structure as in this example::

                    model:
                        _cls_: egrecho.models.ecapa.model.EcapaModel
                        # override_init_model_cfg
                        config: {}
                        # other kwargs placeholder
                        ...
                    feature_extractor:
                        _cls_: egrecho.data.features.feature_extractor_audio.KaldiFeatureExtractor
                        # kwargs passing to _cls_.fetch_from
                        ...

                You most likely won't need this since default behaviours well. However, this arguments give a
                chance to complete/override kwargs.
            base_model_cls:
                Base model class
            single_key:
                Load specify key.
            skip_keys:
                Skip keys, e.g., skip model. Invalid when ``single_key=True``.
            kwargs_recurse_override:
                Whether kwargs recursely overrides hloads.
            kwargs(Dict[str,Any]): Overrides hloads.

                Hint:
                    Example of model-related params.

                    .. code-block::

                        self.fetch_from(..., model=dict(init_weight='last.ckpt', strict=False)

                    - init_weight: Init weight from ('pretrained' or 'random'), or string ckpt
                      name (model_weight.ckpt) or full path to ckpt /path/to/model_weight.ckpt.
                      Default: ``'pretrained'``
                    - map_location: MAP_LOCATION_TYPE as in torch.load(). Defaults to 'cpu'.
                      If you preferring to load a checkpoint saved a GPU model
                      to GPU, set it to None (not move to another GPU) or set a specified device.
                    - strict : bool, optional, Whether to strictly enforce that the keys in checkpoint match
                      the keys returned by this module's state dict. Defaults to True.

        Returns:
            A HResults dict.
        """
        ckpt_dir = Path(srcdir)
        cfg_dir = ckpt_dir / constants.CHECKPOINT_CONFIG_DIRNAME
        if kwargs_recurse_override:
            hloads = HLoads.from_config(hloads, **kwargs)
        else:
            hloads = HLoads.from_config(hloads)
            hloads.update(**kwargs)
        if hloads:
            logger.info_once(
                f"Hloads:\n{pprint2str(hloads)}",
                ranks=0,
            )
        model, components, hloads = self.load_model_with_components(
            cfg_dir,
            hloads=hloads,
            base_model_cls=base_model_cls,
            skip_keys=skip_keys,
            single_key=single_key,
            lazy_model=True,
        )
        feature_extractor = components.pop(constants.EXTRACTOR_KEY, None)
        tokenizer = components.pop(constants.TOKENIZER_KEY, None)
        hresults = HResults(
            model=None,
            feature_extractor=feature_extractor,
            tokenizer=tokenizer,
            **components,
        )
        if return_hloads:
            hresults.hloads = hloads
        if not model:
            return hresults
        else:
            model_cls, init_model_cfg, model_cfg = model

            init_weight: _INIT_WEIGHT = model_cfg.pop(
                "init_weight", InitWeightType.PRETRAINED
            )
            map_location = model_cfg.pop("map_location", "cpu")
            strict = model_cfg.pop("strict", True)
            state_dict = None
            model_instance = self._instantiate_model(model_cls, init_model_cfg)
            if not init_weight or init_weight == InitWeightType.RANDOM:
                model_instance.to(map_location)
            else:
                if init_weight == InitWeightType.PRETRAINED:
                    ckpt_path = ckpt_dir / self.model_weight_ckpt_name
                else:
                    ckpt_path = Path(resolve_file(str(init_weight), ckpt_dir))
                if not ckpt_path.is_file():
                    raise FileNotFoundError(f"{ckpt_path} missing.")
                state_dict = self._load_state_dict_from_disk(
                    ckpt_path, map_location=map_location
                )
                state_dict = state_dict.get("state_dict", state_dict)  # handle pl ckpt

                state_dict = self.modify_state_dict(state_dict, model_cfg)
                device = next(
                    (t for t in state_dict.values() if isinstance(t, torch.Tensor)),
                    torch.tensor(0),
                ).device
                self.load_instance_with_state_dict(model_instance, state_dict, strict)

                model_instance.to(device)
            hresults.model = model_instance
            logger.info(
                f"Model {model_instance.__class__.__name__} was successfully fetched from {ckpt_dir}. "
                f"with init_weight={init_weight}"
            )
            release_memory(state_dict)
            return hresults

    @staticmethod
    def _save_state_dict_to_disk(state_dict, filepath):
        torch.save(state_dict, filepath)

    @staticmethod
    def update_conf_dir(cfg_dir, *components, types_dict: Optional[Dict] = None):
        types_dict = types_dict or {}
        cfg_dir = Path(cfg_dir)
        for cop in components:
            if isinstance(cop, BaseTokenizer):
                cop.save_to(cfg_dir)
                tokenizer_cls = cop.__class__
                types_dict[constants.TOKENIZER_KEY] = get_class_path(tokenizer_cls)
            elif isinstance(cop, BaseFeature):
                cop.save_to(cfg_dir)
                feature_cls = cop.__class__
                types_dict[constants.EXTRACTOR_KEY] = get_class_path(feature_cls)
            else:
                raise TypeError(f"Unsupported type={type(cop)} when saving {cop}")
        if types_dict:
            if (orig_type_yml := (cfg_dir / constants.TYPE_FILENAME)).is_file():
                orig_types_dict = SerializationFn.load_file(orig_type_yml)
                types_dict = {**orig_types_dict, **types_dict}

            SerializationFn.save_file(types_dict, cfg_dir / constants.TYPE_FILENAME)

    @staticmethod
    def copy_conf_dir(srcdir, dstdir):
        if (src_type_yml := (Path(srcdir) / constants.TYPE_FILENAME)).is_file() and (
            dst_type_yml := (Path(srcdir) / constants.TYPE_FILENAME)
        ).is_file():
            src_types_dict = SerializationFn.load_file(src_type_yml)
            dst_types_dict = SerializationFn.load_file(dst_type_yml)
            SerializationFn.save_file(
                {**dst_types_dict, **src_types_dict}, srcdir / constants.TYPE_FILENAME
            )
        shutil.copytree(srcdir, dstdir, dirs_exist_ok=True)

    def load_types_dict(
        self,
        cfg_dir: Union[str, Path],
        types_dict: Optional[Dict] = None,
        base_model_cls: Optional[Union[str, Type]] = None,
        skip_keys: "SkipType" = None,
        single_key: Optional[str] = None,
    ) -> Dict[str, Type]:
        if bool(single_key) and skip_keys:
            raise ConfigurationException(
                f"Ilegal Both single_key={single_key} and skip_keys={skip_keys} are set.",
                ranks=0,
            )
        skip_keys = parse_skip_sets(skip_keys)
        types_dict = types_dict or {}
        if (type_file := (Path(cfg_dir) / constants.TYPE_FILENAME)).is_file():
            types_dict = dict_union(SerializationFn.load_file(type_file), types_dict)

        types_dict = asdict_filt(types_dict, filt_type="none")
        if (
            single_key
            and single_key != constants.MODEL_KEY
            and not types_dict.get(single_key, None)
        ):
            raise ConfigurationException(
                f"Want to load a single key={single_key} which is not extsts in types_dict:\n{pprint2str(types_dict)}"
            )
        elif single_key:
            types_dict = {single_key: types_dict.get(single_key, None)}

        model_only = (
            isinstance(skip_keys, SpecialSkipType)
            and skip_keys == SpecialSkipType.OTHERS
        ) or single_key == constants.MODEL_KEY
        # model only
        if model_only:
            types_dict = {
                constants.MODEL_KEY: types_dict.get(constants.MODEL_KEY),
            }
            types_dict = self._load_model_type(
                types_dict, base_model_cls=base_model_cls
            )
            if not types_dict.get(constants.MODEL_KEY):
                raise ConfigurationException(
                    "Failed request model type, provide it via set it (e.g, config/types.yaml) "
                    "or in Hloads."
                )
            return types_dict
        else:
            if (
                isinstance(skip_keys, SpecialSkipType)
                and skip_keys == SpecialSkipType.MODEL
            ) or single_key != constants.MODEL_KEY:
                skip_keys = set(constants.MODEL_KEY)
            for k in skip_keys:
                types_dict.pop(k, None)
            if constants.MODEL_KEY not in skip_keys:
                types_dict = self._load_model_type(
                    types_dict, base_model_cls=base_model_cls
                )
            # load other class types
            return self._load_componets_type(types_dict)

    @staticmethod
    def _load_model_type(
        types_dict: Optional[Dict] = None,
        base_model_cls: Optional[Union[str, Type]] = None,
    ) -> Dict[str, Union[str, Type]]:
        types_dict = types_dict or {}

        cfg_type = types_dict.pop(constants.MODEL_KEY, None)

        if base_model_cls and isinstance(base_model_cls, str):
            base_model_cls = load_extend_default_type(base_model_cls, TopVirtualModel)

        if bool(cfg_type) + bool(base_model_cls) == 0:
            return types_dict
        elif not bool(cfg_type):
            model_type = base_model_cls
        else:

            if isinstance(cfg_type, Type):
                if base_model_cls and not issubclass(cfg_type, base_model_cls):
                    raise TypeError(
                        f"Config model class of {cfg_type} is not subclass of {base_model_cls}."
                    )
                else:
                    model_type = cfg_type
            else:
                assert isinstance(
                    cfg_type, str
                ), f"cfg_type={cfg_type} must be of class type or str, but got type={type(cfg_type)}"
                if base_model_cls is not None:
                    model_type = load_module_class(cfg_type, base_model_cls)
                else:
                    model_type = load_extend_default_type(cfg_type, TopVirtualModel)
        types_dict[constants.MODEL_KEY] = model_type
        return types_dict

    @staticmethod
    def _load_componets_type(
        types_dict: Optional[Dict] = None,
    ) -> Dict[str, Union[str, Type]]:
        types_dict = types_dict or {}
        for t_key, t_tgt in types_dict.items():
            if t_key == constants.MODEL_KEY:
                continue
            if isinstance(t_tgt, str):
                if t_key == constants.EXTRACTOR_KEY:
                    # support class name of BaseFeature
                    types_dict[t_key] = load_extend_default_type(t_tgt, BaseFeature)
                elif t_key == constants.TOKENIZER_KEY:
                    types_dict[t_key] = load_extend_default_type(t_tgt, BaseTokenizer)
                else:
                    # must full path
                    types_dict[t_key] = load_module_class(t_tgt)

        return types_dict

    def load_model_with_components(
        self,
        cfg_dir: str,
        hloads: HLoads = None,
        base_model_cls: Optional[Union[str, Type]] = None,
        skip_keys: "SkipType" = None,
        single_key: Optional[str] = None,
        lazy_model: bool = False,
    ) -> _MODEL_COMPONENTS:
        """
        Load module class in Hloads. Return tuple contains (MODEL, COMPONETS, HLOADS), where TYPES_DICT
        indicates what classes will be used to instance an object. Model could be:

        - A tuple of (MODEL_INSTANCE, LEFT_MODEL_CFG).
        - A tuple of (MODEL_CLS, INIT_MODEL_CFG, LEFT_MODEL_CFG) resloved as lazy model in :meth:`_instantiate_model`.
        - ``None`` when skip_keys apply on model.

        Args:
            cfg_dir:
                Directory contains cfg files.
            hloads: Path|str|Dict, optional
                Hparam dict/file with hierarchical structure as in this example::

                    model:
                        _cls_: egrecho.models.ecapa.model.EcapaModel
                        # replace default model_config.yaml
                        config_fname: some_config.yaml
                        # override_init_model_cfg
                        config: {}
                        # other kwargs placeholder
                        ...
                    feature_extractor:
                        _cls_: egrecho.data.features.feature_extractor_audio.KaldiFeatureExtractor
                        # kwargs passing to _cls_.fetch_from
                        ...

                You most likely won't need this since default behaviours well. However, this arguments give a
                chance to complete/override kwargs.
            base_model_cls:
                Base model class
            single_key:
                Load specify key.
            skip_keys:
                Skip keys, e.g., skip model. Invalid when ``single_key=True``.
            lazy_model:
                If False, instantiate model else just left mode cls with its init cfg.
                Default: False

        Returns:
            A tuple contains (MODEL, COMPONETS, HLOADS).
        """
        cfg_dir = Path(cfg_dir)

        types_dict = hloads.gather_types_dict()
        types_dict = self.load_types_dict(
            cfg_dir=cfg_dir,
            types_dict=types_dict,
            base_model_cls=base_model_cls,
            skip_keys=skip_keys,
            single_key=single_key,
        )
        types_dict = asdict_filt(types_dict)
        hloads_types_dict = hloads.merge_types_dict(types_dict)
        ignored_keys = hloads_types_dict.keys() - types_dict.keys()

        extra_msg = f"\nKeys to be ignored: {ignored_keys}" if ignored_keys else ""
        logger.info(
            f"About to instantiate:\n {types_dict}, {extra_msg}",
            ranks=0,
        )

        instances = self.instantiate_classes(
            cfg_dir,
            types_dict=types_dict,
            hloads=deepcopy(hloads),
            lazy_model=lazy_model,
        )
        model = instances.pop(constants.MODEL_KEY, None)

        return model, instances, hloads

    def instantiate_classes(
        self,
        cfg_dir: Union[str, Path],
        types_dict: Dict[str, Type],
        hloads: HLoads,
        lazy_model=False,
    ) -> Dict[str, Any]:
        cfg_dir = Path(cfg_dir)
        instances = {}
        for t_key in types_dict:
            cls_type = hloads[t_key].pop("_cls_")
            if t_key == constants.MODEL_KEY:
                instances[t_key] = self.instantiate_model(
                    cfg_dir=cfg_dir,
                    model_cls=cls_type,
                    model_cfg=hloads[t_key],
                    lazy_model=lazy_model,
                )
            else:
                if issubclass(cls_type, SaveLoadMixin):
                    try:
                        instances[t_key] = cls_type.fetch_from(cfg_dir, **hloads[t_key])
                        continue
                    except NotImplementedError as e:  # noqa
                        pass
                cfg_dict = hloads[t_key]
                if (
                    cfg_fname := hloads[t_key].pop("config_fname", None)
                    and (cfg_dir / cfg_fname).is_file()
                ):
                    cfg_dict = dict_union(
                        omegaconf_handler(
                            SerializationFn.load_file(cfg_dir / cfg_fname)
                        ),
                        hloads[t_key],
                    )
                if hasattr(cls_type, "from_dict") and callable(cls_type.from_dict):
                    instances[t_key] = cls_type.from_dict(cfg_dict)
                else:
                    raise RuntimeError(
                        f"Failed to initiate {cls_type}. [Hint] This func:instantiate_classes supports "
                        f"instantiate type {SaveLoadMixin}, class with classmethod cls.from_dict(...)."
                    )
        return instances

    def instantiate_model(
        self,
        cfg_dir: Union[str, Path],
        model_cls: Type,
        model_cfg,
        lazy_model: bool = False,
    ) -> _MODEL_LOADED:
        init_model_cfg, model_cfg = self._sanitize_model_cfg(cfg_dir, model_cfg)
        if not lazy_model:
            return self._instantiate_model(model_cls, init_model_cfg), model_cfg
        else:
            return model_cls, init_model_cfg, model_cfg

    def _sanitize_model_cfg(self, cfg_dir: Union[str, Path], model_cfg):
        """Extracts init_cfg for model initiation."""
        cfg_dir = Path(cfg_dir)
        init_cfg_fname = model_cfg.pop("config_fname", constants.DEFAULT_MODEL_FILENAME)
        override_init_model_cfg = model_cfg.pop("config", None) or {}

        if (init_cfg_file := (cfg_dir / str(init_cfg_fname))).is_file():
            init_model_cfg_from_file = omegaconf_handler(
                SerializationFn.load_file(init_cfg_file)
            )
        else:
            init_model_cfg_from_file = {}
            logger.warning(
                f"Missing default model init config {str(init_cfg_fname)} in {str(cfg_dir)}"
            )
        init_model_cfg = dict_union(
            init_model_cfg_from_file,
            override_init_model_cfg,
        )
        return init_model_cfg, model_cfg

    def _instantiate_model(self, model_cls, init_model_cfg, **kwargs):
        """Instantiate TopVirtualModel, override to support others."""
        if issubclass(model_cls, TopVirtualModel) and not (
            model_cls == TopVirtualModel
        ):
            model = model_cls.from_dict(config=init_model_cfg)
            return model
        else:
            raise NotImplementedError(
                f"Unsuportted initiate model: {model_cls}, Please provide a subclass of {TopVirtualModel.__qualname__}"
            )

    def modify_state_dict(self, state_dict, model_cfg):
        """
        Allows to modify the state dict before loading parameters into a model.
        Args:
            state_dict: The state dict restored from the checkpoint.
            model_cfg: A model level dict object.
        Returns:
            A potentially modified state dict.
        """
        return state_dict

    def load_instance_with_state_dict(self, instance, state_dict, strict):
        """
        Utility method that loads a model instance with the (potentially modified) state dict.

        Args:
            instance: ModelPT subclass instance.
            state_dict: The state dict (which may have been modified)
            strict: Bool, whether to perform strict checks when loading the state dict.
        """
        instance.load_state_dict(state_dict, strict=strict)

    @staticmethod
    def _load_state_dict_from_disk(model_weights, map_location=None):
        # BUG for cuda now, can not del state cache when lightning
        return torch.load(model_weights, map_location="cpu", weights_only=True)


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
        types_dict[constants.MODEL_KEY] = get_class_path(model_type)
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
        (SerializationFn.load_file(type_file)).get(constants.MODEL_KEY)
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


def load_extend_default_type(module_path: str, default_type: Optional[Type] = None):
    """
    Allows simple class name when ``default_type`` is provided.

    - If import path is dot "calender.Calender" format, it should be full import format and the last part is the
      class name. Note that ``default_type`` is **ignored** in this case.
    - If only provide model class name (without dot "."), must provide ``default_type`` as ``base_module_type``,
      then it will resolve the subclasses of ``default_type`` which have been registered via `imports` in python
      file and match the model name in the last part. If one name matches more than one model class,
      it'will failed and you need provide the full path to elimiate ambiguity.

    Args:
        module_path (str):
            The import path containing the module class. For the case only provide class name,
            that class should be a subclass of ``default_type`` registered by ``import`` in your python.
        default_type (Type, optional):
            The default class type.

    Returns:
        Type: The class type loaded from the module path.
    """
    assert isinstance(module_path, str), module_path
    if default_type is None or "." in module_path:
        return load_module_class(module_path)
    assert isinstance(default_type, Type), default_type
    return load_module_class(module_path, default_type)


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


class SpecialSkipType(StrEnum):
    """Special skip mode when loading module.

    - ``model``: exclude model.
    - ``others``: only model.

    """

    MODEL = constants.MODEL_KEY
    OTHERS = "others"

    @staticmethod
    def names():
        return list(SpecialSkipType._member_map_.values())


SkipType = Union[str, Literal["model", "others", "null"], Set[str]]


def parse_skip_sets(skip: SkipType) -> Union[SpecialSkipType, Set[str]]:
    if skip is None:
        return set()
    if isinstance(skip, SpecialSkipType):
        return skip
    if isinstance(skip, str) and (skip in SpecialSkipType.names()):
        skip = SpecialSkipType(skip.lower())
    elif isinstance(skip, str):
        if skip.lower() == "null":
            return set()
        skip = skip.strip().replace(",", " ").split()
    else:
        if not isinstance(skip, Iterable):
            raise TypeError(f"Ilegal type={type(skip)}")
        skip = list(skip)
    return set(skip) if skip else set()
