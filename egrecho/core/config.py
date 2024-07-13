# -*- coding:utf-8 -*-
# Copyright xmuspeech (Author: Leo 2023-03-11)


import collections
import contextlib
from dataclasses import dataclass, is_dataclass
from functools import partial
from pathlib import Path
from typing import Dict, Mapping, Optional, Union

from egrecho.utils.common import (
    DataclassSerialMixin,
    GenericSerialMixin,
    asdict_filt,
    field_dict,
    fields_init_var,
    omegaconf2container,
)
from egrecho.utils.imports import _OMEGACONF_AVAILABLE
from egrecho.utils.io import ConfigFileMixin, SerializationFn, repr_dict
from egrecho.utils.logging import get_logger
from egrecho.utils.misc import ConfigurationException

field_init_dict = partial(field_dict, init_field_only=True)
logger = get_logger(__name__)


@dataclass
class DataclassConfig(ConfigFileMixin, DataclassSerialMixin):
    """
    Base class for ``dataclass`` configuration.

    Surely we could directly interact args in ``dataclass`` with outside ``argparse`` cli.
    However, cli can hardly parse nest cases (e.g., dict, nest sequence, etc.).
    What'more, adding all args to cli is redundant. So our configs folllow below rules:

        - We abandon the ``InitVar`` set for ``dataclasses``, since ``InitVar`` is
            tricky to serialize/deserialize.
        - additional keys in fields' metadata:
            - 'to_dict': if set False, will be ignored when serialization. defaults to True.
            - 'encoding_fn': custom fn when serialization.
            - 'decoding_fn': custom fn when desrialization.
    """

    def __init_subclass__(cls, **kwargs):
        init_var = fields_init_var(cls)
        if init_var:
            raise ValueError(
                f"This class ({cls.__name__}) seems have `InitVar` fields: \n"
                f"{(f.name for f in init_var)}\n"
                "which is difficult to deserialize/serialize. Try fix it by "
                f"replacing them with normal fields."
            )
        super().__init_subclass__(**kwargs)

    @classmethod
    def _valid_input_config(cls, config):
        return (
            isinstance(config, DataclassConfig)
            or isinstance(config, collections.abc.Mapping)
            or (is_dataclass(config) and not isinstance(config, type))
        )

    @classmethod
    def from_config(
        cls,
        config: Union[dict, "DataclassConfig"] = None,
        strict: bool = False,
        **kwargs,
    ):
        """
        Creates an instance from config.

        Input ``config`` can be an instance or a dict, the invalid overwrite args in
        \**kwargs will be informed and ignored if ``strict=True``.

        Args:
            config (Union[dict, DataclassConfig]):
                The configuration.
            strict:
                if True, raise exception if got unexpected arguments.
                Defautls to False.

        Returns:
            DataclassConfig:
                The new config instance.
        """
        config = config or {}
        if not cls._valid_input_config(config):
            raise TypeError(
                f'config param of {cls}.from_config should be of DataclassConfig|dataclass|dict|mapping, but got invalid type {type(config)}.'
            )

        config_kwargs = normalize_dict(config)

        config_kwargs.update(kwargs)
        valid_keys = field_init_dict(cls).keys()
        invalid_keys = config_kwargs.keys() - valid_keys
        invalid_kwargs = {}
        for key in invalid_keys:
            invalid_kwargs[key] = config_kwargs.pop(key, None)
        msg = f"Get invalid kwargs:{invalid_kwargs}"
        if invalid_kwargs and strict:
            raise ConfigurationException(f"{msg}.")
        elif invalid_kwargs:
            logger.warning_once(
                f"Get invalid kwargs:{invalid_kwargs}, ignore them.", ranks=[0]
            )
        else:
            pass
        return cls.from_dict(config_kwargs)

    def update(self, data: dict):
        return type(self).from_config(self, data)

    def simply_repr(self):
        repr_str = repr_dict(self.to_dict())
        # repr_str = re.sub(r"^", " " * 2, repr_str, 0, re.M)
        return repr_str


class GenericFileMixin(GenericSerialMixin):
    def save_to(
        self,
        *args,
        **kwargs,
    ):
        raise NotImplementedError

    @classmethod
    def fetch_from(cls, *args, **kwargs):
        raise NotImplementedError

    def to_cfg_file(
        self, path: Union[Path, str], file_type: Optional[str] = None, **kwargs
    ):
        """
        Saves current instance's configuration to config file. Weights will not be saved.
        Args:
            path: path to config file where model model configuration will be saved

        Returns:
        """
        d = self.to_dict()
        SerializationFn.save_file(d, path=path, file_type=file_type, **kwargs)

    @classmethod
    def from_cfg_file(
        cls, path: Union[Path, str], file_type: Optional[str] = None, **kwargs
    ) -> object:
        """
        Instantiates an instance from config file.
        with model weights be initialized randomly.
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

        config = SerializationFn.load_file(path, file_type=file_type, **kwargs)
        if _OMEGACONF_AVAILABLE:
            from omegaconf import OmegaConf
            from omegaconf.errors import UnsupportedValueType, ValidationError

            with contextlib.suppress(UnsupportedValueType, ValidationError):

                config = OmegaConf.create(config)
                omegaconf_resolve = kwargs.pop('omegaconf_resolve', True)
                if omegaconf_resolve:
                    return omegaconf2container(config)
                else:
                    return config
        return config


def normalize_dict(data: Union[Mapping, DataclassConfig]):
    if isinstance(data, DataclassConfig):
        return data.to_dict()
    else:
        if isinstance(data, collections.abc.Mapping):
            data = {k: data[k] for k in data}
        if (is_dataclass(data) and not isinstance(data, type)) or isinstance(
            data, dict
        ):
            if _OMEGACONF_AVAILABLE:
                data = omegaconf2container(data)
            return asdict_filt(data, filt_type="orig")
        else:
            raise ValueError(
                f"Except type of dict or dataclass, but got {type(data)!r}:{data}"
            )
