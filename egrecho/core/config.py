# -*- coding:utf-8 -*-
# Copyright xmuspeech (Author: Leo 2023-03-11)


import collections
import copy
from dataclasses import dataclass, is_dataclass
from functools import partial
from typing import Union

from egrecho.utils.common import (
    DataclassSerialMixin,
    asdict_filt,
    field_dict,
    fields_init_var,
)
from egrecho.utils.io import ConfigFileMixin, repr_dict
from egrecho.utils.logging import get_logger
from egrecho.utils.misc import ConfigurationException

field_init_dict = partial(field_dict, init_field_only=True)
logger = get_logger()


@dataclass
class DataclassConfig(ConfigFileMixin, DataclassSerialMixin):
    """
    Base class for `dataclass` configuration.

    NOTE: Surely we could directly interact args in `dataclass` with outside `argparse` cli.
    However, cli can hardly parse nest cases (e.g., dict, nest sequence, etc.).
    What'more, adding all args to cli is redundant. So our configs folllow below rules:
        - We abandon the `InitVar` set for `dataclasses`, as `InitVar` is
            tricky to serialize/deserialize.
        - additional keys in fields' metadata:
            - 'cmd': if set True, means we will handle it in outside argparser, but it is silence now. defaults to False.
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
    def from_config(
        cls,
        config: Union[dict, "DataclassConfig"] = None,
        strict: bool = False,
        **kwargs,
    ):
        """
        Create an instance from config.

        Input `config` can be an instance or a dict, the invalid overwrite args in
        **kwargs will be informed and ignored.

        Args:
            config (Union[dict, DataclassConfig]):
                The configuration.

        Returns:
            DataclassConfig:
                The new config instance.
        """
        config = config or {}
        if isinstance(config, cls):
            config_kwargs = config.to_dict(filt_type="orig")
        else:
            config_kwargs = copy.deepcopy(config)

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


def normalize_dict(data: Union[dict, DataclassConfig]):
    if isinstance(data, DataclassConfig):
        return data.to_dict()
    else:
        if (is_dataclass(data) and not isinstance(data, type)) or isinstance(
            data, collections.abc.Mapping
        ):
            return asdict_filt(data, filt_type="orig")
        else:
            raise ValueError(
                f"Except type of dict or dataclass, but got {type(data)!r}:{data}"
            )
