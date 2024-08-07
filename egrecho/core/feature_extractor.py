# -*- coding:utf-8 -*-
# Copyright xmuspeech (Author: Leo 2023-03-11)

import copy
from pathlib import Path
from typing import Any, Dict, Union

import egrecho.utils.constants as constants
from egrecho.utils.common import SaveLoadMixin
from egrecho.utils.io import ConfigFileMixin, is_remote_url, repr_dict
from egrecho.utils.logging import get_logger
from egrecho.utils.misc import ConfigurationException

logger = get_logger()


class BaseFeature(ConfigFileMixin, SaveLoadMixin):
    r"""
    A base class offers serialize methods for feature extractor.

    The implementation of the extraction method is intended for derived classes.
    Its purpose is to align with open-source pretrained models and facilitate
    coordination between model inputs and the frontend data processor.
    Consequently, it is a simplified adaptation of
    the feature_extractor module in `huggingface extractor
    <https://huggingface.co/docs/transformers/main_classes/feature_extractor>`_.
    """

    def __init__(self, **kwargs):
        """Set elements of ``kwargs`` as attributes."""

        # Additional attributes without default values
        for key, value in kwargs.items():
            try:
                setattr(self, key, value)
            except AttributeError as e:
                logger.error(f"Can't set {key} with value {value} for {self}")
                raise e

    @classmethod
    def from_dict(
        cls, feature_extractor_dict: Dict[str, Any], **kwargs
    ) -> "BaseFeature":
        """
        Instantiates a feature extractor from a Python dictionary of parameters.

        Args:
            feature_extractor_dict:
                Dictionary that will be used to instantiate the feature extractor object.
            \**kwargs:
                overwrite kwargs.

        Returns:
            The instantiated feature extractor object.
        """

        feature_extractor = cls(**feature_extractor_dict)

        # Update feature_extractor with kwargs if needed
        override = []
        for key, value in kwargs.items():
            if hasattr(feature_extractor, key):
                setattr(feature_extractor, key, value)
                override.append(key)
        for key in override:
            kwargs.pop(key, None)

        override_info = f"\noverride_args {override}." if override else ""
        unused_info = f"\nunused_args {kwargs}." if kwargs else ""

        logger.info_once(
            f"Feature extractor {feature_extractor}" + override_info + unused_info,
            ranks=[0],
        )

        return feature_extractor

    def to_dict(self) -> Dict[str, Any]:
        """
        Serializes this instance to a Python dictionary.

        Returns:
            Dict[str, Any].
        """

        output = copy.deepcopy(self.__dict__)
        output["feature_extractor_type"] = self.__class__.__name__
        return output

    @classmethod
    def fetch_from(
        cls,
        srcdir: Union[str, Path],
        **kwargs,
    ) -> "BaseFeature":
        if is_remote_url(srcdir):
            raise NotImplementedError("TO DO, support remote file.")
        return cls.create_extractor(srcdir, **kwargs)

    def save_to(
        self,
        savedir,
        **kwargs,
    ):
        savedir = Path(savedir)
        if savedir.is_file():
            raise ConfigurationException(
                f"Provided path ({savedir}) should be a directory, not a file."
            )
        savedir.mkdir(parents=True, exist_ok=True)
        cfg_fname = kwargs.pop("config_fname", constants.DEFAULT_EXTRACTOR_FILENAME)
        cfg_file = savedir / str(cfg_fname)
        self.to_cfg_file(cfg_file)

    @classmethod
    def create_extractor(cls, path: Union[str, Path], **kwargs):
        extractor_dict, kwargs = cls.get_extractor_dict(path, **kwargs)
        return cls.from_dict(extractor_dict, **kwargs)

    @classmethod
    def get_extractor_dict(cls, path: Union[str, Path], **kwargs):
        """
        Get config file from dir/file path.
        """
        path = Path(path)
        if path.is_dir():
            feature_extractor_file = kwargs.pop(
                "config_fname", constants.DEFAULT_EXTRACTOR_FILENAME
            )
            feature_extractor_file = path / constants.DEFAULT_EXTRACTOR_FILENAME
        else:
            feature_extractor_file = path
        extractor_dict = cls.load_cfg_file(feature_extractor_file)
        return extractor_dict, kwargs

    def __repr__(self):
        return f"{self.__class__.__name__}\n{repr_dict(self.to_dict())}"
