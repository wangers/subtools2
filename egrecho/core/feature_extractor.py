# -*- coding:utf-8 -*-
# Copyright xmuspeech (Author: Leo 2023-03-11)

import copy
from pathlib import Path
from typing import Any, Dict, Union

import egrecho.utils.constants as constants
from egrecho.utils.io import ConfigFileMixin, repr_dict
from egrecho.utils.logging import get_logger

logger = get_logger()


class BaseFeature(ConfigFileMixin):
    r"""
    A base class offers serialize methods for acoustic feature extractor.

    The implementation of the extraction method is intended for derived classes.
    Its purpose is to align with open-source pretrained models and facilitate
    coordination between model inputs and the frontend data processor.
    Consequently, this structure is a simplified adaptation of
    the `feature_extractor` module found in `huggingface.transformers`.
    For more in-depth insight, please refer to:
        https://huggingface.co/docs/transformers/main_classes/feature_extractor
    """

    def __init__(self, **kwargs):
        """Set elements of `kwargs` as attributes."""

        # Additional attributes without default values
        for key, value in kwargs.items():
            try:
                setattr(self, key, value)
            except AttributeError as e:
                logger.error(f"Can't set {key} with value {value} for {self}")
                raise e

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
            feature_extractor_file = path / constants.DEFAULT_EXTRACTOR_FILENAME
        extractor_dict = cls.load_cfg_file(feature_extractor_file)
        return extractor_dict, kwargs

    @classmethod
    def from_dict(
        cls, feature_extractor_dict: Dict[str, Any], **kwargs
    ) -> "BaseFeature":
        """
        Instantiates a feature extractor from a Python dictionary of parameters.

        Args:
            feature_extractor_dict (`Dict[str, Any]`):
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
            `Dict[str, Any]`.
        """

        output = copy.deepcopy(self.__dict__)
        output["feature_extractor_type"] = self.__class__.__name__
        return output

    def __repr__(self):
        return f"{self.__class__.__name__}\n{repr_dict(self.to_dict())}"
