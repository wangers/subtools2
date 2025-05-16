# -*- coding:utf-8 -*-
# Copyright xmuspeech (Author: Leo 2023-08)

import collections
from dataclasses import dataclass, field
from functools import lru_cache
from pathlib import Path
from typing import TYPE_CHECKING, Any, ClassVar, Dict, List, Optional, Union

from egrecho.core.config import DataclassConfig, GenericFileMixin
from egrecho.utils.constants import DATASET_META_FILENAME, DEFAULT_DATA_FILES
from egrecho.utils.data_utils import ClassLabel, Split
from egrecho.utils.io import (
    DataFilesDict,
    SerializationFn,
    sanitize_patterns,
    yaml_load_string,
)
from egrecho.utils.logging import get_logger
from egrecho.utils.misc import ConfigurationException

logger = get_logger()
if TYPE_CHECKING:
    from torch.utils.data import Dataset


@dataclass
class DataBuilderConfig(DataclassConfig):
    """
    Base class for :class:`DataBuilder` configuration.

    Args:
        data_dir (Optional[Union[str, Path]]):
            Path (e.g. ``"./data"``) directory have data files.
        file_patterns (Optional[Union[str, List[str], Dict[str, str]]]):
            str(s) to source data file(s), support pattern matching.
            e.g., ``"egs.train.csv"`` or ``"egs.*.csv"``. More over, with an absolute
            path pattern (e.g., ``"/export_path/egs.train.csv"``), it will invalids ``data_dir`` and
            search files in abs path.
    """

    yaml_inline_list: ClassVar[bool] = False
    data_dir: Optional[str] = field(default=None, metadata={"to_dict": False})
    file_patterns: Optional[Union[str, List[str], Dict[str, str]]] = field(
        default_factory=lambda: DEFAULT_DATA_FILES
    )

    def __post_init__(self):
        # parse str from cli
        if isinstance(self.file_patterns, str):
            self.file_patterns = yaml_load_string(self.file_patterns)


class DataBuilder(GenericFileMixin):
    """
    Base builder class for building dataset.

    The subclass should define a class attribute ``CONFIG_CLS`` that extends arguments,
    and the configuration class name should have an additional ``"Config"`` suffix.
    Subclasses should implement the dataset setup method which is necessary:

    - :meth:`train_dataset`
    - :meth:`val_dataset`
    - :meth:`test_dataset`

    Its instance stores config instance, data filenamess of splits, infos, etc.
    The :func:`build_dataset` function returns either a single split dataset
    or a dict of split datasets according to data files dict.

    NOTE:
        In case of overheading dataset building in your procedure, you can use the warpper
        ``functools.lru_cache`` as ``def _get_data_files(self):`` style.

        Keep in mind that that cached result won't changed
        if you modify the related data files in this instance. You can use
        ``def from_config(...)`` to get a new instance.
    """

    CONFIG_CLS: DataBuilderConfig

    def __init__(self, config: DataBuilderConfig):
        if not isinstance(config, self.CONFIG_CLS):
            raise ValueError(
                f"Parameter config in `{self.__class__.__name__}(config)` "
                f"should be an {self.CONFIG_CLS.__name__} instance, "
                f"but got {type(config)!r}. Try "
                f"`{self.__class__.__name__}.from_config(CONFIG_PATH, DATA_DIR, FILE_PATTERNS)`"
            )

        self._data_dir = (
            Path(config.data_dir).resolve()
            if config.data_dir is not None
            else Path().resolve()
        )
        self._config = config
        logger.info(
            f"Initiate builder: ({type(self).__qualname__}) with config ({type(config).__name__}): \n"
            f"{config.simply_repr()}base data_dir: ({self._data_dir})",
            ranks=[0],
        )

        data_dict = self._get_data_files()  # check file_patterns.
        for key in data_dict.keys():
            if key not in Split.names():
                raise ConfigurationException(
                    f"Split keys should in {Split.names()},"
                    f"But got an invalid ({key}) in data dict keys ({data_dict.keys()})."
                    f"Try to fix your file_patterns: ({config.file_patterns})"
                )

        dataset_meta_file = self.data_dir / DATASET_META_FILENAME
        self._dataset_meta = (
            SerializationFn.load_file(dataset_meta_file)
            if dataset_meta_file.is_file()
            else None
        )

    @classmethod
    def fetch_from(
        cls,
        path: Optional[str] = None,
        data_dir: Optional[str] = None,
        file_patterns: Optional[Union[str, List[str], Dict[str, str]]] = None,
        **kwargs,
    ) -> "DataBuilder":
        if Path(path).is_file():
            config = cls.load_cfg_file(path)
        else:
            raise FileExistsError(f"{path} missing.")
        return cls.from_config(
            config,
            data_dir=data_dir,
            file_patterns=file_patterns,
            **kwargs,
        )

    @classmethod
    def from_config(
        cls,
        config: Optional[Union[dict, DataBuilderConfig]] = None,
        data_dir: Optional[str] = None,
        file_patterns: Optional[Union[str, List[str], Dict]] = None,
        **kwargs,
    ) -> "DataBuilder":
        r"""
        Creates a new :class:`DataBuilder` instance by providing a configuration
        in the form of a dictionary or a instance if :class:`DataBuilderConfig`. All params after
        ``config`` will overwrite it.

        Args:
            config (Optional[Union[dict, DataBuilderConfig]]):
                A dict or an instance of :class:`DataBuilderConfig`.
            data_dir (Optional[str]):
                Path (e.g. ``"./data"``) directory have data files.
            file_patterns (Optional[Union[str, List[str], Dict]]):
                Str(s) of source data file(s), support pattern matching.
                (e.g., ``"egs.train.csv"`` or ``"egs.*.csv"``.) Moreover, with an absolute
                path pattern (e.g., ``"/export_path/egs.train.csv"``), it will invalids ``data_dir`` and
                search files in that abs path.
            \**kwargs (additional keyword arguments):
                Arguments to override config.

        Returns:
            DataBuilder: The new Databuilder instance.
        """
        if isinstance(config, cls.CONFIG_CLS):
            config = config
        elif isinstance(config, collections.abc.Mapping):
            config = dict(config) if config else {}
        else:
            raise TypeError(
                f"Parameter config should be an instance of {cls.CONFIG_CLS.__name__} or dict. "
                f"but got {type(config)}: {config}."
            )
        if data_dir is not None:
            kwargs["data_dir"] = data_dir
        if file_patterns is not None:
            kwargs["file_patterns"] = file_patterns

        ret_config = cls.CONFIG_CLS.from_config(config, **kwargs)
        logger.debug(f"Initiate builder: ({cls.__name__}) with config: {ret_config}.")
        return cls(ret_config)

    def save_config(self, path: str):
        """
        save the configuration to a file.

        Args:
            path (Union[Path, str]):
                The path of the output file.
        """

        self.config.to_file(path)

    def dump_config(self) -> Dict[str, Any]:
        """
        Dump the configuration to a dict.
        """
        self.config.to_dict()

    def build_dataset(
        self,
        split: Optional[Union[str, Split]] = None,
    ) -> Union["Dataset", Dict[str, "Dataset"]]:
        """
        Build dataset.

        Args:
            split (Optional[Union[str, Split]]):
                If None, returns all splits in dict, else specified split.

        Returns:
            Union[IterableDataset, Dict[str, IterableDataset]]:
                The constructed datapipe(s).
        """
        if split is None:
            datasets = {}
            splits = list(self.data_files.keys())
            if str(Split.TRAIN) in splits:
                datasets[str(Split.TRAIN)] = self.build_single_dataset(Split.TRAIN)
            if str(Split.VALIDATION) in splits:
                datasets[str(Split.VALIDATION)] = self.build_single_dataset(
                    Split.VALIDATION
                )
            if str(Split.TEST) in splits:
                datasets[str(Split.TEST)] = self.build_single_dataset(Split.TEST)
            return datasets
        else:
            return self.build_single_dataset(split)

    def build_single_dataset(
        self,
        split: Optional[Union[str, Split]] = Split.TRAIN,
    ) -> "Dataset":
        """
        Function to build single split datapipe.

        Args:
            split (Optional[Union[str, Split]]):
                The split name.

        Returns:
            IterableDataset:
                The constructed data pipe.
        """
        if split == Split.TRAIN:
            return self.train_dataset()
        elif split == Split.VALIDATION:
            return self.val_dataset()
        elif split == Split.TEST:
            return self.test_dataset()
        else:
            raise ConfigurationException(
                f"split name should one of {Split.names()}, but got {split}"
            )

    def train_dataset(self):
        raise NotImplementedError

    def val_dataset(self):
        raise NotImplementedError

    def test_dataset(self):
        raise NotImplementedError

    # alias
    @property
    def config(self):
        return self._config

    @property
    def data_dir(self):
        return self._data_dir

    @property
    def train_data_files(self):
        return self.data_files.get(Split.TRAIN, None)

    @property
    def val_data_files(self):
        return self.data_files.get(Split.VALIDATION, None)

    @property
    def test_data_files(self):
        return self.data_files.get(Split.TEST, None)

    @property
    def data_files(self) -> DataFilesDict:
        """
        Property method for returning data files information.

        Returns:
            DataFilesDict:
                The data files can be find by split key.
        """
        return self._get_data_files()

    @lru_cache()
    def _get_data_files(self):
        try:
            return DataFilesDict.from_local_or_remote(
                sanitize_patterns(self.config.file_patterns), base_path=self.data_dir
            )
        except FileNotFoundError as e:
            raise ConfigurationException(
                f"Failed match file pattens {self.config.file_patterns} in dir: {self.data_dir}\n{e}"
            )

    def estimate_length(self):
        """estimate dataset length."""
        raise ConfigurationException(
            "This method should be implemented in the derived class if needed."
        )

    @property
    def num_classes(self) -> Optional[int]:
        """Property that returns the number of classes if it is a multiclass task."""

        return self.class_label.num_classes if self.class_label else None

    @property
    def class_label(self) -> ClassLabel:
        """Property that returns the labels. Should be implemented in the derived class if needed."""
        return None

    @property
    def feature_extractor(self):
        """Property that returns the feature extractor. Should be implemented in the derived class if needed."""
        return None

    @property
    def feature_size(self) -> int:
        """Property that returns the feat dim."""
        return self.feature_extractor.feature_size if self.feature_extractor else None

    @property
    def inputs_dim(self):
        """Property that returns the inputs_dim for downstream model. Should be implemented in the derived class if needed."""
        return None

    @property
    def dataset_meta(self):
        return self._dataset_meta
