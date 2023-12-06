import copy
import inspect
import re
from abc import ABC, abstractmethod
from dataclasses import InitVar, dataclass, field
from functools import partial
from itertools import chain
from pathlib import Path
from typing import TYPE_CHECKING, Dict, List, Optional, Sequence, Type, Union

import egrecho.utils.constants as constants
from egrecho.core.config import DataclassConfig
from egrecho.data.builder.voyages import VoyageTemplate
from egrecho.utils.common import (
    DataclassSerialMixin,
    alt_none,
    asdict_filt,
    field_dict,
    get_diff_dict,
)
from egrecho.utils.data_utils import Split, SplitInfo
from egrecho.utils.io import (
    DataFilesDict,
    JsonMixin,
    SerializationFn,
    is_relative_path,
    load_json,
    repr_dict,
    sanitize_patterns,
    yaml_load_stream,
)
from egrecho.utils.logging import get_logger
from egrecho.utils.misc import ConfigurationException

logger = get_logger()
if TYPE_CHECKING:
    from torch.utils.data import Dataset

field_init_dict = partial(field_dict, init_field_only=True)

from egrecho.utils.common import rich_exception_info


# flake8:noqa
@dataclass
class DataPipeInfo(JsonMixin):
    """
    Infos about dataset.
    """

    description: str = field(default_factory=str)
    voyage_task: Optional[VoyageTemplate] = None
    class_label: Optional[VoyageTemplate] = None
    total_info: Optional[Dict] = None

    def copy(self) -> "DataPipeInfo":
        return self.__class__(**{k: copy.deepcopy(v) for k, v in self.__dict__.items()})


class DewPipeInfoMixin:
    """
    Compose of global-wise info `DewPipeInfo` and split-wise info `SplitInfo`.
    Make `DewPipe` easier to access infos.
    """

    def __init__(self, dew_pipe_info: DataPipeInfo, split_info: SplitInfo):
        self._dew_pipe_info = dew_pipe_info
        self._split_info = split_info

    @property
    def voyage_task(self):
        return self._dew_pipe_info.voyage_task

    @property
    def class_label(self):
        return self._dew_pipe_info.class_label

    @property
    def total_info(self):
        return self._dew_pipe_info.total_info

    @property
    def split_info(self):
        return self._split_info.info

    @property
    def split_name(self):
        return self._split_info.name


def _no_op(x, **kwargs):
    return x


# class DewPipe(DewPipeInfoMixin, IterDataPipe):
#     """
#     This wrapper of torch `IterDataPipe`, and attach infos of datasets.
#     """
#     def __init__(
#         self,
#         data_pipe: IterDataPipe,
#         pipe_line: Optional[Callable] = None,
#         dew_pipe_info: Optional[DewPipeInfo] = None,
#         split_info: Optional[SplitInfo] = None,
#         **kwargs,
#     ):
#         dew_pipe_info = dew_pipe_info.copy() if dew_pipe_info is not None else DewPipeInfo()
#         split_info = split_info.copy() if split_info is not None else SplitInfo()
#         DewPipeInfoMixin.__init__(self, dew_pipe_info=dew_pipe_info, split_info=split_info)
#         pipe_line = alt_none(pipe_line, _no_op)
#         self._data = pipe_line(data_pipe, **kwargs)

#     def __iter__(self) -> Iterator:
#         yield from self.data

#     @property
#     def data(self) -> IterDataPipe:
#         return self._data

#     def __len__(self):
#         return len(self.data)

#     def head(self, n: int = 2):
#         from torchdata.datapipes.iter import Header
#         dew_pipe = copy.deepcopy(self)
#         return list(Header(dew_pipe.data, limit=n))

#     def __repr__(self):
#         try:
#             len_val = len(self)
#         except:
#             len_val = "<unknown>"

#         return f"<class DewPipe> (len={len_val}) \n"


@dataclass
class DataBuilderConfig(DataclassConfig, DataclassSerialMixin):
    """
    Base class for `PipeBuilder` configuration.

    Args:
        data_dir (Optional[Union[str, Path]]):
            Path (e.g. `"./data"`) directory have data files.
        file_patterns (Optional[Union[str, List[str], Dict]]):
            str(s) to source data file(s), support pattern matching.
            e.g., `"egs.train.csv"` or `"egs.*.csv"`. More over, with an absolute
            path pattern (e.g., `"/export_path/egs.train.csv"`), it will invalids `data_dir` and
            search files in abs path.
        split_kwargs (Optional[Dict[Union[str, Split], Dict]]):
            Arguments specified for single split dataset.
            (e.g., {"train": {KEY: VALUE}}, the (KEY, VALUE) appears in derived subclasses.)
    """

    split_name: str = field(default=Split.TRAIN, metadata={'encoding_fn': str})
    file_patterns: Optional[Union[str, List[str]]] = field(
        default=None, metadata={'cmd': True}
    )

    def __post_init__(self):
        self.file_patterns = alt_none(
            self.file_patterns, constants.DEFAULT_FILES[self.split_name]
        )

    @classmethod
    def from_config(
        cls,
        config: Union[dict, "DataBuilderConfig"] = None,
        strict: bool = False,
        **kwargs,
    ):
        """
        Create an new instance from config.

        Input `config` can be an instance or a dict, the invalid overwrite args in
        **kwargs will be informed and ignored.

        Args:
            config (Union[dict, DataBuilderConfig]):
                The configuration.

        Returns:
            DataBuilderConfig:
                The new config instance.
        """
        config = alt_none(config, {})
        if isinstance(config, cls):
            config_kwargs = config.to_dict(filt_type='orig')
        else:
            config_kwargs = copy.deepcopy(config)

        config_kwargs.update(kwargs)
        valid_keys = field_init_dict(cls).keys()
        invalid_keys = config_kwargs.keys() - valid_keys
        invalid_kwargs = {}
        for key in invalid_keys:
            invalid_kwargs[key] = config_kwargs.pop(key, None)
        msg = f'Get invalid kwargs:{invalid_kwargs}'
        if invalid_kwargs and strict:
            raise ConfigurationException(f'{msg}.')
        elif invalid_kwargs:
            logger.info(f'Get invalid kwargs:{invalid_kwargs}, ignore them.')
        else:
            pass
        return cls.from_dict(config_kwargs)


class BuilderConfigDict(dict):
    def __init__(self, *args, split_name=None, **kwargs):
        super().__init__(*args, **kwargs)
        self.split_name = split_name

    def __setitem__(self, key: str, value: DataBuilderConfig):
        if key != value.split_name:
            raise ValueError(
                f"Cannot add elem. (key mismatch: '{key}' != '{value.split_name}')"
            )
        if key in self:
            raise ValueError(f"BuilderConfig {key} already present")
        super().__setitem__(key, value)

    def add(self, config: DataBuilderConfig):
        """Add a config."""
        if config.split_name in self:
            raise ValueError(f"BuilderConfig {config.split_name} already present")
        super().__setitem__(config.split_name, config)

    @classmethod
    def from_split_dict(cls, configs: Union[List, Dict]):
        """Returns a new BuilderConfigDict initialized from a Dict or List of `DataBuilderConfig`."""
        if isinstance(configs, dict):
            configs = list(configs.values())

        config_dict = cls()

        for config in configs:
            if not isinstance(config, DataBuilderConfig):
                raise ValueError(
                    f'Need to recieve type: {DataBuilderConfig.__name__}, but got type:{type(config).__name__}: {config}.'
                )
            config_dict.add(config)

        return config_dict

    def to_split_dict(self) -> List[DataBuilderConfig]:
        out = []
        for split_name, config in self.items():
            config = copy.deepcopy(config)
            config.split_name = split_name
            out.append(config)
        return out

    def copy(self):
        return BuilderConfigDict.from_split_dict(self.to_split_dict())

    def simply_repr(self):
        repr_str = repr_dict(asdict_filt(dict(self), filt_type='default'))
        # repr_str = re.sub(r"^", " " * 2, repr_str, 0, re.M)
        return repr_str


class DataBuilder(ABC):
    """
    Base builder class for building datapipes.

    The class attribute `CONFIG_CLS` in subclass can extend args,
    and the config class name is set with an extra `Config` string.
    Subclasses should implement the key function:
        `build_single_dataset`: build a split dataset.

    Its instance stores config instance, data files of splits, info, etc.
    and function `build_dataset` returns either a single split datapipe
    or a dict of split dataset, according to its function parameter `split`.
    """

    CONFIG_CLS: DataBuilderConfig

    def __init__(self, config: BuilderConfigDict, data_dir: Optional[str] = None):
        if not all(
            isinstance(flatten_config, self.CONFIG_CLS)
            for flatten_config in config.values()
        ):
            raise ValueError(
                f'Parameter config in `{self.__class__.__name__}(config)` '
                f'should be a dict contains {self.CONFIG_CLS.__name__} instances, Try '
                f'`{self.__class__.__name__}.from_config(CONFIG_PATH, DATA_DIR)`'
            )
        logger.info(
            f"Initiate builder: ({type(self).__qualname__}) with config dict: \n"
            # f"{repr_dict(asdict_filt(dict(config), filt_type='default'))}base data_dir: ({data_dir})",
            f"{config.simply_repr()}base data_dir: ({data_dir})",
            ranks=[0],
        )

        self._configs = config
        self._data_dir = data_dir
        file_patterns = self._get_file_patterns()
        self._get_data_files()  # check file_patterns.

        # all data files in one data_dir.
        if (
            all(
                is_relative_path(file_pattern)
                for file_pattern in chain(file_patterns.values())
            )
            and data_dir
        ):
            dataset_meta_file = Path(data_dir) / constants.DATASET_META_FILENAME
            self._dataset_meta = (
                load_json(dataset_meta_file) if dataset_meta_file.is_file() else None
            )

    @classmethod
    def from_file(
        cls,
        path: Union[str, Path] = None,
        data_dir: Optional[str] = None,
    ) -> "DataBuilder":
        if Path(path).is_file():
            config = SerializationFn.load_file(path)
        else:
            raise FileExistsError(f'{path} missing.')
        return cls.from_config(config, data_dir=data_dir)

    @classmethod
    def from_config(
        cls,
        dict_or_list: Optional[Union[Dict, List[Dict], DataBuilderConfig]] = None,
        data_dir: Optional[str] = None,
    ) -> "DataBuilder":
        """
        Create a new `DataBuilder` instance by providing a configuration
        in the form of a dictionary, a list of dictionaries, or a `DataBuilderConfig` instance.

        Args:
            dict_or_list (Optional[Union[Dict, List[Dict], DataBuilderConfig]]):
                The configuration to build the data pipeline. It can be provided as a dictionary,
                a list of dictionaries, or an instance of `DataBuilderConfig`.
            data_dir (Optional[str]):
                Path (e.g. `"./data"`) directory have data files.
            file_patterns (Optional[Union[str, List[str], Dict]]):
                str(s) of source data file(s), support pattern matching.
                e.g., `"egs.train.csv"` or `"egs.*.csv"`. More over, with an absolute
                path pattern (e.g., `"/export_path/egs.train.csv"`), it will invalids `data_dir` and
                search files in abs path.
            **kwargs (additional keyword arguments):
                Arguments to overwrite config.

        Returns:
            DataBuilder: The new Databuilder instance

        Raises:
            ConfigurationException:
                If the `dict_or_list` parameter has an unsupported type.

        Example
        -------
        """
        if isinstance(dict_or_list, Sequence):
            configs = list(dict_or_list)
        elif isinstance(dict_or_list, dict):
            configs = list(dict_or_list.values())
        else:
            raise ConfigurationException(f'Error config type: {type(dict_or_list)}')
        return cls.from_flatten_configs(*configs, data_dir=data_dir)

    @classmethod
    def from_flatten_configs(
        cls,
        *configs: Union[List, Dict, DataBuilderConfig],
        data_dir: Optional[str] = None,
    ) -> "DataBuilder":
        """
        Create a new `DataBuilder` instance from a set of flattened configurations.

        Args:
            *configs (Union[List, Dict, DataBuilderConfig]):
                Variable-length argument list
                containing one or more configurations for `DataBuilder` instances. Each configuration
                can be a list, dictionary, or an instance of `DataBuilderConfig` class.
            data_dir (Optional[str]):
                Path (e.g. `"./data"`) directory have data files.

        Returns:
            DataBuilder: The new Databuilder instance
        """
        if len(configs) == 1:
            if isinstance(configs[0], Sequence):
                configs = list(configs[0])
        merged_config = cls.merge_configs(configs)
        return cls(merged_config, data_dir=data_dir)

    @classmethod
    def merge_configs(
        cls, configs: Sequence[Union[DataBuilderConfig, dict]]
    ) -> BuilderConfigDict:
        """
        Merge a sequence of DataBuilder configurations into a single BuilderConfigDict.

        Args:
            configs (Sequence[Union[DataBuilderConfig, dict]]):
                A sequence containing DataBuilder
                configurations as dictionaries or instances of `DataBuilderConfig`.

        Returns:
            BuilderConfigDict[str, DataBuilderConfig]:
                A merged configuration dictionary containing
                all the split names as keys and `DataBuilderConfig` instances as values.

        Raises:
            ValueError: If any configuration in the sequence is missing the 'split_name' key.
        """
        merged_config = BuilderConfigDict()
        for config in configs:
            if isinstance(config, dict):
                split_name = config.pop("split_name", None)
                if split_name is None:
                    raise ValueError("Each config must have a 'split_name' key.")
            merged_config.add(cls.CONFIG_CLS.from_config(config, strict=True))
        return merged_config

    def dump_config(self, path: Union[Path, str], **kwargs):
        """
        Dump the configuration dict to a file.

        Args:
            path (Union[Path, str]):
                The path of the output file.
        """
        encode_kwargs = kwargs.pop('encode_kwargs', {})
        d = asdict_filt(self.configs, **encode_kwargs)
        SerializationFn.save_file(d, path, **kwargs)

    # alias
    @property
    def configs(self):
        return self._configs

    @property
    def data_dir(self):
        return self._data_dir

    @property
    def data_files(self):
        return self._get_data_files()

    def _get_file_patterns(self):
        file_patterns = {
            split_name: flatten_config.file_patterns
            for split_name, flatten_config in self.configs.items()
        }
        return file_patterns

    def _get_data_files(self):
        file_patterns = self._get_file_patterns()
        return DataFilesDict.from_local_or_remote(
            sanitize_patterns(file_patterns), base_path=self.data_dir
        )

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
            splits = self.configs.keys()
            return {s: self.build_single_dataset(s) for s in splits}
        else:
            return self.build_single_dataset(split)

    @abstractmethod
    def build_single_dataset(
        self,
        split: Optional[Union[str, Split]] = Split.TRAIN,
    ) -> "Dataset":
        """
        Core function to build single split datapipe.

        Args:
            split (Optional[Union[str, Split]]):
                The split name.

        Returns:
            IterableDataset:
                The constructed data pipe.
        """
        raise NotImplementedError

    @property
    def dataset_meta(self):
        return self._dataset_meta

    @classmethod
    def build_dataloader():
        ...
