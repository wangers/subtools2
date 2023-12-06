import copy
import inspect
import re
from abc import ABC, abstractmethod
from dataclasses import InitVar, dataclass, field
from functools import partial
from pathlib import Path
from typing import (
    TYPE_CHECKING,
    Callable,
    ClassVar,
    Dict,
    Iterator,
    List,
    Optional,
    Type,
    Union,
)

import egrecho.utils.constants as constants
from egrecho.core.config import DataclassConfig
from egrecho.data.builder.voyages import VoyageTemplate
from egrecho.utils.logging import get_logger
from egrecho.utils.common import (
    DataclassSerialMixin,
    alt_none,
    asdict_filt,
    field_dict,
    get_diff_dict,
)
from egrecho.utils.data_utils import Split, SplitInfo
from egrecho.utils.io.files import DataFilesDict, DataFilesList, sanitize_patterns
from egrecho.utils.io.utils import ConfigFileMixin, JsonMixin, load_json, repr_dict

logger = get_logger()
if TYPE_CHECKING:
    from torch.utils.data import Dataset

field_init_dict = partial(field_dict, init_field_only=True)


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


class DewPipeBuch(dict):
    """
    A dict to hold `DewPipe`.
    """

    def _check_values(self):
        for dp in self.values():
            if not isinstance(dp, ()):
                raise TypeError(
                    f"Values in `DewPipeBuch` should be of type `DewPipe` but got type '{type(dp)}'"
                )

    @property
    def data(self) -> Dict[str, "IterDataPipe"]:
        self._check_values()
        return {k: dp.data for k, dp in self.items()}

    def head(self, n: int = 2):
        self._check_values()
        return {k: dp.head(n) for k, dp in self.items()}

    def __repr__(self):
        repr = "\n".join([f"{k}: {v}" for k, v in self.items()])
        repr = re.sub(r"^", " " * 4, repr, 0, re.M)
        return f"DatasetDict({{\n{repr}\n}})"


@dataclass
class PipeBuilderConfig(DataclassConfig, DataclassSerialMixin):
    """
    Base class for `PipeBuilder` configuration.

    NOTE: Any subclass have `__post_init__` func should inherit its parent's
    `__post_init__` function as we implemented some arg
    checking in this base config class: (e.g., `split_kwargs`.)

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

    data_dir: Optional[Union[str, Path]] = field(
        default=None, metadata={"cmd": True, "to_dict": False}
    )
    file_patterns: Optional[Union[str, List[str], Dict]] = field(
        default=None, metadata={"cmd": True}
    )
    split_kwargs: Optional[Dict[Union[str, Split], Dict]] = field(default_factory=dict)

    def __post_init__(self):
        self.file_patterns = alt_none(self.file_patterns, constants.DEFAULT_FILES)
        self._check_split_kwargs(self.split_kwargs)

    @classmethod
    def _check_split_kwargs(cls, split_kwargs):
        """
        Check the validity of `split_kwargs`.

        Args:
            split_kwargs (dict): The split arguments to be checked.
        """
        if not isinstance(split_kwargs, dict):
            raise ValueError(
                f"Parameter `split_kwargs` of {cls.__name__} should be type of dict,"
                f" but got {type(split_kwargs)}: {split_kwargs}."
            )
        error_keys = field_init_dict(PipeBuilderConfig).keys()
        valid_keys = field_init_dict(cls).keys() - error_keys
        for split, kwargs in split_kwargs:
            assert isinstance(kwargs, dict), kwargs
            for key in kwargs:
                if key in error_keys:
                    raise KeyError(
                        f"Split part: ({split}) in parameter `split_kwargs` "
                        f"of {cls.__name__} has error key: {key},"
                        " which should be set outside for global."
                    )
                if key not in valid_keys:
                    raise KeyError(
                        f"Split part: ({split}) in parameter `split_kwargs` "
                        f"of {cls.__name__} has invalid key: {key},"
                    )

    def spawn_split_config(self, split: Union[str, Split]) -> "PipeBuilderConfig":
        """
        Spawn a new config instance according to `self.split_kwargs`.

        Args:
            split (Union[str, Split]):
                The split name.

        Returns:
            PipeBuilderConfig:
                A new config instance or a copy of itself.
        """
        if split not in self.split_kwargs:
            return copy.deepcopy(self)
        else:
            spwan_dict = self.to_dict(filt_type="orig").update(self.split_kwargs[split])
            return self.from_dict(spwan_dict)

    def combine_split_config(
        self, other: "PipeBuilderConfig", split: Split
    ) -> "PipeBuilderConfig":
        """
        Combine another config instance.

        This is used to dump split config to a target one,
        gathers the different configuration into `self.split_kwargs`.

        NOTE: global args (`data_dir`, 'file_patterns') are ignored.

        Args:
            other (PipeBuilderConfig):
                The other config instance to be combined.
            split (Split):
                The split name.

        Raises:
            ValueError:
                If the `other` parameter has an invalid type.

        Returns:
            PipeBuilderConfig:
                The updated config instance.
        """
        if not isinstance(other, type(self)):
            raise ValueError(f"The config {other} has invalid type: {type(other)}")
        src_dict = self.to_dict()
        curr_dict = other.to_dict()
        curr_dict.pop("split_kwargs", None)
        diff_dict = get_diff_dict(src_dict, curr_dict)

        if diff_dict:
            self.split_kwargs[split] = diff_dict
        return self

    @classmethod
    def from_config(cls, config: Union[dict, "PipeBuilderConfig"] = None, **kwargs):
        """
        Create an instance from config.

        Input `config` can be an instance or a dict, the invalid overwrite args in
        **kwargs will be informed and ignored.

        Args:
            config (Union[dict, PipeBuilderConfig]):
                The configuration.

        Returns:
            PipeBuilderConfig:
                The new config instance.
        """
        config = alt_none(config, {})
        if isinstance(config, cls):
            config_kwargs = config.to_dict(filt_type="orig")
        else:
            config_kwargs = copy.deepcopy(config)
        # for key, value in kwargs.items():
        #     if value is not None:
        #         if key not in config_kwargs or config_kwargs[key] != value:
        #             config_kwargs[key] = value
        config_kwargs.update(kwargs)
        valid_keys = field_init_dict(cls).keys()
        invalid_keys = config_kwargs.keys() - valid_keys
        invalid_kwargs = {}
        for key in invalid_keys:
            invalid_kwargs[key] = config_kwargs.pop(key, None)
        if invalid_kwargs:
            logger.info(f"Get invalid kwargs:{invalid_kwargs}, ignore them.")
        return cls.from_dict(config_kwargs)

    # @classmethod
    # def from_dict(cls, data: dict):
    #     data = copy.deepcopy(data)
    #     return cls(**data)

    # def to_dict(self) -> dict:
    #     data = self.args_overrided()
    #     try:
    #         if not data['split_kwargs']:  # for empty dict `{}` case.
    #             data.pop('split_kwargs')
    #     except KeyError:
    #         pass
    #     data.pop('data_dir', None)  # skip dump data_dir & file_patterns.
    #     data.pop('file_patterns', None)
    #     return data


class PipeBuilder(ABC):
    """
    Base builder class for building datapipes.

    The class attribute `CONFIG_CLS` in subclass can extend args,
    and the config class name is set with an extra `Config` string.
    Subclasses should implement the key function:
        `build_single_datapipe`: build datapipe of a split dataset.

    Its instance stores config instance, data files of splits, info, etc.
    and function `build_datapipe` returns either a single split datapipe
    or a dict of split datapipes according to data files dict.
    """

    CONFIG_CLS = PipeBuilderConfig

    def __init__(self, config: PipeBuilderConfig):
        if not isinstance(config, self.CONFIG_CLS):
            raise ValueError(
                f"Parameter config in `{self.__class__.__name__}(config)` "
                f"should be an {self.CONFIG_CLS.__name__} instance, "
                f"but got {type(config)}. Try "
                f"`{self.__class__.__name__}.from_config(CONFIG_PATH, DATA_DIR, FILE_PATTERNS)`"
            )
        DataFilesDict.from_local_or_remote(
            sanitize_patterns(config.file_patterns), base_path=config.data_dir
        )  # check file_patterns.
        self.data_dir = (
            Path(config.data_dir) if config.data_dir is not None else Path().resolve()
        )
        dataset_meta_file = self.data_dir / constants.DATASET_META_FILENAME
        self.config = config

        self._dataset_meta = (
            load_json(dataset_meta_file) if dataset_meta_file.is_file() else None
        )

    @classmethod
    def from_config(
        cls,
        path_or_obj: Optional[Union[str, Path, PipeBuilderConfig]] = None,
        data_dir: Optional[str] = None,
        file_patterns: Optional[Union[str, List[str], Dict]] = None,
        **kwargs,
    ):
        """
        Load a `PipeBuilder` instance.

        First get an instance of `cls.CONFIG_CLS` by passing args `path_or_obj`.
        It can be a path of `json/yaml` file or a directly instance. Then
        a new builder is initiated by this config instance.

        Args:
            path_or_obj (Optional[Union[str, Path, PipeBuilderConfig]]):
                The path or instance of `PipeBuilderConfig`.
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
            PipeBuilder: The new PipeBuilder instance.

        Example
        -------
        """
        if isinstance(path_or_obj, cls.CONFIG_CLS):
            config = path_or_obj
        else:
            config = cls.CONFIG_CLS.load_cfg_file(path_or_obj) if path_or_obj else {}
        if data_dir is not None:
            kwargs["data_dir"] = data_dir
        if file_patterns is not None:
            kwargs["file_patterns"] = file_patterns

        ret_config = cls.CONFIG_CLS.from_config(config, **kwargs)
        logger.info(f"Initiate builder: ({cls.__name__}) with config: {ret_config}.")
        return cls(ret_config)

    def update(self, **kwargs):
        """
        Update builder with kwargs on fly.

        Args:
            **kwargs:
                Arguments to update the builder.

        Returns:
            PipeBuilder:
                The updated PipeBuilder instance.
        """
        config = kwargs.pop("config", None)
        config = copy.deepcopy(alt_none(config, self.config))
        return self.from_config(config, **kwargs)

    def _spawn_splits_configs(self) -> Dict[str, PipeBuilderConfig]:
        """
        Spawn split configurations.

        Returns:
            Dict:
                A dictionary of split configurations.
        """
        configs = {}
        for split in self.data_files:
            configs[split] = self.config.spawn_split_config(split)
        return configs

    # alias
    @property
    def configs(self):
        return self._spawn_splits_configs()

    def dump_config(self, path: Union[Path, str]):
        """
        Dump the configuration to a file.

        Args:
            path (Union[Path, str]):
                The path of the output file.
        """
        self.config.to_cfg_file(path)

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
            splits = self.data_files.keys()
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
    def data_files(self) -> DataFilesDict:
        """
        Property method for returning data files information.

        Returns:
            DataFilesDict:
                The data files can be find by split key.
        """
        return DataFilesDict.from_local_or_remote(
            sanitize_patterns(self.config.file_patterns), base_path=self.config.data_dir
        )

    @property
    def dataset_meta(self):
        return self._dataset_meta

    @classmethod
    def build_dataloader():
        ...


def resolve_ckpt1(
    checkpoint: str = "last.ckpt",
    dirpath: Optional[str] = None,
):
    """Resolve checkpoint from local or remote.

    Args:
        checkpoint (str):
            can be either:

        - remote url (e.g., startwith "http"): return it directly, otherwise change to local mode.
        - absolute file path: return it if exists, otherwise raise a FileExistError
        - relative file name: rel to `dirpath` and resolve it locally

    """
    ...


def resolve_ckpt_local(
    dirpath: Optional[str] = None,
    checkpoint: str = "last.ckpt",
    version: str = "version",
    best_k_fname: str = constants.BEST_K_MAP_FNAME,
    best_k_mode: Literal["max", "min"] = "min",
    ckpt_subdir: str = constants.CHECKPOINT_DIR_NAME,
) -> str:
    """Resolve checkpoint path from local fs.

    Automatically search checkpoint in a directory's checkpoints subdir, normally names as `checkpoints`.
    The `dirpath` may has such default structure::

        ./dirpath/
        ├── hparams.yaml
        └── checkpoints
            ├── best_k_models.yaml
            ├── last.ckpt
            └── abc.ckpt

        or

        ./dirpath/version_1
                ├── hparams.yaml
                └── checkpoints
                    ├── best_k_models.yaml
                    ├── last.ckpt
                    └── abc.ckpt

    First search `checkpoints` subdir in dirpath or its `version` subdir, then match the valid checpoint path
    in `checkpoints`.

    Args:
        dirpath (Path or str, optional):
            The root path. Defaults to None, which means the current directory.
        checkpoint (str, optional):
            The file name of checkpoint to resolve, needs a suffix like ".ckpt/.pt",
            While checkpoint="best" is a preseved key means it will find `best_k_fname` which is
            a file contains `Dict[BEST_K_MODEL_PATH, BEST_K_SCORE]`, and sort by its score to
            match a best ckpt. Defaults to "last.ckpt".
        version (str, optional):
            The versioned subdir name. Conmmonly subdir is named as "version_0/version_1", if you specify
            the version name with a version num, it will search that version dir, otherwise choose the max number
            of version (above "version_1"). Defaults to "version".
        best_k_fname (str, optional):
            The filename for the best_k map file. Note that the best model path in best map file may
            not in this directory since it is stored in training stage, so we assume that its basename
            can matching ckpts in the same level. Defaults to best_k_models.yaml.
        best_k_mode (Literal["max", "min"], optional):
            The mode for selecting the best_k checkpoint. Defaults to "min".
        ckpt_subdir (str, optional):
            The name of the checkpoints subdir. Defaults to "checkpoints".


    Returns:
        str: The resolved checkpoint path.

    Examples:
        >>> resolve_ckpt_local('./dirpath', checkpoint='best')
        '/path/to/xxxl.ckpt'

    """
    dirpath = str(dirpath) if dirpath is not None else str(Path().resolve())
    if not Path(dirpath).is_dir():
        raise ValueError(f"dirpath=({dirpath}) is not a dir.")
    dirpath = _search_vallid_dir(dirpath, ckpt_subdir=ckpt_subdir, version=version)
    checkpoint_dir = Path(dirpath) / ckpt_subdir
    assert checkpoint_dir.is_dir()
    if checkpoint == "best":
        best_k_fpath = checkpoint_dir / best_k_fname
        if not best_k_fpath.is_file():
            raise FileExistsError(
                f"Set checkpoint='best' to find best model needs best_k map file but failed to resolve ({best_k_fname}) in ({best_k_fpath!r})."
            )
        best_k_maps = SerializationFn.load_file(best_k_fpath)
        _op = min if best_k_mode == "min" else max
        best_model_path_record = _op(best_k_maps, key=best_k_maps.get)  # type: ignore[arg-type]
        best_model_name = Path(best_model_path_record).name

        # the best model path in best map file may not in this directory, fetch its file base name
        best_model_path = checkpoint_dir / best_model_name
        if not best_model_path.is_file():
            raise FileExistsError(
                f"Resolved best model name {best_model_name} in best k map but failed to find it in {best_model_path!r}."
            )
        checkpoint = str(best_model_path)
    else:
        checkpoint = str(resolve_file(str(checkpoint), str(checkpoint_dir)))
        if not Path(checkpoint).is_file():
            raise FileExistsError(f"Failed to get checkpoint in {checkpoint}.")
    return str(Path(checkpoint))


def _search_vallid_dir(
    dirpath: str,
    version: str = "version",
    ckpt_subdir: str = constants.CHECKPOINT_DIR_NAME,
):
    """Search for a valid directory containing the 'checkpoints' subdir."""
    base_dir = Path(dirpath)
    assert base_dir.is_dir()

    # Check base directory
    if (base_dir / ckpt_subdir).is_dir():
        return dirpath

    # Check specified version subdir
    if (base_dir / version / ckpt_subdir).is_dir():
        return str(base_dir / version)
    # Check max version num
    elif version:
        exist_versions = []
        for d in base_dir.iterdir():
            if d.is_dir() and d.name.startswith(f"{version+'_'}"):
                exist_versions.append(int(d.name.split("_")[1]))
        if len(exist_versions) > 0:
            max_ver = max(exist_versions)
            version_dirname = version + f"_{max_ver}"
            if (base_dir / version_dirname / ckpt_subdir).is_dir():
                return str(base_dir / version_dirname)
    raise ValueError(
        f"Failed to resolve a dir contains ckpts subdir ({ckpt_subdir}) in dirpath={dirpath}, version subdir={version}."
        f" Set dirpath or its version subdir contains dir ({ckpt_subdir})."
    )
