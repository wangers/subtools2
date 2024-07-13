# -*- coding:utf-8 -*-
# Copyright xmuspeech (Author: Leo 2023-03)

import contextlib
import csv
import io
import json
import re
import sys
from codecs import StreamReader, StreamWriter
from io import BytesIO, StringIO
from pathlib import Path
from typing import (
    Any,
    ClassVar,
    Dict,
    Generator,
    Iterable,
    List,
    Optional,
    Tuple,
    Union,
)

import yaml

from egrecho.utils.common import alt_none
from egrecho.utils.imports import _OMEGACONF_AVAILABLE, is_package_available
from egrecho.utils.patch import gzip_open_patch, stringify_path
from egrecho.utils.types import is_tensor

if is_package_available("orjson"):
    import orjson  # type: ignore[arg-type]

    def json_decode_line(line, **kwargs):
        try:
            return orjson.loads(line, **kwargs)
        except:  # noqa
            return json.loads(line, **kwargs)

else:
    json_decode_line = json.loads


def auto_open(path: Union[str, Path], mode: str = "r", **kwargs):
    """
    Open a Path, if it is end with 'gz', will call gzip.open first.

    Note: just support local path now.
    """
    strpath = stringify_path(path)
    if strpath == "-":
        if mode == "r":
            return StdStreamWrapper(sys.stdin)
        elif mode == "w":
            return StdStreamWrapper(sys.stdout)
        else:
            raise ValueError(
                f"Cannot open stream for '-' with mode other 'r' or 'w' (got: '{mode}')"
            )

    if isinstance(path, (BytesIO, StringIO, StreamWriter, StreamReader)):
        return path
    else:
        compressed = str(path).endswith(".gz")
        if compressed and "t" not in mode and "b" not in mode:
            # Opening as bytes not requested explicitly, use "t" to tell gzip to handle unicode.
            mode = mode + "t"
        open_fn = gzip_open_patch if compressed else open

    return open_fn(path, mode, **kwargs)


def csv_to_list(file):
    lists = []
    with open(file, newline="") as csvfile:
        reader = csv.DictReader(csvfile, delimiter=" ", skipinitialspace=True)
        for rows in reader:
            lists.append(rows)
    return lists


def torchaudio_info_unfixed(path_or_fileobj: Union[Path, str, io.BytesIO]):
    is_mp3 = isinstance(path_or_fileobj, (str, Path)) and str(path_or_fileobj).endswith(
        ".mp3"
    )
    is_fileobj = isinstance(path_or_fileobj, io.BytesIO)
    return is_mp3 or is_fileobj


class JsonMixin:
    """
    Loads/save json mixin.
    """

    @staticmethod
    def load_json(path: Union[Path, str], **kwargs) -> Dict:
        data = load_json(path, **kwargs)
        return data

    def to_json(self, path: Union[Path, str], **kwargs):
        output = self.to_dict()
        for key, value in output.items():
            if is_tensor(value):
                output[key] = value.tolist()
        save_json(output, path, **kwargs)

    @classmethod
    def from_json(cls, path: Union[Path, str], **kwargs) -> object:
        data = cls.load_json(path, **kwargs)
        return cls.from_dict(data)


class YamlMixin:
    """
    Loads/save yaml mixin.
    """

    yaml_inline_list: ClassVar[str] = True

    @staticmethod
    def load_yaml(path: Union[Path, str], **kwargs) -> Dict:
        data = load_yaml(path, **kwargs)
        return data

    def to_yaml(self, path: Union[Path, str], inline_list=None, **kwargs):
        output = self.to_dict()
        for key, value in output.items():
            if is_tensor(value):
                output[key] = value.tolist()
        inline_list = alt_none(inline_list, self.yaml_inline_list)
        save_yaml(output, path, inline_list=inline_list, **kwargs)

    @classmethod
    def from_yaml(cls, path: Union[Path, str], **kwargs) -> object:
        data = cls.load_yaml(path, **kwargs)
        return cls.from_dict(data)


class ConfigFileMixin(JsonMixin, YamlMixin):
    """
    To serialize/deserialize config files in local, support json and yaml.
    """

    yaml_inline_list: ClassVar[bool] = True

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
                    from egrecho.utils.common import omegaconf2container

                    return omegaconf2container(config)
                else:
                    return config
        return config

    def to_cfg_file(
        self, path: Union[Path, str], file_type: Optional[str] = None, **kwargs
    ):
        file_type = alt_none(file_type, Path(path).suffix.split(".")[-1])
        if file_type in ("yaml", "yml"):
            self.to_yaml(path, inline_list=self.yaml_inline_list, **kwargs)
        elif file_type == "json":
            self.to_json(path, **kwargs)
        else:
            raise ValueError(f"unsuport config file type: {file_type}")

    @classmethod
    def from_cfg_file(
        cls, path: Union[Path, str], file_type: Optional[str] = None, **kwargs
    ) -> object:
        data = cls.load_cfg_file(path, file_type, **kwargs)
        return cls.from_dict(data)

    # alias
    load_file = load_cfg_file
    to_file = to_cfg_file
    from_file = from_cfg_file


class DictFileMixin(ConfigFileMixin):
    yaml_inline_list: ClassVar[bool] = False


class SerializationFn:
    """
    Serialization fn mixin.
    """

    @staticmethod
    def load_file(
        path: Union[Path, str], file_type: Optional[str] = None, **kwargs
    ) -> Dict:
        file_type = alt_none(file_type, Path(path).suffix.split(".")[-1])
        if file_type in ("yaml", "yml"):
            return load_yaml(path, **kwargs)
        elif file_type == "json":
            return load_json(path, **kwargs)
        else:
            raise ValueError(f"unsuport file type: {file_type}")

    @staticmethod
    def save_file(
        data: dict, path: Union[Path, str], file_type: Optional[str] = None, **kwargs
    ):
        file_type = alt_none(file_type, Path(path).suffix.split(".")[-1])
        yaml_inline_list = kwargs.pop("inline_list", False)
        if file_type in ("yaml", "yml"):
            save_yaml(data, path, inline_list=yaml_inline_list, **kwargs)
        elif file_type == "json":
            save_json(data, path, **kwargs)
        else:
            raise ValueError(f"unsuport file type: {file_type}")


dump_yaml_kwargs = {
    "default_flow_style": False,
    "allow_unicode": True,
    "sort_keys": False,
}

dump_json_kwargs = {
    "ensure_ascii": False,
    "sort_keys": False,
    'indent': 2,
}


class _DefaultLoader(getattr(yaml, "CSafeLoader", yaml.SafeLoader)):  # type: ignore
    pass


# https://stackoverflow.com/a/37958106/2732151
def remove_implicit_resolver(cls, tag_to_remove):
    if "yaml_implicit_resolvers" not in cls.__dict__:
        cls.yaml_implicit_resolvers = cls.yaml_implicit_resolvers.copy()

    for first_letter, mappings in cls.yaml_implicit_resolvers.items():
        cls.yaml_implicit_resolvers[first_letter] = [
            (tag, regexp) for tag, regexp in mappings if tag != tag_to_remove
        ]


remove_implicit_resolver(_DefaultLoader, "tag:yaml.org,2002:timestamp")
remove_implicit_resolver(_DefaultLoader, "tag:yaml.org,2002:float")


_DefaultLoader.add_implicit_resolver(
    "tag:yaml.org,2002:float",
    re.compile(
        """^(?:
     [-+]?(?:[0-9][0-9_]*)\\.[0-9_]*(?:[eE][-+]?[0-9]+)?
    |[-+]?(?:[0-9][0-9_]*)(?:[eE][-+]?[0-9]+)
    |\\.[0-9_]+(?:[eE][-+][0-9]+)?
    |[-+]?[0-9][0-9_]*(?::[0-5]?[0-9])+\\.[0-9_]*
    |[-+]?\\.(?:inf|Inf|INF)
    |\\.(?:nan|NaN|NAN))$""",
        re.X,
    ),
    list("-+0123456789."),
)


# in order to dump list to yaml in python style.
class _InlineListDumper(yaml.SafeDumper):
    pass


def _yaml_list_representer(dumper, data):
    return dumper.represent_sequence("tag:yaml.org,2002:seq", data, flow_style=True)


_InlineListDumper.add_representer(tuple, _yaml_list_representer)
_InlineListDumper.add_representer(list, _yaml_list_representer)


def load_yaml(path: Union[Path, str], Loader=_DefaultLoader) -> Dict:
    with auto_open(path, "r") as fin:
        return yaml.load(fin, Loader=Loader)


def save_yaml(data: Any, path: Union[Path, str], inline_list=False, **kwargs) -> Dict:
    kwargs = {**dump_yaml_kwargs, **kwargs}
    with auto_open(path, "w") as fou:
        if not inline_list:
            yaml.dump(data, fou, Dumper=yaml.SafeDumper, **kwargs)
        else:
            yaml.dump(data, fou, Dumper=_InlineListDumper, **kwargs)


def yaml_load_string(stream):
    if stream.strip() == "-":
        value = stream
    else:
        value = yaml.load(stream, Loader=_DefaultLoader)
    if isinstance(value, dict) and value and all(v is None for v in value.values()):
        if len(value) == 1 and stream.strip() == next(iter(value.keys())) + ":":
            value = stream

        else:
            keys = set(stream.strip(" {}").replace(" ", "").split(","))
            if len(keys) > 0 and keys == set(value.keys()):
                value = stream
    return value


def load_json(path: Union[Path, str], **kwargs) -> Dict:
    with auto_open(path, "r", encoding="utf-8") as f:
        return json.load(f, **kwargs)


def save_json(data: Any, path: Union[Path, str], **kwargs) -> None:
    kwargs = {**dump_json_kwargs, **kwargs}
    with auto_open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, **kwargs)


def load_jsonl_lazy(path: Union[Path, str], **kwargs) -> Generator:
    """
    Load json lines in a lazy way.
    """
    with auto_open(path, "r", encoding="utf-8") as f:
        for line in f:
            rs = json_decode_line(line, **kwargs)
            yield rs


def save_jsonl(
    data: Iterable[Dict[str, Any]], path: Union[Path, str], **kwargs
) -> None:
    """
    Save json lines.
    """
    with auto_open(path, "w", encoding="utf-8") as f:
        for d in data:
            print(json.dump(d, ensure_ascii=False), file=f, **kwargs)


def load_csv_lazy(path: Union[Path, str], **fmtparams) -> Generator:
    """
    Load csv lines in a lazy way.
    """
    reader = csv.DictReader(path, **fmtparams)
    for d in reader:
        yield d


def save_csv(
    data: Iterable[Dict[str, Any]],
    path: Union[Path, str],
    fieldnames: List[str],
    **fmtparams,
) -> None:
    """
    Save csv lines.
    """
    encoding = fmtparams.pop("encoding", "utf-8")
    with auto_open(path, "w", encoding=encoding) as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames, **fmtparams)
        writer.writeheader()
        for d in data:
            writer.writerow(d)


def repr_dict(
    data: Dict[str, Any], sort_keys: bool = False, inline_list=True, **kwds
) -> str:
    """
    Make dict more readable.
    """
    # out_str = json.dumps(data,indent=2,sort_keys=True)
    for key, value in data.items():
        if is_tensor(value):
            data[key] = value.tolist()
    dumper = _InlineListDumper if inline_list else yaml.SafeDumper
    out_str = yaml.dump(
        data,
        Dumper=dumper,
        sort_keys=sort_keys,
        **kwds,
    )
    return out_str


class InvalidPathExtension(ValueError):
    pass


def extension_contains(ext: str, path: Union[str, Path]) -> bool:
    return any(ext == sfx for sfx in Path(path).suffixes)


def buf_count_newlines(fname: str) -> int:
    """
    Count the number of lines in a file
    """

    def _make_gen(reader):
        b = reader(1024 * 1024)
        while b:
            yield b
            b = reader(1024 * 1024)

    with auto_open(fname, "rb") as f:
        count = sum(buf.count(b"\n") for buf in _make_gen(f.read))
    return count


def read_key_first_lists(
    file_path: Union[str, Path],
    vector: bool = False,
    every_bytes: int = 10000000,
) -> List[Tuple[str, Union[str, List[str]]]]:
    """
    Reads txt line by line, and items sep with blank in one line.

    Returns a list of tuples contains two item, the first column and
    the second formated other columns. (the second can be a list of strings when `vector=True`).
    """

    rs_lst = []
    with Path(file_path).open(encoding="utf-8") as reader:
        while True:
            lines = reader.readlines(every_bytes)
            if not lines:
                break
            for line in lines:
                if vector:
                    # split_line => n
                    split_line = line.strip().split()
                    # split_line => n-1
                    key = split_line.pop(0)
                    rs_lst.append((key, split_line))
                else:
                    key, value = line.strip().split(maxsplit=1)
                    rs_lst.append((key, value))
    return rs_lst


def read_lists(
    file_path: Union[str, Path],
    vector: bool = False,
    every_bytes: int = 10000000,
) -> List[Union[str, List[str]]]:
    """
    Reads txt line by line, and items sep with blank in one line.

    Returns a list of strings
    (or a list of lists contains all strings splitted when `vector=True`).
    """

    rs_lst = []
    with Path(file_path).open(encoding="utf-8") as reader:
        while True:
            lines = reader.readlines(every_bytes)
            if not lines:
                break
            for line in lines:
                if vector:
                    split_line = line.strip().split()
                    rs_lst.append(split_line)
                else:
                    rs_lst.append(line)
    return rs_lst


def read_key_first_lists_lazy(
    file_path: Union[str, Path],
    vector: bool = False,
) -> Generator[Tuple[str, Union[str, List[str]]], None, None]:
    """
    Reads txt line by line lazy, and items sep with blank in one line.

    Generates tuples contains two item, the first column and
    the second formated other columns. (the second can be a list of strings when `vector=True`).
    """

    with Path(file_path).open(encoding="utf-8") as reader:
        for line in reader:
            if vector:
                # split_line => n
                split_line = line.strip().split()
                # split_line => n-1
                key = split_line.pop(0)
                value = split_line
            else:
                key, value = line.strip().split(maxsplit=1)
            yield key, value


def read_lists_lazy(
    file_path: Union[str, Path],
    vector: bool = False,
) -> Generator[Union[str, List[str]], None, None]:
    """
    Reads txt line by line lazy, and items sep with blank in one line.

    Returns a list of strings
    (or a list of lists contains all strings splitted when `vector=True`).
    """

    with Path(file_path).open(encoding="utf-8") as reader:
        for line in reader:
            if vector:
                split_line = line.strip().split()
                yield split_line
            else:
                yield line


class StdStreamWrapper:
    def __init__(self, stream):
        self.stream = stream

    def close(self):
        pass

    def __enter__(self):
        return self.stream

    def __exit__(self, exc_type, exc_val, exc_tb):
        pass

    def __getattr__(self, item: str):
        if item == "close":
            return self.close
        return getattr(self.stream, item)
