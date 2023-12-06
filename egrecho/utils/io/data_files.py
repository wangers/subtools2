# -*- coding:utf-8 -*-
# Copyright xmuspeech (Author: Leo 2023-04-22)
#           The HuggingFace Datasets Authors 2020
# With little modification from HuggingFace Datasets
#    (https://github.com/huggingface/datasets/blob/main/src/datasets/data_files.py).

import os
from functools import lru_cache, partial
from pathlib import Path, PurePath
from typing import Dict, List, Optional, Tuple, Union
from urllib.parse import urlparse

from tqdm.contrib.concurrent import thread_map

from egrecho.utils.common import list2tuple
from egrecho.utils.data_utils import Split
from egrecho.utils.logging import get_logger
from egrecho.utils.patch import FsspecLocalGlob

logger = get_logger()


WILDCARD_CHARACTERS = "*[]"
FILES_TO_IGNORE = [
    "*.md",
    "*config*.json*",
    "*config*.yaml*" "*info*.json",
    "*info*.yaml",
]


class Url(str):
    pass


def contains_wildcards(pattern: str) -> bool:
    return any(
        wilcard_character in pattern for wilcard_character in WILDCARD_CHARACTERS
    )


def sanitize_patterns(
    patterns: Union[Dict, List, str]
) -> Dict[str, Union[List[str], "DataFilesList"]]:
    """
    Take the data_files patterns from the user, and format them into a dictionary.
    Each key is the name of the split, and each value is a list of data files patterns (paths or urls).
    The default split is "train".

    Returns:
        patterns: dictionary of split_name -> list of file_patterns
    """
    if isinstance(patterns, dict):
        return {
            str(key): value if isinstance(value, list) else [value]
            for key, value in patterns.items()
        }
    elif isinstance(patterns, str):
        return {str(Split.TRAIN): [patterns]}
    elif isinstance(patterns, list):
        return {str(Split.TRAIN): patterns}
    else:
        return {str(Split.TRAIN): list(patterns)}


class DataFilesList(List[Union[Path, Url]]):
    """
    List of data files (absolute local paths or URLs).
    - ``from_local_or_remote``: resolve patterns from a local path

    Moreover DataFilesList has an additional attribute ``origin_metadata``.
    It can store:
    - the last modified time of local files.
    - Url metadata is not implemented currently.
    """

    def __init__(
        self, data_files: List[Union[Path, Url]], origin_metadata: List[Tuple[str]]
    ):
        super().__init__(data_files)
        self.origin_metadata = origin_metadata

    @classmethod
    def from_local_or_remote(
        cls,
        patterns: List[str],
        base_path: Optional[str] = None,
        allowed_extensions: Optional[List[str]] = None,
        use_auth_token: Optional[Union[bool, str]] = None,
        skip_metadata: bool = True,
    ) -> "DataFilesList":
        base_path = base_path if base_path is not None else str(Path().resolve())
        data_files = resolve_patterns_locally_or_by_urls(
            base_path, patterns, allowed_extensions
        )
        origin_metadata = (
            _get_origin_metadata_locally_or_by_urls(
                data_files, use_auth_token=use_auth_token
            )
            if skip_metadata
            else None
        )
        return cls(data_files, origin_metadata)


class DataFilesDict(Dict[str, DataFilesList]):
    """
    Dict of split_name -> list of data files (absolute local paths or URLs).
    - ``from_local_or_remote``: resolve patterns from a local path

    Moreover each list is a DataFilesList. For more info, see ``DataFilesList``.
    """

    @classmethod
    def from_local_or_remote(
        cls,
        patterns: Dict[str, Union[List[str], DataFilesList]],
        base_path: Optional[str] = None,
        allowed_extensions: Optional[List[str]] = None,
        use_auth_token: Optional[Union[bool, str]] = None,
    ) -> "DataFilesDict":
        out = cls()
        for key, patterns_for_key in patterns.items():
            out[key] = (
                DataFilesList.from_local_or_remote(
                    patterns_for_key,
                    base_path=base_path,
                    allowed_extensions=allowed_extensions,
                    use_auth_token=use_auth_token,
                )
                if not isinstance(patterns_for_key, DataFilesList)
                else patterns_for_key
            )
        return out


@list2tuple
@lru_cache(maxsize=8)
def resolve_patterns_locally_or_by_urls(
    base_path: str, patterns: List[str], allowed_extensions: Optional[Tuple[str]] = None
) -> List[Union[Path, Url]]:
    """
    Resolve the paths and URLs of the data files from the patterns passed by the user.
    URLs are just returned as is.

    You can use patterns to resolve multiple local files. Here are a few examples:
    - *.csv to match all the CSV files at the first level
    - **.csv to match all the CSV files at any level
    - data/* to match all the files inside "data"
    - data/** to match all the files inside "data" and its subdirectories

    The patterns are resolved using the fsspec glob.
    Here are some behaviors specific to fsspec glob that are different from glob.glob, Path.glob, Path.match or fnmatch:
    - '*' matches only first level items
    - '**' matches all items
    - '**/*' matches all at least second level items

    More generally:
    - '*' matches any character except a forward-slash (to match just the file or directory name)
    - '**' matches any character including a forward-slash /

    Hidden files and directories (i.e. whose names start with a dot) are ignored, unless they are explicitly requested.
    The same applies to special directories that start with a double underscore like "__pycache__".
    You can still include one if the pattern explicilty mentions it:
    - to include a hidden file: "*/.hidden.txt" or "*/.*"
    - to include a hidden directory: ".hidden/*" or ".*/*"
    - to include a special directory: "__special__/*" or "__*/*"
        e.g., glob.glob('**/*', recursive=True), the last /* is invalid as greedy mode of first pattern '**'.

    Args:
        base_path (str): Base path to use when resolving relative paths.
        patterns (List[str]): Unix patterns or paths or URLs of the data files to resolve.
            The paths can be absolute or relative to base_path.
        allowed_extensions (Optional[list], optional): White-list of file extensions to use. Defaults to None (all extensions).
            For example: allowed_extensions=["csv", "json", "txt", "parquet"]

    Returns:
        List[Union[Path, Url]]: List of paths or URLs to the local or remote files that match the patterns.
    """
    data_files = []
    if isinstance(patterns, str):
        patterns = [patterns]
    for pattern in patterns:
        if is_remote_url(Url(pattern)):
            data_files.append(pattern)
        else:
            for path in _resolve_single_pattern_locally(
                base_path, pattern, allowed_extensions
            ):
                data_files.append(path)

    if not data_files:
        error_msg = f"Unable to resolve any data file that matches '{patterns}' at {Path(base_path).resolve()}"
        if allowed_extensions is not None:
            error_msg += f" with any supported extension {list(allowed_extensions)}"
        raise FileNotFoundError(error_msg)
    return data_files


@list2tuple
@lru_cache(maxsize=8)
def _resolve_single_pattern_locally(
    base_path: str, pattern: str, allowed_extensions: Optional[Tuple[str]] = None
) -> List[Path]:
    """
    Return the absolute paths to all the files that match the given patterns.
    It also supports absolute paths in patterns.
    If an URL is passed, it is returned as is.
    """
    if is_relative_path(pattern):
        pattern = os.path.join(base_path, pattern)
    else:
        base_path = os.path.splitdrive(pattern)[0] + os.sep

    glob_iter = [
        PurePath(filepath)
        for filepath in FsspecLocalGlob.glob(pattern)
        if FsspecLocalGlob.isfile(filepath)
    ]
    matched_paths = [
        Path(os.path.abspath(filepath))
        for filepath in glob_iter
        if (
            filepath.name not in FILES_TO_IGNORE
            or PurePath(pattern).name == filepath.name
        )
        and not _is_inside_unrequested_special_dir(
            os.path.relpath(filepath, base_path), os.path.relpath(pattern, base_path)
        )
        and not _is_unrequested_hidden_file_or_is_inside_unrequested_hidden_dir(
            os.path.relpath(filepath, base_path), os.path.relpath(pattern, base_path)
        )
    ]  # ignore .ipynb and __pycache__, but keep /../
    if allowed_extensions is not None:
        out = [
            filepath
            for filepath in matched_paths
            if any(suffix[1:] in allowed_extensions for suffix in filepath.suffixes)
        ]
        if len(out) < len(matched_paths):
            invalid_matched_files = list(set(matched_paths) - set(out))
            logger.info(
                f"Some files matched the pattern '{pattern}' at {Path(base_path).resolve()} but don't have valid data file extensions: {invalid_matched_files}"
            )
    else:
        out = matched_paths
    if not out and not contains_wildcards(pattern):
        error_msg = f"Unable to find '{pattern}' at {Path(base_path).resolve()}"
        if allowed_extensions is not None:
            error_msg += f" with any supported extension {list(allowed_extensions)}"
        raise FileNotFoundError(error_msg)
    return sorted(out)


# alias
resolve_patterns = resolve_patterns_locally_or_by_urls


def resolve_file(fname: str, base_path: Optional[str] = None):
    """
    Resolve a single file

    if given base_path and rel fname, get the file subject the base_path.

    """
    if is_remote_url(Url(fname)):
        return fname
    else:
        if is_relative_path(fname):
            base_path = base_path if base_path is not None else str(Path().resolve())

            return os.path.join(base_path, fname)
        else:
            return fname


def _get_single_origin_metadata_locally_or_by_urls(
    data_file: Union[Path, Url], use_auth_token: Optional[Union[bool, str]] = None
) -> Tuple[str]:
    if isinstance(data_file, Url):
        data_file = str(data_file)
        logger.warning_once("NotImplement get info from remote url.")
        return ("url_none",)
    else:
        data_file = str(data_file.resolve())
        return (str(os.path.getmtime(data_file)),)


def _get_origin_metadata_locally_or_by_urls(
    data_files: List[Union[Path, Url]],
    max_workers=64,
    use_auth_token: Optional[Union[bool, str]] = None,
) -> Tuple[str]:
    return thread_map(
        partial(
            _get_single_origin_metadata_locally_or_by_urls,
            use_auth_token=use_auth_token,
        ),
        data_files,
        max_workers=max_workers,
        desc="Resolving data files",
        disable=len(data_files) <= 16,
    )


def is_relative_path(url_or_filename: str) -> bool:
    return urlparse(url_or_filename).scheme == "" and not os.path.isabs(url_or_filename)


def is_local_path(url_or_filename: str) -> bool:
    # On unix the scheme of a local path is empty (for both absolute and relative),
    # while on windows the scheme is the drive name (ex: "c") for absolute paths.
    # for details on the windows behavior, see https://bugs.python.org/issue42215
    return urlparse(url_or_filename).scheme == "" or os.path.ismount(
        urlparse(url_or_filename).scheme + ":/"
    )


def is_remote_url(url_or_filename: str) -> bool:
    parsed = urlparse(url_or_filename)
    return parsed.scheme in ("http", "https", "s3", "gs", "hdfs", "ftp")


def _is_inside_unrequested_special_dir(matched_rel_path: str, pattern: str) -> bool:
    """
    When a path matches a pattern, we additionnally check if it's inside a special directory
    we ignore by default (if it starts with a double underscore).

    Users can still explicitly request a filepath inside such a directory if "__pycache__" is
    mentioned explicitly in the requested pattern.

    Some examples:

    base directory:

        ./
        └── __pycache__
            └── b.txt

    >>> _is_inside_unrequested_special_dir("__pycache__/b.txt", "**")
    True
    >>> _is_inside_unrequested_special_dir("__pycache__/b.txt", "*/b.txt")
    True
    >>> _is_inside_unrequested_special_dir("__pycache__/b.txt", "__pycache__/*")
    False
    >>> _is_inside_unrequested_special_dir("__pycache__/b.txt", "__*/*")
    False
    """
    # We just need to check if every special directories from the path is present explicly in the pattern.
    # Since we assume that the path matches the pattern, it's equivalent to counting that both
    # the parent path and the parent pattern have the same number of special directories.
    data_dirs_to_ignore_in_path = [
        part
        for part in PurePath(matched_rel_path).parent.parts
        if part.startswith("__")
    ]
    data_dirs_to_ignore_in_pattern = [
        part for part in PurePath(pattern).parent.parts if part.startswith("__")
    ]
    return len(data_dirs_to_ignore_in_path) != len(data_dirs_to_ignore_in_pattern)


def _is_unrequested_hidden_file_or_is_inside_unrequested_hidden_dir(
    matched_rel_path: str, pattern: str
) -> bool:
    """
    When a path matches a pattern, we additionnally check if it's a hidden file or if it's inside
    a hidden directory we ignore by default, i.e. if the file name or a parent directory name starts with a dot.

    Users can still explicitly request a filepath that is hidden or is inside a hidden directory
    if the hidden part is mentioned explicitly in the requested pattern.

    Some examples:

    base directory:

        ./
        └── .hidden_file.txt

    >>> _is_unrequested_hidden_file_or_is_inside_unrequested_hidden_dir(".hidden_file.txt", "**")
    True
    >>> _is_unrequested_hidden_file_or_is_inside_unrequested_hidden_dir(".hidden_file.txt", ".*")
    False

    base directory:

        ./
        └── .hidden_dir
            └── a.txt

    >>> _is_unrequested_hidden_file_or_is_inside_unrequested_hidden_dir(".hidden_dir/a.txt", "**")
    True
    >>> _is_unrequested_hidden_file_or_is_inside_unrequested_hidden_dir(".hidden_dir/a.txt", ".*/*")
    False
    >>> _is_unrequested_hidden_file_or_is_inside_unrequested_hidden_dir(".hidden_dir/a.txt", ".hidden_dir/*")
    False

    base directory:

        ./
        └── .hidden_dir
            └── .hidden_file.txt

    >>> _is_unrequested_hidden_file_or_is_inside_unrequested_hidden_dir(".hidden_dir/.hidden_file.txt", "**")
    True
    >>> _is_unrequested_hidden_file_or_is_inside_unrequested_hidden_dir(".hidden_dir/.hidden_file.txt", ".*/*")
    True
    >>> _is_unrequested_hidden_file_or_is_inside_unrequested_hidden_dir(".hidden_dir/.hidden_file.txt", ".*/.*")
    False
    >>> _is_unrequested_hidden_file_or_is_inside_unrequested_hidden_dir(".hidden_dir/.hidden_file.txt", ".hidden_dir/*")
    True
    >>> _is_unrequested_hidden_file_or_is_inside_unrequested_hidden_dir(".hidden_dir/.hidden_file.txt", ".hidden_dir/.*")
    False
    """
    # We just need to check if every hidden part from the path is present explicly in the pattern.
    # Since we assume that the path matches the pattern, it's equivalent to counting that both
    # the path and the pattern have the same number of hidden parts.
    hidden_directories_in_path = [
        part
        for part in PurePath(matched_rel_path).parts
        if part.startswith(".") and not set(part) == {"."}
    ]
    hidden_directories_in_pattern = [
        part
        for part in PurePath(pattern).parts
        if part.startswith(".") and not set(part) == {"."}
    ]
    return len(hidden_directories_in_path) != len(hidden_directories_in_pattern)
