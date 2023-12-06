# -*- coding:utf-8 -*-
# Copyright xmuspeech (Author: Leo 2023-05)

import os
import re
import tarfile
import warnings
from io import BufferedIOBase
from typing import IO, Dict, Iterator, Optional, Tuple, cast

from egrecho.utils.patch import StreamWrapper, validate_pathname_binary_tuple

__all__ = ["load_from_tar", "webdataset"]


def load_from_tar(data, mode: str = "r:*") -> Iterator[Tuple[str, BufferedIOBase]]:
    r"""
    Opens/decompresses tar binary streams from data which contains tuples of path name and
    tar binary stream, and yields a tuple of path name and extracted binary stream (functional name: ``load_from_tar``).

    Args:
        data: data that provides tuples of path name and tar binary stream
        mode: File mode used by `tarfile.open` to read file object.
            Mode has to be a string of the form `'filemode[:compression]'`

    Note:
        User should be responsible to close file handles explicitly
        or let Python's GC close them periodically.

    Example:
        >>> from egrecho.data.iterable.processors import open_files, load_from_tar
        >>> list(load_from_tar(open_files(['test_fake.tar',], mode="b")))
        >>> [('test_fake.tar/tom.txt',
        ... StreamWrapper<test_fake.tar/tom.txt,<ExFileObject name=None>>),
        >>> ('test_fake.tar/jimmy.txt',
        ... StreamWrapper<test_fake.tar/jimmy.txt,<ExFileObject name=None>>)]
    """
    for sample in data:
        validate_pathname_binary_tuple(sample)
        pathname, data_stream = sample
        try:
            if isinstance(data_stream, StreamWrapper) and isinstance(
                data_stream.file_obj, tarfile.TarFile
            ):
                tar = data_stream.file_obj
            else:
                reading_mode = (
                    mode
                    if hasattr(data_stream, "seekable") and data_stream.seekable()
                    else mode.replace(":", "|")
                )
                tar = tarfile.open(
                    fileobj=cast(Optional[IO[bytes]], data_stream), mode=reading_mode
                )
            for tarinfo in tar:
                if not tarinfo.isfile():
                    continue
                extracted_fobj = tar.extractfile(tarinfo)
                if extracted_fobj is None:
                    warnings.warn(
                        f"failed to extract file {tarinfo.name} from source tarfile {pathname}"
                    )
                    raise tarfile.ExtractError
                inner_pathname = os.path.normpath(os.path.join(pathname, tarinfo.name))
                yield inner_pathname, StreamWrapper(
                    extracted_fobj, data_stream, name=inner_pathname
                )
        except Exception as e:
            warnings.warn(
                f"Unable to extract files from corrupted tarfile stream {pathname} due to: {e}, abort!"
            )
            raise e
        finally:
            if isinstance(data_stream, StreamWrapper):
                data_stream.autoclose()


def pathsplit(p):
    """Split a path into a WebDataset prefix and suffix.

    The prefix is used for grouping files into samples,
    the suffix is used as key in the output dictionary.
    The suffix consists of all components after the last
    "." in the filename.

    In torchdata, the prefix consists of the .tar file
    path followed by the file name inside the archive.

    Any backslash in the prefix is replaced by a forward
    slash to make Windows prefixes consistent with POSIX
    paths.
    """

    # convert Windows pathnames to UNIX pathnames, otherwise
    # we get an inconsistent mix of the Windows path to the tar
    # file followed by the POSIX path inside that tar file
    p = p.replace("\\", "/")
    if "." not in p:
        return p, ""
    # we need to use a regular expression because os.path is
    # platform specific, but tar files always contain POSIX paths
    match = re.search(r"^(.*?)(\.[^/]*)$", p)
    if not match:
        return p, ""
    prefix, suffix = match.groups()
    return prefix, suffix


def webdataset(source_data) -> Iterator[Dict]:
    r"""
    data that accepts stream of (path, data) tuples, usually,
    representing the pathnames and files of a tar archive (functional name:
    ``webdataset``). This aggregates consecutive items with the same basename
    into a single dictionary, using the extensions as keys (WebDataset file
    convention). Any text after the first "." in the filename is used as
    a key/extension.

    File names that do not have an extension are ignored.

    Args:
        source_data: data yielding a stream of (path, data) pairs

    Returns:
        data yielding a stream of dictionaries

    Examples:
        >>> from egrecho.data.iterable.processors import open_files, load_from_tar, webdataset
        >>> list(webdataset(load_from_tar(open_files(['test_fake.tar',], mode="b"))))
        >>> [{'__key__': 'test_fake.tar/tom',
        ... '.txt': StreamWrapper<test_fake.tar/tom.txt,<ExFileObject name=None>>},
        >>> {'__key__': 'test_fake.tar/jimmy',
        ... '.txt': StreamWrapper<test_fake.tar/jimmy.txt,<ExFileObject name=None>>}]
    """
    sample = {}
    current = ""
    for path, data in source_data:
        assert isinstance(path, str), path
        prefix, suffix = pathsplit(path)
        if suffix == "":
            # files with empty suffixes can be used for metadata
            # they cannot be used for data since they wouldn't have a key
            continue
        if prefix != current:
            if current != "":
                yield sample
            sample = {}
            current = prefix
            sample["__key__"] = current
        sample[suffix] = data
    if sample != {}:
        yield sample
