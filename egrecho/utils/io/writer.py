# -*- coding:utf-8 -*-
# Copyright xmuspeech (Author: Leo 2023-03)

import io
import json
import tarfile
import warnings
from pathlib import Path
from typing import TYPE_CHECKING, Any, List, Optional, Union

from egrecho.utils.io.utils import (
    InvalidPathExtension,
    auto_open,
    extension_contains,
    json_decode_line,
)


class SequentialDewWriter:
    """
    Sequently store dews (manifest), support json lines (jsonl).

    This implementation is mostly based on lhotse:
    https://github.com/lhotse-speech/lhotse/blob/master/lhotse/serialization.py#SequentialJsonlWriter
    """

    def __init__(self, path: Union[str, Path], overwrite: bool = True) -> None:
        self.path = path
        self.file = None
        if not extension_contains(".jsonl", self.path):
            raise InvalidPathExtension(
                f"SequentialWriter supports only json lines format."
                f"but path='{path}'."
            )
        self.mode = "w"
        self.ignore_ids = set()
        if Path(self.path).is_file() and not overwrite:
            self.mode = "a"
            with auto_open(self.path, "r") as f:
                self.ignore_ids = {
                    data["id"]
                    for data in (json_decode_line(line) for line in f)
                    if "id" in data
                }

    def __enter__(self) -> "SequentialDewWriter":
        self._maybe_open()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        self.close()

    def __contains__(self, item: Union[str, Any]) -> bool:
        if isinstance(item, str):
            return item in self.ignore_ids
        try:
            return item.id in self.ignore_ids
        except (AttributeError, KeyError):
            # For the case without id key.
            return False

    def _maybe_open(self):
        if self.file is None:
            self.file = auto_open(self.path, self.mode)

    def close(self):
        if self.file is not None:
            self.file.close()
            self.file = None

    def contains(self, item: Union[str, Any]) -> bool:
        return item in self

    def write(self, manifest: Any, flush: bool = False) -> bool:
        """
        Serializes a manifest item (e.g. :class:`~egrecho.core.DictDew.`) to JSON and stores it in a JSONL file.

        :param manifest: the manifest to be written.
        :param flush: should we flush the file after writing (ensures the changes
            are synced with the disk and not just buffered for later writing).
        """
        try:
            if manifest.id in self.ignore_ids:
                return False
        except (AttributeError, KeyError):
            pass
        self._maybe_open()
        try:
            manifest = manifest.to_dict()
        except Exception:
            pass
        print(json.dumps(manifest, ensure_ascii=False), file=self.file)
        if flush:
            self.file.flush()
        return True


class ShardWriter:
    r"""Create a ShardWriter, data should be of webdataset format.

    Args:
        pattern:
            output file pattern.
        shard_size:
            maximum number of records per shard, if None, means infinite.

    NOTE:
        - If `pattern` is a specify filepath, it will write to one tarfile.
        - If given shard_size, it will streamly write items to many tarfiles with a max size `shard_size`,
          in this case, the pattern must be specified such as '%06d' for str matching.

    Example:
        >>> samples = [
        >>>    {'__key__': 'tom', 'txt': 'i want eat.'},
        >>>    {'__key__': 'jimmy', 'txt': 'i want sleep.'}
        >>> ]
        >>> with ShardWriter('./test_fake.tar') as writer:
        >>>     for item in samples:
        >>>         writer.write(item)
    """

    def __init__(
        self,
        pattern: Union[str, Path],
        shard_size: Optional[int] = None,
    ):
        self.pattern = pattern
        if self.sharding_enabled and shard_size is None:
            raise RuntimeError(
                "shard_size must be specified when sharding is enabled via a formatting marker such as '%06d'"
            )
        if not self.sharding_enabled and shard_size is not None:
            warnings.warn(
                "Sharding is disabled because `pattern` doesn't contain a formatting marker (e.g., '%06d'), "
                "but shard_size is not None - ignoring shard_size."
            )
        self.tarmode = "w|gz" if pattern.endswith("gz") else "w|"
        self.shard_size = shard_size

        self.fname = None
        self.stream = None
        self.tarstream = None
        self.num_shards = 0
        self.count = 0
        self.total = 0

    @property
    def sharding_enabled(self) -> bool:
        return "%" in self.pattern

    def next_stream(self):
        """Close the current stream and move to the next."""
        self.close()
        if self.sharding_enabled:
            self.fname = self.pattern % self.num_shards
            self.num_shards += 1
        else:
            self.fname = self.pattern
        self.stream = auto_open(self.fname, "wb")
        self.tarstream = tarfile.open(fileobj=self.stream, mode=self.tarmode)
        self.count = 0

    def write(self, obj) -> int:
        """Write a sample.

        :param obj: sample to be written
        """
        written = 0
        if self.total == 0 or (
            self.sharding_enabled
            and self.count > 0
            and self.count % self.shard_size == 0
        ):
            self.next_stream()

        self._write_one(obj)
        self.count += 1
        self.total += 1
        written = 1
        return written

    def _write_one(self, obj):
        if "__key__" not in obj:
            raise ValueError("object must contain a __key__")
        for k, v in list(obj.items()):
            if k[0] == "_":
                continue
            if isinstance(v, str):
                v = io.BytesIO(v.encode("utf-8"))
                obj[k] = v
            if not isinstance(v, io.BytesIO):
                raise (f"Expect {k} is `io.BytesIO`, but got {type(v)}.")
        key = obj["__key__"]
        for k in obj.keys():
            if k == "__key__":
                continue
            v = obj[k]
            v.seek(0)
            ti = tarfile.TarInfo(key + "." + k)
            ti.size = len(v.getvalue())
            self.tarstream.addfile(ti, v)

    def close(self):
        """Finish all writing."""
        if self.tarstream is not None:
            self.tarstream.close()
        if self.stream is not None:
            self.stream.close()

    @property
    def output_paths(self) -> List[str]:
        if self.sharding_enabled:
            return [self.pattern % i for i in range(self.num_shards)]
        return [self.pattern]

    def __enter__(self):
        """Enter context."""
        return self

    def __exit__(self, *args, **kw):
        """Exit context."""
        self.close()
