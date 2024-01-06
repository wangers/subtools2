# -*- coding:utf-8 -*-
# Copyright xmuspeech (Author: Leo 2023-10)

from functools import lru_cache
from pathlib import Path
from typing import TYPE_CHECKING, Optional, Union

from egrecho.utils.imports import _KALDI_NATIVE_IO_AVAILABLE, lazy_import

if TYPE_CHECKING:
    import numpy as np
else:
    np = lazy_import("numpy")


def valid_kaldi_storage_name(name: str):
    """As we will add ``.scp``, ``.ark`` suffix in writer, suffix should not contains them."""
    if len(suf := name.rsplit(".", 1)) == 2:
        if any(suf[1] == s for s in ("scp", "ark")):
            raise ValueError(
                "Kaldi writer storage file name should not with suffix of `(.scp, .ark)`, "
                f"But got a invalid name: {name}."
            )


def close_cached_kaldi_handles() -> None:
    """
    Closes the cached file handles in ``lookup_cache_or_open`` and
    ``lookup_reader_cache_or_open`` (see respective docs for more details).
    """
    lookup_matrix_reader_cache_or_open.cache_clear()
    lookup_vector_reader_cache_or_open.cache_clear()


@lru_cache(maxsize=None)
def lookup_matrix_reader_cache_or_open(storage_path: str):
    """
    Helper internal function used in :class:`KaldiMatrixReader`.
    It opens kaldi scp files and keeps their handles open in a global program cache
    to avoid excessive amount of syscalls when the Reader class is instantiated
    and destroyed in a loop repeatedly (frequent use-case).

    The file handles can be freed at any time by calling :meth`close_cached_file_handles`.
    """
    if not _KALDI_NATIVE_IO_AVAILABLE:
        raise ValueError(
            "To read Kaldi feats.scp, please ``pip install kaldi_native_io`` first."
        )
    import kaldi_native_io

    return kaldi_native_io.RandomAccessFloatMatrixReader(f"scp:{storage_path}")


@lru_cache(maxsize=None)
def lookup_vector_reader_cache_or_open(storage_path: str):
    """
    Helper internal function used in :class:`KaldiVectorReader`.
    It opens kaldi scp files and keeps their handles open in a global program cache
    to avoid excessive amount of syscalls when the Reader class is instantiated
    and destroyed in a loop repeatedly (frequent use-case).

    The file handles can be freed at any time by calling :meth:`close_cached_file_handles`.
    """
    if not _KALDI_NATIVE_IO_AVAILABLE:
        raise ValueError(
            "To read Kaldi scp, please ``pip install kaldi_native_io`` first."
        )
    import kaldi_native_io

    return kaldi_native_io.RandomAccessFloatVectorReader(f"scp:{storage_path}")


class KaldiVectorReader:
    """
    Reads Kaldi's vector (``1-D float32``) file (e.g., "xvector.scp") using kaldi_native_io.
    ``storage_path`` corresponds to the path to file with suffix ``.scp``.
    ``storage_key`` corresponds to the utterance-id in Kaldi.

    .. caution::
        Requires ``kaldi_native_io`` to be installed (``pip install kaldi_native_io``).
    """

    def __init__(self, storage_path: Union[Path, str], *args, **kwargs):
        super().__init__()
        self.storage_path = storage_path
        if str(storage_path).endswith(".scp"):
            self.storage = lookup_vector_reader_cache_or_open(self.storage_path)
        else:
            if not _KALDI_NATIVE_IO_AVAILABLE:
                raise ValueError(
                    "To read Kaldi file, please 'pip install kaldi_native_io' first."
                )
            import kaldi_native_io

            self.storage = None
            self.reader = kaldi_native_io.FloatVector

    def read(
        self,
        key: str,
    ) -> np.ndarray:
        if self.storage is not None:
            arr = np.copy(self.storage[key])
        else:
            arr = self.reader.read(self.storage_path).numpy()

        return arr


class KaldiVectorWriter:
    """
    Write vector data (``1-D float32``) to Kaldi's ".scp" and ".ark" files using kaldi_native_io.
    ``storage_path`` corresponds to a directory where we'll create "xvector.scp"
    and "xvector.ark" files.
    ``storage_key`` corresponds to the utterance-id in Kaldi.
    ``storage_name`` specified the stem name, i.e., "xvector".

    Example::

        >>> data = np.random.randn(192).astype(np.float32)
        >>> with KaldiVectorWriter('xvectordir') as w:
        ...     w.write('utt1', data)
        >>> reader = KaldiVectorReader('xvectordir/xvector.scp')
        >>> read_data = reader.read('utt1')
        >>> np.testing.assert_equal(data, read_data)

    .. caution::
        Requires ``kaldi_native_io`` to be installed (``pip install kaldi_native_io``).
    """

    def __init__(
        self,
        storage_path: Union[str, Path],
        storage_name: str = "xvector",
        **kwargs,
    ):
        if not _KALDI_NATIVE_IO_AVAILABLE:
            raise ValueError(
                "To write Kaldi feats.scp, please 'pip install kaldi_native_io' first."
            )
        import kaldi_native_io

        super().__init__()
        self.storage_dir = Path(storage_path)
        valid_kaldi_storage_name(storage_name)
        self.storage_dir.mkdir(parents=True, exist_ok=True)
        self.storage_path_ = str(self.storage_dir / (storage_name + ".scp"))

        self.storage = kaldi_native_io.FloatVectorWriter(
            f"ark,scp:{self.storage_dir/(storage_name+'.ark')},"
            f"{self.storage_dir/(storage_name+'.scp')}"
        )

    @property
    def storage_path(self) -> str:
        return self.storage_path_

    def write(self, key: str, value: np.ndarray) -> str:
        self.storage.write(key, value)
        return key

    def close(self) -> None:
        return self.storage.close()

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()


class KaldiMatrixReader:
    """
    Reads Kaldi's "feats.scp" file using kaldi_native_io.
    ``storage_path`` corresponds to the path to ``feats.scp``.
    ``storage_key`` corresponds to the utterance-id in Kaldi.

    referring:
        https://github.com/lhotse-speech/lhotse/blob/master/lhotse/features/io.py

    .. caution::
        Requires ``kaldi_native_io`` to be installed (``pip install kaldi_native_io``).
    """

    def __init__(self, storage_path: Union[Path, str], *args, **kwargs):
        super().__init__()
        self.storage_path = storage_path
        if str(storage_path).endswith(".scp"):
            self.storage = lookup_matrix_reader_cache_or_open(self.storage_path)
        else:
            if not _KALDI_NATIVE_IO_AVAILABLE:
                raise ValueError(
                    "To read Kaldi feats.scp, please 'pip install kaldi_native_io' first."
                )
            import kaldi_native_io

            self.storage = None
            self.reader = kaldi_native_io.FloatMatrix

    def read(
        self,
        key: str,
        left_offset_frames: int = 0,
        right_offset_frames: Optional[int] = None,
    ) -> np.ndarray:
        if self.storage is not None:
            arr = np.copy(self.storage[key])
        else:
            arr = self.reader.read(self.storage_path).numpy()

        return arr[left_offset_frames:right_offset_frames]


class KaldiMatrixWriter:
    """
    Write data to Kaldi's "feats.scp" and "feats.ark" files using kaldi_native_io.
    ``storage_path`` corresponds to a directory where we'll create "feats.scp"
    and "feats.ark" files.
    ``storage_key`` corresponds to the utterance-id in Kaldi.
    ``storage_name`` specified the stem name, i.e., "feats".

    referring:
        https://github.com/lhotse-speech/lhotse/blob/master/lhotse/features/io.py

    The following ``compression_method`` values are supported by kaldi_native_io::

        kAutomaticMethod = 1
        kSpeechFeature = 2
        kTwoByteAuto = 3
        kTwoByteSignedInteger = 4
        kOneByteAuto = 5
        kOneByteUnsignedInteger = 6
        kOneByteZeroOne = 7

    .. note:: Setting compression_method works only with 2D arrays.

    Example::

        >>> data = np.random.randn(131, 80)
        >>> with KaldiMatrixWriter('featdir') as w:
        ...     w.write('utt1', data)
        >>> reader = KaldiMatrixReader('featdir/feats.scp')
        >>> read_data = reader.read('utt1')
        >>> np.testing.assert_equal(data, read_data)

    .. caution::
        Requires ``kaldi_native_io`` to be installed (``pip install kaldi_native_io``).
    """

    def __init__(
        self,
        storage_path: Union[str, Path],
        storage_name: str = "feats",
        *,
        compression_method: int = 1,
        **kwargs,
    ):
        if not _KALDI_NATIVE_IO_AVAILABLE:
            raise ValueError(
                "To write Kaldi feats.scp, please 'pip install kaldi_native_io' first."
            )
        import kaldi_native_io

        super().__init__()
        self.storage_dir = Path(storage_path)
        valid_kaldi_storage_name(storage_name)
        self.storage_dir.mkdir(parents=True, exist_ok=True)
        self.storage_path_ = str(self.storage_dir / (storage_name + ".scp"))
        self.storage = kaldi_native_io.CompressedMatrixWriter(
            f"ark,scp:{self.storage_dir/(storage_name+'.ark')},"
            f"{self.storage_dir/(storage_name+'.scp')}"
        )
        self.compression_method = kaldi_native_io.CompressionMethod(compression_method)

    @property
    def storage_path(self) -> str:
        return self.storage_path_

    def write(self, key: str, value: np.ndarray) -> str:
        self.storage.write(key, value, self.compression_method)
        return key

    def close(self) -> None:
        return self.storage.close()

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()
