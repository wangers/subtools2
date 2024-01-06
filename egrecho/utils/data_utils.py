# -*- coding:utf-8 -*-
# Copyright xmuspeech (Author: Leo 2023-03)

import collections
import copy
import queue
import random
import warnings
from collections import deque
from dataclasses import InitVar, dataclass, fields
from itertools import accumulate, chain, count
from pathlib import Path
from queue import Empty
from typing import (
    Any,
    Callable,
    ClassVar,
    Dict,
    Generator,
    Iterable,
    Iterator,
    List,
    Optional,
    Sequence,
    Union,
    cast,
)

import numpy as np

from egrecho.utils.common import asdict_filt
from egrecho.utils.io.utils import DictFileMixin
from egrecho.utils.types import Split


def split_sequence(
    seq: Sequence[Any],
    split_num: int,
    mode: str = "batch",
    shuffle: bool = False,
    drop_last: bool = False,
) -> List[List[Any]]:
    """
    Split a sequence into ``num_splits`` equal parts. The element order can be randomized.
    Raises a ``ValueError`` if ``split_num`` is larger than ``len(seq)``.
    Support mode of 'shard' or 'batch'. If 'batch', the splits lists as original sequence
    else shard the original sequence, e.g., for spliting ``[0, 1 , 2, 3]`` to 2 parts,
    shard mode result: ``[[0, 2], [1, 3]]`` while batch mode reult: ``[[0, 1], [2, 3]]``.

    Args:
        seq (Sequence):
            Input iterable.
        num_splits (int):
            Split num.
        mode (str):
            ('shard', 'batch')
        shuffle (bool):
            If true, shuffle input sequence before split it.
        drop_last (bool):
            If true, drop last items when ``len(seq)`` is not divisible by ``num_splits``.

    Returns:
        List of smaller squences.
    """
    seq = list(seq)
    num_items = len(seq)
    if split_num > num_items:
        raise ValueError(
            f"Cannot split iterable into more splits ({split_num}) than its number of items {num_items}"
        )
    if shuffle:
        random.shuffle(seq)

    if mode == "batch":
        if drop_last:
            # Equally-sized splits; discards the remainder.
            chunk_size = num_items // split_num
            chunk_sizes = (chunk_size for _ in range(split_num))
        else:
            # If total length can't divided by 'num_splits', get its even chunk_sizes (differ by at most 1).
            chunk_sizes = ichunk_size(num_items, split_num=split_num)

        # [(start, end), ...]
        accumul = list(accumulate(chunk_sizes, initial=0))

        split_indices = (
            (accumul[i - 1], stop) for i, stop in enumerate(accumul[1:], 1)
        )

        splits = [seq[begin:end] for begin, end in split_indices]

    elif mode == "shard":
        if drop_last:
            end = split_num * (num_items // split_num)
        else:
            end = num_items
        splits = [seq[i:end:split_num] for i in range(split_num)]
    else:
        raise NotImplementedError
    return splits


class Dillable:
    """
    Mix-in that will leverage ``dill`` instead of ``pickle``
    when pickling an object.

    It is useful when the user can't avoid ``pickle`` (e.g. in multiprocessing),
    but needs to use unpicklable objects such as lambdas.
    """

    def __getstate__(self):
        import dill

        return dill.dumps(self.__dict__)

    def __setstate__(self, state):
        import dill

        self.__dict__ = dill.loads(state)


def _async_generator(queue: queue.Queue, fn: Callable, kwargs: dict):
    for i, result in enumerate(fn(**kwargs)):
        queue.put(result)
    return i


def iflatmap_unordered(
    nj: int, fn: Callable[..., Iterable], kwargs_iter: Iterable[dict]
) -> Iterable:
    """
    Parrallized mapping operation.

    Note: Data are in kwargs_iter, and flats reults of all jobs to a queue.
    This operation don't keep the original order in async way.

    Args:
        nj (int):
            num of jobs.
        fn (Callable):
            a function can yied results from given args.
        kwargs_iter (Iterable[dict]):
            kwargs map to ``fn``.
    """
    from multiprocess import Manager, Pool  # Support lambda function.

    with Pool(nj) as pool:
        with Manager() as manager:
            queue = manager.Queue()
            async_results = [
                pool.apply_async(_async_generator, (queue, fn, kwargs))
                for kwargs in kwargs_iter
            ]
            while True:
                try:
                    yield queue.get(timeout=0.1)
                except Empty:
                    if (
                        all(async_result.ready() for async_result in async_results)
                        and queue.empty()
                    ):
                        break
            [async_result.get() for async_result in async_results]


def infinite_randint_iter(
    rng: np.random.Generator, buffer_size: int, random_batch_size=1000
) -> Iterator[int]:
    while True:
        yield from (
            int(i) for i in rng.integers(0, buffer_size, size=random_batch_size)
        )


def buffer_shuffle(
    data: Iterable,
    buffer_size: int = 10_000,
    rng: Optional[np.random.Generator] = None,
) -> Generator:
    """
    Buffer shuffle the data.

    Args:
        data (Iterable):
            data source.
        buffer_size (int):
            defaults to 10000.
        rng (np.random.Generator):
            fix random.

    Returns:
        Generator yields data item.
    """
    if rng is None:
        rng = np.random.default_rng()
    rng = copy.deepcopy(rng)
    random_indices_iter = infinite_randint_iter(rng, buffer_size)

    buffer = []
    for x in data:
        # random replace item from buffer
        if len(buffer) == buffer_size:
            i = next(random_indices_iter)
            yield buffer[i]
            buffer[i] = x
        # fills the buffer to a fixed size
        else:
            buffer.append(x)
    # shuffle the left
    rng.shuffle(buffer)
    yield from buffer


def ichunk_size(
    total_len: int,
    split_num: Optional[int] = None,
    chunk_size: Optional[int] = None,
    even: bool = True,
) -> Generator:
    """
    Infer an enven split chunksize generator before applying split operation if needed.

    Args:
        total_len (int):
            The lengths to be divided.
        chunk_size (int):
        split_num (int):
            Number of splits, can be provided to infer chunksizes.
        even (bool):
            If True, the max differ between chunksize is 1.

    Returns:
        A generator yields sizes of total length.

    Example:
        >>> list(ichunk_size(15, chunk_size=10, even=True))  # chunk_size=10, total_len=15, adapt (10, 5) -> (8, 7).
        [8, 7]
        >>> list(ichunk_size(10, split_num=4, even=True))  # split_num=4, total_len=10, adapt chunksize (3, 3, 3, 1) -> (3, 3, 2, 2).
        [3, 3, 2, 2]
    """
    if total_len < 1:
        raise ValueError(f"invalid total_len({total_len}).")
    if chunk_size is not None and split_num is not None:
        raise ValueError(
            f"Can't set both chunk_size and split_num, got split_num={split_num}, chunk_size={chunk_size}."
        )
    elif split_num is not None:
        assert split_num > 0
        q, r = divmod(total_len, int(split_num))
        full_size = q + (1 if r > 0 else 0)
    elif chunk_size is not None:
        assert chunk_size > 0
        q, r = divmod(total_len, int(chunk_size))
        split_num = q + (1 if r > 0 else 0)
        if even:
            q, r = divmod(total_len, split_num)
            full_size = q + (1 if r > 0 else 0)
        else:
            full_size = int(chunk_size)
    else:
        raise ValueError(
            f"Failed to infer chunksize, got invalid args total_len={total_len}, chunk_size={chunk_size} split_num={split_num}."
        )
    if even:
        partial_size = full_size - 1
        num_full = total_len - partial_size * split_num
        num_partial = split_num - num_full
    else:
        num_full = total_len // full_size
        partial_size = total_len - full_size * num_full
        num_partial = split_num - num_full

    # yield full_size for expect time.
    if num_full > 0:
        for _ in range(num_full):
            yield full_size
    # yield left partial_size.
    if num_partial > 0:
        for _ in range(num_partial):
            yield partial_size


def ilen(iterable):
    """
    Return the number of items in iterable inputs.

    This consumes the iterable data, so handle with care.

    Example:
        >>> ilen(x for x in range(1000000) if x % 3 == 0)
        333334
    """
    counter = count()
    deque(zip(iterable, counter), maxlen=0)
    return next(counter)


def uniq_key(keys):
    seen = set()
    for key in keys:
        if key not in seen:
            seen.add(key)
            yield key


def zip_dict(*dicts):
    """
    Iterate over items of dictionaries grouped by their keys.
    """
    for key in uniq_key(chain(*dicts)):
        # Will raise KeyError if the dict don't have the same keys
        yield key, tuple(d[key] for d in dicts)


@dataclass
class ClassLabel(DictFileMixin):
    """
    The instance of this class stores the string names of labels,
    can be used for mapping str2label or label2str.

    Modified from `HuggingFace Datasets
    <https://github.com/huggingface/datasets/blob/main/src/datasets/features/features.py#ClassLabel>`_.

    There are 3 ways to define a ``ClassLabel``, which correspond to the 3 arguments:
        * ``num_classes``: Create 0 to (num_classes-1) labels.
        * ``names``: List of label strings.
        * ``names_file``: File (Text) containing the list of labels.

    Under the hood the labels are stored as integers.
    You can use negative integers to represent unknown/missing labels.

    Serialize/deserialize of yaml files will be in a more readable way (``from_yaml``, '`to_yaml``):

        names:                  ->                names:

        - negative              ->                  '0': negative
        - positive              ->                  '1': positive

    Args:
        num_classes (`int`, *optional*):
            Number of classes. All labels must be ``< num_classes``.
        names (`list` of `str`, *optional*):
            String names for the integer classes.
            The order in which the names are provided is kept.
        names_file (`str`, *optional*):
            Path to a file with names for the integer classes, one per line.

    Example:
        >>> label = ClassLabel(num_classes=3, names=['speaker1', 'speaker2', 'speaker3'])
        >>> label
        ClassLabel(num_classes=3, names=['speaker1', 'speaker2', 'speaker3'], id=None)
        >>> label.encode_label('speaker1')
        1
        >>> label.encode_label(1)
        1
        >>> label.encode_label('1')
        1
    """

    num_classes: InitVar[
        Optional[int]
    ] = None  # Pseudo-field: ignored by asdict/fields when converting to/from dict
    names: List[str] = None
    names_file: InitVar[
        Optional[str]
    ] = None  # Pseudo-field: ignored by asdict/fields when converting to/from dict
    id: Optional[str] = None

    _str2int: ClassVar[Dict[str, int]] = None
    _int2str: ClassVar[List[str]] = None

    def __post_init__(self, num_classes, names_file):
        self.num_classes = num_classes
        self.names_file = names_file
        if self.names_file is not None and self.names is not None:
            raise ValueError("Please provide either names or names_file but not both.")
        # Set self.names
        if self.names is None:
            if self.names_file is not None:
                self.names = self._load_names_from_file(self.names_file)
            elif self.num_classes is not None:
                self.names = [str(i) for i in range(self.num_classes)]
            else:
                raise ValueError(
                    "Please provide either num_classes, names or names_file."
                )
        elif not isinstance(self.names, (list, tuple)):
            raise TypeError(f"Please provide names as a list, is {type(self.names)}")
        # Set self.num_classes
        if self.num_classes is None:
            self.num_classes = len(self.names)
        elif self.num_classes != len(self.names):
            raise ValueError(
                "ClassLabel number of names do not match the defined num_classes. "
                f"Got {len(self.names)} names VS {self.num_classes} num_classes"
            )
        # Prepare mappings
        self._int2str = [str(name) for name in self.names]
        self._str2int = {name: i for i, name in enumerate(self._int2str)}
        if len(self._int2str) != len(self._str2int):
            raise ValueError(
                "Some label names are duplicated. Each label name should be unique."
            )

    def encode_label(self, example_data):
        if self.num_classes is None:
            raise ValueError(
                "Trying to use ClassLabel with undefined number of class. "
                "Please set ClassLabel.names or num_classes."
            )

        # If a string is given, convert to associated integer
        if isinstance(example_data, str):
            example_data = self.str2int(example_data)

        # Allowing -1 to mean no label.
        if not -1 <= example_data < self.num_classes:
            raise ValueError(
                f"Class label {example_data:d} greater than configured num_classes {self.num_classes}"
            )
        return example_data

    def str2int(self, values: Union[str, Iterable]) -> Union[int, Iterable]:
        """
        Conversion class name ``string`` => ``integer``.

        Example:
            >>> label = ClassLabel(num_classes=3, names=['speaker1', 'speaker2', 'speaker3'])
            >>> label.str2int('speaker1')
            0
        """
        if not isinstance(values, str) and not isinstance(
            values, collections.abc.Iterable
        ):
            raise ValueError(
                f"Values {values} should be a string or an Iterable (list, numpy array, pytorch, tensorflow tensors)"
            )
        return_list = True
        if isinstance(values, str):
            values = [values]
            return_list = False

        output = [self._strval2int(value) for value in values]
        return output if return_list else output[0]

    def _strval2int(self, value: str) -> int:
        failed_parse = False
        value = str(value)
        # first attempt - raw string value
        int_value = self._str2int.get(value)
        if int_value is None:
            # second attempt - strip whitespace
            int_value = self._str2int.get(value.strip())
            if int_value is None:
                # third attempt - convert str to int
                try:
                    int_value = int(value)
                except ValueError:
                    failed_parse = True
                else:
                    if int_value < -1 or int_value >= self.num_classes:
                        failed_parse = True
        if failed_parse:
            raise ValueError(f"Invalid string class label {value}")
        return int_value

    def int2str(self, values: Union[int, Iterable]) -> Union[str, Iterable]:
        """
        Conversion ``integer`` => class name ``string``.

        Regarding unknown/missing labels: passing negative integers raises ``ValueError``.

        Example:
            >>> label = ClassLabel(num_classes=3, names=['speaker1', 'speaker2', 'speaker3'])
            >>> label.int2str(0)
            'speaker1'
        """
        if not isinstance(values, int) and not isinstance(
            values, collections.abc.Iterable
        ):
            raise ValueError(
                f"Values {values} should be an integer or an Iterable (list, numpy array, pytorch, tensorflow tensors)"
            )
        return_list = True
        if isinstance(values, int):
            values = [values]
            return_list = False

        for v in values:
            if not 0 <= v < self.num_classes:
                raise ValueError(f"Invalid integer class label {v:d}")

        output = [self._int2str[int(v)] for v in values]
        return output if return_list else output[0]

    @staticmethod
    def _load_names_from_file(names_filepath):
        with open(names_filepath, encoding="utf-8") as f:
            return [
                name.strip() for name in f.read().split("\n") if name.strip()
            ]  # Filter empty names

    def to_dict(self):
        output = asdict_filt(self)
        output["names"] = {
            str(label_id): label_name
            for label_id, label_name in enumerate(output["names"])
        }
        return output

    @classmethod
    def from_dict(cls, data: Dict):
        """
        Deserialize from dict.

        transform dict to list:

          names:                ->                  names:

          '0': negative         ->                  - negative

          '1': positive         ->                  - positive
        """

        if isinstance(data.get("names"), dict):
            label_ids = sorted(data["names"], key=int)
            if label_ids and [int(label_id) for label_id in label_ids] != list(
                range(int(label_ids[-1]) + 1)
            ):
                raise ValueError(
                    f"ClassLabel expected a value for all label ids [0:{int(label_ids[-1]) + 1}] but some ids are missing."
                )
            data["names"] = [data["names"][label_id] for label_id in label_ids]
        field_names = {f.name for f in fields(cls)}
        return cls(**{k: v for k, v in data.items() if k in field_names})

    def to_file(self, path: Union[Path, str]):
        """
        Serialize to a dict file.

        names:                  ->                names:

        - negative              ->                  '0': negative
        - positive              ->                  '1': positive
        """
        path = Path(path)
        if not path.suffix:
            path = path.with_suffix(".yaml")
        super().to_file(path)
        return path

    @classmethod
    def from_file(cls, path: Union[Path, str]) -> "ClassLabel":
        path = Path(path)
        suffix = path.suffix
        if suffix in (".yaml", ".json", ".yml"):
            return cast(cls, super().from_file(path))
        warnings.warn(f"{str(path)} is treated as a normal text file.")
        return cls(names_file=path)


@dataclass
class SplitInfo:
    """A container of split dataset info."""

    name: Union[str, Split] = ""
    patterns: Union[str, List[str]] = ""
    num_examples: int = 0
    meta: Optional[Dict] = None

    def __post_init__(self):
        self.name = str(self.name)

    def to_dict(self):
        return asdict_filt(self)


class SplitInfoDict(dict, DictFileMixin):
    def __getitem__(self, key: Union[Split, str]):
        return super().__getitem__(str(key))

    def __setitem__(self, key: Union[Split, str], value: SplitInfo):
        if key != value.name:
            raise ValueError(
                f"Cannot add elem. (key mismatch: '{key}' != '{value.name}')"
            )
        if key in self:
            raise ValueError(f"Split {key} already present")
        super().__setitem__(key, value)

    @classmethod
    def from_dict(cls, split_infos: Dict) -> "SplitInfoDict":
        split_dict = cls()
        split_infos = copy.deepcopy(split_infos)
        for key, split_info in split_infos.items():
            if isinstance(split_info, dict):
                split_info = SplitInfo(**split_info)
            if not split_info.name:
                split_info.name = str(key)
            split_dict[split_info.name] = split_info
        return split_dict

    def to_dict(self) -> Dict:
        split_dict = {}
        for split_info in self.values():
            d = asdict_filt(split_info)
            name = d.pop("name")
            split_dict[name] = d
        return split_dict

    @property
    def total_num_examples(self):
        """Return the total number of examples."""
        return sum(s.num_examples for s in self.values())


def get_num_batch(n: int, batch_size: int, drop_last: bool = False):
    if drop_last:
        return n // batch_size
    else:
        return (n + batch_size - 1) // batch_size


def try_length(data) -> Optional[int]:
    """Try to get the length of an object, fallback to return ``None``."""
    try:
        # try getting the length
        length = len(data)  # type: ignore [arg-type]
    except (TypeError, NotImplementedError):
        length = None
    return length


def wavscp2dicts(
    wav_file, col_utt_name="id", col_path_name="audio_path"
) -> List[Dict[str, str]]:
    """Read wav.scp to a list of dict"""
    wav_list = []
    with open(wav_file, "r", encoding="utf8") as fin:
        for line in fin:
            arr = line.strip().split()
            assert len(arr) == 2
            item = {}
            utt = arr[0]
            wav_path = arr[1]
            item[col_utt_name] = utt
            item[col_path_name] = wav_path
            wav_list.append(item)
    return wav_list
