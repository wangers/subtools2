# -*- coding:utf-8 -*-
# Copyright xmuspeech (Author: Leo 2023-07)


import inspect
from typing import Callable, Iterable, List, Union

from torch.utils.data import IterableDataset

from egrecho.data.iterable.sampler import BaseDistSampler
from egrecho.data.processors import partition_one
from egrecho.utils.imports import is_package_available


class IterabelDatasetWrapper(IterableDataset):
    r"""
    A pytorch dataset wrapper to iterate the given sampler.

    Args:
        sampler:
            a iterator can yield samples.

    Example:
        >>> from egrecho.data.iterable import IterDistSampler
        >>> from torch.utils.data import IterableDataset
        >>> data_source = range(10)
        >>> sampler = IterDistSampler(data_source, shuffle=True)
        >>> dataset = IterabelDatasetWrapper(sampler)
        >>> assert isinstance(dataset, IterableDataset)
        >>> list(dataset)
        [5, 6, 0, 7, 3, 2, 4, 9, 1, 8]
    """

    def __init__(self, sampler: BaseDistSampler):
        self.sampler = sampler
        self.epoch = self.sampler.epoch

    def set_epoch(self, epoch: int) -> None:
        self.epoch = epoch
        self.sampler.set_epoch(epoch)

    def __iter__(self):
        yield from iter(self.sampler)

    @property
    def src_nsamples(self):
        if not hasattr(self, "_src_nsamples"):
            self._src_nsamples = None
        return self._src_nsamples

    @src_nsamples.setter
    def src_nsamples(self, n: Union[List[int], int]):
        self._src_nsamples = n

    def src_length(self):
        if self.src_nsamples is None:
            try:
                src_length = len(self.sampler.data_source)

            except Exception:  # flake8: ignore
                return
        else:
            src_length = self.src_nsamples

        if self.sampler.partition and isinstance(src_length, (list, tuple)):
            dist_shard_sizes = []

            dist_shard_sizes = [
                sum(list(partition_one(src_length, i, self.sampler.dist_world_size)))
                for i in range(self.sampler.dist_world_size)
            ]
            return min(dist_shard_sizes)
        elif self.sampler.partition:
            return src_length // self.sampler.dist_world_size
        elif isinstance(src_length, (list, tuple)):
            return sum(src_length)
        else:
            return src_length


class Processor(IterableDataset):
    r"""
    Allows applying lazy chain-like transforms on iterable dataset.

    With a Callable adapter, which can be a generator function or a class with "__iter__" and "__next__" attributes,
    a iterator can be transformed to another. This transform occurs when the `__iter__` of this `Processor` is called.
    Actually it didn't do inner restrict for transform adapter, user should avoid mistakes carefully.
    (e.g., unexpected consuming iterator). Referring to:
        https://github.com/wenet-e2e/wenet/blob/main/wenet/dataset/dataset.py#Processosr

    Args:
        source:
            iterable source.
        adapter:
            transform source iterator lazy.
        *args:
            args for adapter.
        **kw:
            kwargs for adapter.

    Example:
        >>> from egrecho.data.iterable import IterDistSampler, processors, Processor
        >>> from torch.utils.data import IterableDataset
        >>> data_source = range(10)
        >>> sampler = IterDistSampler(data_source, shuffle=True)
        >>> dataset = IterabelDatasetWrapper(sampler)
        >>> list(dataset)
        [5, 6, 0, 7, 3, 2, 4, 9, 1, 8]
        >>> dataset = Processor(dataset, processors.batch, 3)
        >>> list(dataset)
        [[5, 6, 0], [7, 3, 2], [4, 9, 1], [8]]
    """

    def __init__(self, source: Iterable, adapter: Callable, *args, **kw):
        assert callable(adapter)
        self.source = source
        self.adapter = adapter
        self.args = args
        self.kw = kw

    def set_epoch(self, epoch):
        self.source.set_epoch(epoch)

    def __iter__(self):
        """Return an iterator over the source dataset processed by the
        given processor.
        """
        assert self.source is not None
        assert callable(self.adapter)

        if _is_iter_adapter(self.adapter):
            return iter(self.adapter(iter(self.source), *self.args, **self.kw))
        return self.adapter(iter(self.source), *self.args, **self.kw)

    def apply(self, adapter, *args, **kw):
        assert callable(adapter)
        return Processor(self, adapter, *args, **kw)

    @property
    def src_nsamples(self):
        if not hasattr(self, "_src_nsamples"):
            self._src_nsamples = None
        return self._src_nsamples

    @src_nsamples.setter
    def src_nsamples(self, n: int):
        self._src_nsamples = n

    def src_length(self):
        if self.src_nsamples is None:
            try:
                return (self.source).src_length()
            except:  # noqa
                return
        return self.src_nsamples

    def add_length(self, n):
        add_length(self, n)

    def __repr__(self):
        num_samples = None
        try:
            num_samples = self.src_length()
        except Exception:  # flake8: ignore
            pass
        return f"{str(self.__class__.__qualname__)}(adapter: {str(self.adapter.__qualname__)}, src_length: {num_samples})"

    def __getstate__(self):
        if is_package_available("dill"):
            import dill

            return dill.dumps(self.__dict__)
        else:
            return self.__dict__

    def __setstate__(self, state):
        if is_package_available("dill"):
            import dill

            self.__dict__ = dill.loads(state)
        else:
            self.__dict__ = state


def _is_iter_adapter(adapter) -> bool:
    """
    A simple function to detect iterable class. not strictly right cause the class may
    have no valid "__next__" function.
    """
    if not inspect.isclass(adapter) or not hasattr(adapter, "__iter__"):
        return False
    if not inspect.isfunction(adapter.__iter__):
        return False
    return True


def add_length(obj, n):
    """Adds a __len__ method to `IterableDataset` and has no effect on real iteration."""

    def length(self):
        return n

    Combined = type(
        obj.__class__.__name__ + "_Length",
        (obj.__class__, IterableDataset),
        {"__len__": length},
    )
    obj.__class__ = Combined
    return obj
