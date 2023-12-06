# -*- coding:utf-8 -*-
# Copyright xmuspeech (Author: Leo 2023-07)

import threading
from collections import deque
from concurrent.futures import Future, ThreadPoolExecutor, TimeoutError
from dataclasses import dataclass
from functools import partial
from typing import Any, Callable, Deque, Iterator, Optional, TypeVar

import torch
import torch.distributed as dist
from torch.utils.data.dataloader import DataLoader as _DataLoader
from torch.utils.data.dataloader import _MultiProcessingDataLoaderIter

from egrecho.utils.dist import DistInfo

T_co = TypeVar("T_co", covariant=True)

default_timeout_in_s = 10 * 60


class SyncDataLoader(_DataLoader):
    r"""
    A decorator of pytorch `DataLoader`, extended for iterable dataset.

    Mainly to support synchronizing data across distributed processes to prevent hanging
    during training. referring to `torchdata`:
        https://github.com/pytorch/data/blob/main/torchdata/datapipes/iter/util/distributed.py

    Args:
        args & kwargs:
            See pytorch `DataLoader`.

            Additional:
                sync_timeout (int, optional): Timeout for waiting data in seconds. Default value equals to 10 min.
                fullsync (bool, optional): synchronizs distributed data. Defaults to True.

    Example:
        >>> # On each spawned worker
        >>> import torch.nn.parallel.DistributedDataParallel as DDP
        >>> def worker(rank):
        >>>     dist.init_process_group("nccl", rank=rank, world_size=2)
        >>>     # Rank 1 gets one more input than rank 0
        >>>     inputs = [torch.tensor([1.]).to(rank) for _ in range(10 + rank)]
        >>>     model = DDP(torch.nn.Linear(1, 1).to(rank), device_ids=[rank])
        >>>     dataloader = DataLoader(inputs, num_workers=2)
        >>>     for input in dataloader:
        >>>         loss = model(input).sum()
        >>>         loss.backward()
        >>>         # All ranks reach here without hanging/erroring, got 10 items.
    """

    def __init__(self, *args: Any, **kwargs: Any):
        self.sync_timeout: int = kwargs.pop("sync_timeout", default_timeout_in_s)
        self.fullsync: bool = kwargs.pop("fullsync", True)
        super().__init__(*args, **kwargs)

        self.fullsyn_ext = None
        if self.fullsync:
            dist_info = DistInfo.detect(allow_env=False)
            if dist_info.world_size > 1:
                self.fullsyn_ext = FullSync(sync_timeout=self.sync_timeout)

    def __iter__(self) -> Iterator[T_co]:
        if self.fullsyn_ext is not None:
            self.fullsyn_ext.reset(shutdown_workers=not self.persistent_workers)
            data_source = super().__iter__()
            self.fullsyn_ext.attach_iterator(data_source)

            while True:
                try:
                    batch = self.fullsyn_ext.return_next()
                    yield batch
                except Exception as e:
                    if isinstance(e, StopIteration):
                        self.fullsyn_ext.reset(
                            shutdown_workers=not self.persistent_workers
                        )
                        break
                    else:
                        raise e
        else:
            data_source = super().__iter__()
            for batch in data_source:
                yield batch


class PrefetchTimeoutError(RuntimeError):
    def __init__(self, timeout: int) -> None:
        super().__init__(f"Fail to fetch data within {timeout} seconds")
        self.timeout = timeout


class _EndOfPrefetch:
    ...


@dataclass
class Expected:
    r"""
    Expected data provided to callback function in ``_PrefetchExecutor``.
    """
    index: int
    error: Optional[BaseException] = None

    def has_error(self) -> bool:
        return self.error is not None


class _PrefetchExecutor:
    def __init__(
        self,
        iterator: Iterator,
        prefetch_size: int = 1,
        callback_fn: Optional[Callable[[Expected], None]] = None,
        timeout: int = default_timeout_in_s,
    ) -> None:
        self._iterator = iterator
        self.prefetch_size = prefetch_size
        self.callback_fn = callback_fn
        self.timeout = timeout
        # Use max_workers as 1 to guarantee the order of data fetched from iterator
        self._executor = ThreadPoolExecutor(max_workers=1)
        self._futures: Deque[Future] = deque()
        self._lock = threading.RLock()
        self._end_flag = False
        self._is_shutdown = False
        self._idx = 0
        for _ in range(prefetch_size):
            with self._lock:
                if self._end_flag:
                    break
            fetch_future: Future = self._executor.submit(self.fetch_next)
            fetch_future.add_done_callback(partial(self._done_callback_fn, self._idx))
            self._futures.append(fetch_future)
            with self._lock:
                self._idx += 1

    def fetch_next(self):
        return next(self._iterator)

    def _done_callback_fn(self, index: int, f: Future):
        if f.exception():
            with self._lock:
                self._end_flag = True
        if self.callback_fn is not None:
            self.callback_fn(Expected(index, f.exception()))

    def return_next(self):
        if self._futures:
            fetch_future = self._futures.popleft()
            try:
                data = fetch_future.result(timeout=self.timeout)
            except TimeoutError:
                raise PrefetchTimeoutError(self.timeout)
            with self._lock:
                if not self._end_flag and not self._is_shutdown:
                    next_future = self._executor.submit(self.fetch_next)
                    next_future.add_done_callback(
                        partial(self._done_callback_fn, self._idx)
                    )
                    self._futures.append(next_future)
                    self._idx += 1
        else:
            data = _EndOfPrefetch()
        return data

    def shutdown(self):
        self._is_shutdown = True
        while self._futures:
            self._futures.popleft().cancel()
        self._executor.shutdown(wait=True)


class FullSync:
    def __init__(
        self,
        sync_timeout: int,
    ):
        self._executor: Optional[_PrefetchExecutor] = None
        if not dist.is_available():
            raise RuntimeError(
                "Torch Distributed is required to be available for fullsync."
            )
        self._process_group: Optional[dist.ProcessGroup] = None
        self.sync_timeout = sync_timeout
        self._world_size = 1
        self._lock = threading.RLock()
        self._cv = threading.Condition(lock=self._lock)
        self._error = None
        self._sync_counter = torch.tensor([0], dtype=torch.int32)
        self._done_callback: bool = False

    def attach_iterator(self, iterator):
        self.shutdown()
        self.initiate_group()
        assert self._executor is None
        self._executor = _PrefetchExecutor(
            iterator, 1, self._callback_fn, self.sync_timeout
        )

    def initiate_group(self):
        if not (dist.is_available() and dist.is_initialized()):
            raise RuntimeError("Torch Distributed is required to be initialized")
        if self._process_group is None:
            self._process_group = dist.new_group(backend="gloo")
        self._world_size = dist.get_world_size()

    def _callback_fn(self, exp: Expected) -> None:
        with self._cv:
            if exp.has_error():
                if not isinstance(exp.error, StopIteration):
                    self._error = exp.error  # type: ignore[assignment]
                self._sync_counter = torch.tensor([0], dtype=torch.int32)
            else:
                self._sync_counter = torch.tensor([1], dtype=torch.int32)
            dist.all_reduce(
                tensor=self._sync_counter,
                op=dist.ReduceOp.SUM,
                group=self._process_group,
            )
            self._done_callback = True
            self._cv.notify()

    def return_next(self):
        with self._cv:
            is_success = self._cv.wait_for(
                lambda: self._done_callback is True,
                self.sync_timeout,
            )
            if not is_success:
                raise PrefetchTimeoutError(self.sync_timeout)
            if self._error is not None:
                raise self._error
            if bool(self._sync_counter < self._world_size):
                raise StopIteration
            self._done_callback = False
            data = self._executor.return_next()  # type: ignore[attr-defined]
        if isinstance(data, _EndOfPrefetch):
            raise StopIteration
        return data

    def reset(self, shutdown_workers: bool = False):
        if self._executor is not None:
            # FIXME: how to correctly shutdown workers.
            # if shutdown_workers and self._executor._iterator:
            #     _shutdown_workers(self._executor._iterator)
            self._executor.shutdown()
            self._executor = None
        self._world_size = 1
        with self._cv:
            self._error = None
            self._sync_counter = torch.tensor([0], dtype=torch.int32)
            self._done_callback = False

    def shutdown(self):
        if self._executor is not None:
            self._executor.shutdown()
            self._executor = None


def _shutdown_workers(iterator: Iterator) -> None:
    if isinstance(iterator, _MultiProcessingDataLoaderIter):
        iterator._shutdown_workers()
