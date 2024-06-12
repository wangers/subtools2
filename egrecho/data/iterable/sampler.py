# -*- coding:utf-8 -*-
# Copyright xmuspeech (Author: Leo 2023-07)

from typing import Iterable, Optional

import numpy as np
from torch.utils.data import Sampler

from egrecho.data.dew import DewSamples
from egrecho.data.processors import buffer_shuffle, partition_one
from egrecho.utils.dist import DistInfo, EnvInfo, WorkerInfo
from egrecho.utils.logging import get_logger

logger = get_logger()


class BaseDistSampler(Sampler):
    r"""
    To make dataloading more explicitly, we use DistSampler to handle shuffling, sharding
    data source accross distributed process, etc.

    Args:
        shuffle (bool, optional):
            If ``True`` (default), sampler will shuffle the indices.
        partition (bool, optional):
            Whether apply sharding according to the env. Defaults to False
        env (EnvInfo, optional):
            DistInfo and WorkerInfo used to apply sharding across
            different processes, will try to detect by default.
        seed (int, optional):
            random seed used to shuffle the sampler if :attr:`shuffle=True`.
            This number should be identical across all
            processes in the distributed group. Default: ``42``.
    """

    def __init__(
        self,
        shuffle: bool = True,
        partition: bool = False,
        env: Optional[EnvInfo] = None,
        seed: int = 42,
    ):
        self.shuffle = shuffle
        self.seed = seed
        self.partition = partition
        self.epoch = 0
        if env and env.dist_info:
            logger.info_once(
                f"Sharding data use cunstom dist env: {env.dist_info}.", ranks=[0]
            )
            self._custom_dist = True
            self._env = env
        elif env and env.worker_info:
            self._env = env
            self.env.dist_info = DistInfo.detect()
            self._custom_dist = False
        else:
            # should save distinfo here as some cases（e.g., mp.spwan）don't duplicate
            # distinfo from parent process after multi workers run.
            dist_info = DistInfo.detect()
            env = EnvInfo(dist_info, None)

            logger.info_once(f"auto detect dist info: {dist_info}.", ranks=[0])
            self._env = env
            self._custom_dist = False
        self._custom_worker = self.env.worker_info is not None

    def set_dist_info(self, dist_info: Optional[DistInfo] = None):
        """
        Some case (lazy initiate ddp), dist info can be set before epoch loops.
        """
        if dist_info is not None:
            self.env.dist_info = dist_info
            self._custom_dist = True
        else:
            if not self._custom_dist:
                self.env.dist_info = DistInfo.detect()

    def _auto_check_worker_info(self):
        """
        Try to check local env in multi workers, can be called when start iterating a new epoch.
        """
        if not self._custom_worker:
            self.env.worker_info = WorkerInfo.detect()

    def __iter__(self):
        raise NotImplementedError(
            "Sub-classes of BaseDistSampler have to implement __iter__()"
        )

    def set_epoch(self, epoch):
        self.epoch = epoch

    @property
    def num_shards(self):
        return self._env.num_shards

    @property
    def shard_rank(self):
        return self._env.shard_rank

    @property
    def worker_id(self):
        return self._env.worker_info.id

    @property
    def num_workers(self):
        return self._env.worker_info.num_workers

    @property
    def dist_rank(self):
        return self.env.dist_info.rank

    @property
    def dist_world_size(self):
        return self.env.dist_info.world_size

    @property
    def env(self):
        return self._env


class EgrechoDistSampler(BaseDistSampler):
    r"""
    Provide `DewSamples` as data source.

    Args:
        data_source:
            `DewSamples` used for sampling and iterating.
        shuffle (bool, optional):
            If ``True`` (default), sampler will shuffle the
            indices.
        partition (bool, optional):
            Whether apply sharding according to the env. Defaults to True
        env (EnvInfo, optional):
            DistInfo and WorkerInfo used to apply sharding across
            different processes, will try to detect by default.
        buffer_shuffle_size (int, optional):
            The size of cache to be shuffled of some cases (i.e., data source is iterator)
            which need lazy shuffle. Defaults to 20000.
        seed (int, optional):
            random seed used to shuffle the sampler if :attr:`shuffle=True`.
            This number should be identical across all
            processes in the distributed group. Default: ``42``.
    """

    def __init__(
        self,
        data_source: DewSamples,
        shuffle: bool = True,
        partition: bool = True,
        buffer_shuffle_size: int = 20_000,
        seed: int = 42,
        env: Optional[EnvInfo] = None,
    ):
        super().__init__(
            shuffle=shuffle,
            partition=partition,
            env=env,
            seed=seed,
        )
        if not isinstance(data_source, DewSamples):
            raise TypeError(
                f"{self.__class__.__name__} is used for egrecho {DewSamples.__name__}, "
                f"but got {type(data_source)}"
            )
        self.data_source = data_source
        self._ready_data = self.data_source
        self.buffer_shuffle_size = buffer_shuffle_size
        self._iterator = None

    def _maybe_shuffle(self):
        if self.shuffle:
            rng = np.random.default_rng(self.seed + self.epoch)
            self._ready_data = self.data_source.shuffle(
                rng=rng, buffer_size=self.buffer_shuffle_size
            )
        return self

    def __iter__(self):
        self._maybe_shuffle()
        self._iterator = iter(self._ready_data)
        self._auto_check_worker_info()
        if self.partition:
            if self.shard_rank >= self.num_shards or self.shard_rank < 0:
                raise ValueError(
                    f"Invalid rank {self.shard_rank}, rank should be in the interval"
                    f" [0, {self.num_shards-1}]"
                )
            self._iterator = partition_one(
                self._iterator, self.shard_rank, self.num_shards
            )
        else:
            self._iterator = partition_one(
                self._iterator, self.worker_id, self.num_workers
            )
        return self._iterator

    @property
    def is_lazy(self):
        return self.data_source.is_lazy


class IndexDistSampler(BaseDistSampler):
    r"""
    Provide eager-stype data source.

    Args:
        data_source:
            eager-stype data source.
        shuffle (bool, optional):
            If ``True`` (default), sampler will shuffle the
            indices.
        partition (bool, optional):
            Whether apply sharding according to the env. Defaults to True
        env (EnvInfo, optional):
            DistInfo and WorkerInfo used to apply sharding across
            different processes, will try to detect by default.
        seed (int, optional):
            random seed used to shuffle the sampler if :attr:`shuffle=True`.
            This number should be identical across all
            processes in the distributed group. Default: ``42``.
    """

    def __init__(
        self,
        data_source: Iterable,
        shuffle: bool = True,
        partition: bool = True,
        seed: int = 42,
        env: Optional[EnvInfo] = None,
    ):
        super().__init__(
            shuffle=shuffle,
            partition=partition,
            env=env,
            seed=seed,
        )
        self.data_source = data_source

        if isinstance(data_source, DewSamples):
            if data_source.is_lazy:
                raise RuntimeError(
                    f"{self.__class__.__name__} is used for eager-style data source, "
                    f"but got lazy {type(data_source)}, change to `IterDistSampler`/`EgrechoDistSampler` instead."
                )
        self._indices = list(range(len(data_source)))

    def _maybe_shuffle(self):
        if self.shuffle:
            rng = np.random.default_rng(self.seed + self.epoch)
            self._indices = rng.shuffle(self._indices)
        return self

    def __iter__(self):
        self._maybe_shuffle()
        self._auto_check_worker_info()
        if self.partition:
            self._indices = list(
                partition_one(self._indices, self.shard_rank, self.num_shards)
            )
        else:
            self._indices = list(
                partition_one(self._indices, self.worker_id, self.num_workers)
            )

        for indice in self._indices:
            yield self.data_source[indice]


class IterDistSampler(BaseDistSampler):
    r"""
    Provide iter-stype data source.

    Args:
        data_source:
            iter-stype data source.
        shuffle (bool, optional):
            If ``True`` (default), sampler will shuffle the
            indices.
        partition (bool, optional):
            Whether apply sharding according to the env. Defaults to True
        env (EnvInfo, optional):
            DistInfo and WorkerInfo used to apply sharding across
            different processes, will try to detect by default.
        buffer_shuffle_size (int, optional):
            The size of cache to be shuffled of some cases (i.e., data source is iterator)
            which need lazy shuffle. Defaults to 20000.
        seed (int, optional):
            random seed used to shuffle the sampler if :attr:`shuffle=True`.
            This number should be identical across all
            processes in the distributed group. Default: ``42``.
    """

    def __init__(
        self,
        data_source: Iterable,
        shuffle: bool = True,
        partition: bool = True,
        buffer_shuffle_size: int = 20_000,
        seed: int = 42,
        env: Optional[EnvInfo] = None,
    ):
        super().__init__(
            shuffle=shuffle,
            partition=partition,
            env=env,
            seed=seed,
        )
        if isinstance(data_source, DewSamples):
            if not data_source.is_lazy:
                logger.warning(
                    f"{self.__class__.__name__} is used for iter-style data source, "
                    f"but got eager {type(data_source)}, it will not be full random while shuffling.",
                    ranks=[0],
                )

        self.data_source = data_source
        self._ready_data = self.data_source
        self.buffer_shuffle_size = buffer_shuffle_size
        self._iterator = None

    def _maybe_shuffle(self):
        if self.shuffle:
            rng = np.random.default_rng(self.seed + self.epoch)
            self._ready_data = buffer_shuffle(
                self.data_source, rng=rng, buffer_size=self.buffer_shuffle_size
            )
        return self

    def __iter__(self):
        self._maybe_shuffle()
        self._iterator = iter(self._ready_data)
        self._auto_check_worker_info()
        if self.partition:
            if self.shard_rank >= self.num_shards or self.shard_rank < 0:
                raise ValueError(
                    f"Invalid rank {self.shard_rank}, rank should be in the interval"
                    f" [0, {self.num_shards-1}]"
                )
            self._iterator = partition_one(
                self._iterator, self.shard_rank, self.num_shards
            )
        else:
            self._iterator = partition_one(
                self._iterator, self.worker_id, self.num_workers
            )
        return self._iterator
