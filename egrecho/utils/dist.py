# -*- coding:utf-8 -*-
# Copyright xmuspeech (Author: Leo 2023-06)

import itertools
import os
import sys
import textwrap
from contextlib import closing, suppress
from dataclasses import dataclass
from typing import Any, Callable, Dict, Literal, Optional

import torch

from egrecho.utils.imports import torch_dist_is_available


def is_main_rank():
    if use_ddp():
        if torch.distributed.get_rank() == 0:
            return True
        else:
            return False
    return True


def use_ddp():
    return torch_dist_is_available() and torch.distributed.is_initialized()


def cleanup_ddp():
    if use_ddp():
        torch.distributed.destroy_process_group()


def send_exit_ddp(stop_flag=0):
    """reduce stop signal across ddp progresses, controled by main rank."""
    from egrecho.utils.cuda_utils import get_current_device

    if use_ddp():
        flag_tensor = torch.zeros(1).to(get_current_device())
        if is_main_rank():
            flag_tensor += stop_flag
        torch.distributed.all_reduce(flag_tensor, op=torch.distributed.ReduceOp.SUM)
        torch.distributed.barrier()
        stop_flag = flag_tensor
    return stop_flag


def init_nccl_ddp():
    if not torch_dist_is_available():
        raise RuntimeError("torch.distributed is not available.")
    if not torch.distributed.is_nccl_available():
        raise RuntimeError("NCCL is not available.")
    torch.distributed.init_process_group(backend="nccl")


def get_free_port():
    """
    Select a free port for localhost.

    Useful in single-node training when we don't want to connect to a real main node but have to set the
    `MASTER_PORT` environment variable.
    """
    import socket

    with closing(socket.socket(socket.AF_INET, socket.SOCK_STREAM)) as s:
        # Set port as 0, socket will auto-select a free port. And then fetch this port.
        s.bind(("", 0))
        return s.getsockname()[1]


def is_port_in_use(port: int = None) -> bool:
    """
    Checks if a port is in use on `localhost`.
    """
    import socket

    port = port or 29500
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        return s.connect_ex(("localhost", port)) == 0


@dataclass
class DistInfo:
    """
    Contains the environment for the current dist rank.
    """

    world_size: int
    rank: int

    @classmethod
    def detect(cls, group=None, allow_env: bool = True) -> "DistInfo":
        """Tries to automatically detect the pytorch distributed environment paramters.

        Note:
            If `allow_env = True`, some other dist environment may be detected.
            This detection may not work in processes spawned from the distributed processes (e.g. DataLoader workers)
            as the distributed framework won't be initialized there.
            It will default to 1 distributed process in this case.
        """
        if allow_env and ("RANK" in os.environ and "WORLD_SIZE" in os.environ):
            rank = int(os.environ["RANK"])
            world_size = int(os.environ["WORLD_SIZE"])
        elif use_ddp():
            group = group or torch.distributed.group.WORLD
            world_size = torch.distributed.get_world_size(group)
            rank = torch.distributed.get_rank(group)
        else:
            world_size = None
            rank = 0

        if world_size is None or world_size == -1:
            world_size = 1

        return cls(world_size=world_size, rank=rank)


@dataclass
class WorkerInfo:
    """
    Contains the environment for the current dataloader within the current training process.
    """

    num_workers: int
    id: int

    @classmethod
    def detect(cls, allow_env: bool = True) -> "WorkerInfo":
        """Automatically detects the number of pytorch workers and the current rank.

        Note:
            If `allow_env = True`, some other worker environment may be detected.
            This only works reliably within a dataloader worker as otherwise the necessary information won't be present.
            In such a case it will default to 1 worker
        """

        if allow_env and ("WORKER" in os.environ and "NUM_WORKERS" in os.environ):
            id = int(os.environ["WORKER"])
            num_workers = int(os.environ["NUM_WORKERS"])
        else:
            worker_info = torch.utils.data.get_worker_info()
            num_workers = worker_info.num_workers if worker_info is not None else 1
            id = worker_info.id if worker_info is not None else 0

        return cls(id=id, num_workers=num_workers)


@dataclass
class EnvInfo:
    """
    Container of DistInfo and WorkerInfo.
    """

    dist_info: Optional[DistInfo] = None
    worker_info: Optional[WorkerInfo] = None

    @classmethod
    def from_args(
        cls,
        world_size: int,
        rank: int,
        num_workers: int,
        worker_id: int,
    ) -> "EnvInfo":
        """Set env info from args.

        Args:
            world_size:
                The worldsize used for distributed training (=total number of distributed processes)
            rank:
                The distributed global rank of the current process
            num_workers:
                The number of workers per distributed training process
            worker_id:
                The rank of the current worker within the number of workers of
                the current training process
        """
        dist_info = DistInfo(world_size, rank)
        worker_info = WorkerInfo(num_workers, worker_id)
        return cls(dist_info=dist_info, worker_info=worker_info)

    @property
    def num_shards(self) -> int:
        """Returns the total number of shards.

        Note:
            This may not be accurate in a non-dataloader-worker process like the main training process
            as it doesn't necessarily know about the number of dataloader workers.
        """
        assert self.worker_info is not None
        assert self.dist_info is not None
        return self.dist_info.world_size * self.worker_info.num_workers

    @property
    def shard_rank(self) -> int:
        """Returns the rank of the current process wrt. the total number of shards.

        Note:
            This may not be accurate in a non-dataloader-worker process like the main training process as it
            doesn't necessarily know about the number of dataloader workers.
        """
        assert self.worker_info is not None
        assert self.dist_info is not None
        return self.dist_info.rank * self.worker_info.num_workers + self.worker_info.id

    def __repr__(self) -> str:
        return (
            f"{self.__class__.__name__}(\n\t{DistInfo},\n\t"
            + f"{WorkerInfo}\n\tnum_shards: {self.num_shards},\n\tshard_rank: {self.shard_rank})"
        )


class TorchMPLauncher:
    r"""Launches processes that run a given function in parallel, and joins them all at the end.

    Worker processes gives a rank to os.envrion["LOCAL_RANK"] that ranges from 0 to N - 1.

    Referring to ``lightning/fabric``:
        https://github.com/Lightning-AI/lightning/blob/master/src/lightning/fabric/strategies/launchers/multiprocessing.py

    Note:
        - This launcher requires all objects to be pickleable.
        - Entry point to the program/script should guarded by ``if __name__ == "__main__"``.
        - In environments like Ipython notebooks where 'spawn' is not available, 'fork' works better.
        - Start method 'fork' the user must ensure that no CUDA context gets created in the main process before
          the launcher is invoked, i.e., torch.cuda should be uninitialized.

    Args:
        num_processes: number works.
        port: master port, if not set, will auto find in localhost.
        disable_mem_share: mem_share is the feature of torch.multiprocessing.
            Required set True when running models on CPU, see: method::`_disable_module_memory_sharing`.
        start_method: The method how to start the processes.

    Example::

        launcher = TorchMPLauncher(num_processes=4)
        launcher.launch(my_function, arg1, arg2, kwarg1=value1)
    """

    def __init__(
        self,
        num_processes: int,
        port: Optional[int] = None,
        disable_mem_share: bool = False,
        start_method: Literal["spawn", "fork", "forkserver"] = "spawn",
    ) -> None:
        self.num_processes = num_processes
        self.port = port
        self.disable_mem_share = disable_mem_share
        self._start_method = start_method
        if start_method not in torch.multiprocessing.get_all_start_methods():
            raise ValueError(
                f"The start method '{self._start_method}' is not available on this platform. Available methods are:"
                f" {', '.join(torch.multiprocessing.get_all_start_methods())}"
            )

    def launch(self, function: Callable, *args: Any, **kwargs: Any) -> Any:
        """Launches processes that run the given function in parallel.

        The function is allowed to have a return value. However, when all processes join, only the return value
        of worker process 0 gets returned from this `launch` method in the main process.

        Arguments:
            function: The entry point for all launched processes.
            *args: Optional positional arguments to be passed to the given function.
            **kwargs: Optional keyword arguments to be passed to the given function.
        """
        if self._start_method in ("fork", "forkserver"):
            if torch.cuda.is_initialized():
                raise RuntimeError(
                    f"Find torch.cuda has been initiated, {self._start_method} of `torch.multiprocessing` "
                    "can't create processes."
                )
        if self._start_method == "spawn":
            self._check_missing_main_guard()
        os.environ["MASTER_PORT"] = str(
            self.port if self.port is not None else get_free_port()
        )
        context = torch.multiprocessing.get_context(self._start_method)
        return_queue = context.SimpleQueue()

        process_args = [function, args, kwargs, return_queue]

        process_context = torch.multiprocessing.start_processes(
            self._wrap_func,
            args=process_args,
            nprocs=self.num_processes,
            start_method=self._start_method,
            join=False,
        )
        self.procs = process_context.processes
        while not process_context.join():
            pass

        return return_queue.get()

    def _wrap_func(
        self,
        process_idx: int,
        function: Callable,
        args: Any,
        kwargs: Any,
        return_queue,
    ) -> None:
        from egrecho.utils.cuda_utils import to_device

        if self._start_method == "spawn" and self.disable_mem_share:
            args, kwargs = _disable_module_memory_sharing((args, kwargs))
        _set_num_threads_if_needed(num_processes=self.num_processes)
        os.environ["LOCAL_RANK"] = str(process_idx)

        if process_idx == 0:
            try:
                results = function(*args, **kwargs)
                return_queue.put(to_device(results, "cpu"))
            except KeyboardInterrupt:
                self.kill()
                raise
        else:
            results = function(*args, **kwargs)

    def kill(self, signum) -> None:
        for proc in self.procs:
            if proc.is_alive() and proc.pid is not None:
                sys.stdout.write(
                    f"pid {os.getpid()} killing {proc.pid} with {signum}\n"
                )
                sys.stdout.flush()

                with suppress(ProcessLookupError):
                    os.kill(proc.pid, signum)

    @staticmethod
    def _check_missing_main_guard() -> None:
        """Raises an exception if the ``__name__ == "__main__"`` guard is missing."""
        if not getattr(torch.multiprocessing.current_process(), "_inheriting", False):
            return
        message = textwrap.dedent(
            """
            Launching multiple processes with the 'spawn' start method requires that your script guards the main
            function with an `if __name__ == \"__main__\"` clause. For example:

            def main():
                # Put your code here
                ...

            if __name__ == "__main__":
                main()

            Alternatively, you can run with `strategy="ddp"` to avoid this error.
            """
        )
        raise RuntimeError(message)

    def __getstate__(self) -> Dict:
        state = self.__dict__.copy()
        state["procs"] = []  # SpawnProcess can't be pickled
        return state


def _disable_module_memory_sharing(data: Any) -> Any:
    """Disables memory sharing on parameters and buffers of `nn.Module`s contained in the given collection.

    Note: This is only required when running on CPU.

    """
    # PyTorch enables memory sharing automatically on all tensors that are passed through `mp.spawn`.
    # For model weights and buffers, this is undesired and can lead to race conditions between processes.
    # Hence, we copy the tensors in the entire module to ensure it doesn't share memory with other processes.
    from egrecho.utils.apply import apply_to_collection

    @torch.no_grad()
    def unshare(module: torch.Module) -> torch.Module:
        for tensor in itertools.chain(module.parameters(), module.buffers()):
            tensor.data = tensor.data.clone()
        return module

    return apply_to_collection(data, function=unshare, dtype=torch.Module)


def _num_cpus_available() -> int:
    if hasattr(os, "sched_getaffinity"):
        return len(os.sched_getaffinity(0))

    cpu_count = os.cpu_count()
    return 1 if cpu_count is None else cpu_count


def _suggested_max_num_threads(num_processes: int = 1) -> int:
    if num_processes < 1:
        raise ValueError(f"`num_processes` should be >= 1, got {num_processes}.")
    try:
        import psutil

        cpu_available = psutil.cpu_count(logical=False)
        cpu_available = 1 if cpu_available is None else cpu_available
    except ImportError:
        cpu_available = _num_cpus_available()
    return max(1, cpu_available // num_processes)


def _set_num_threads_if_needed(num_processes: int = 1) -> None:
    if "OMP_NUM_THREADS" not in os.environ:
        num_threads = _suggested_max_num_threads(num_processes)
        torch.set_num_threads(num_threads)
        os.environ["OMP_NUM_THREADS"] = str(num_threads)
