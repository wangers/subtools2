# -*- coding:utf-8 -*-
# Copyright xmuspeech (Author: Leo 2023-03)

import collections
import enum
import gc
import os
import shutil
import subprocess
import warnings
from abc import ABC
from contextlib import contextmanager, nullcontext
from functools import lru_cache, partial
from typing import Any, Dict, List, Optional, Sequence, Tuple, Union, cast

import torch

from egrecho.utils.apply import apply_to_collection
from egrecho.utils.imports import _TORCH_GREATER_EQUAL_2_0
from egrecho.utils.logging import get_logger
from egrecho.utils.misc import ConfigurationException

logger = get_logger(__name__)


def avoid_float16_autocast_context():
    """
    If the current autocast context is float16, cast it to bfloat16
    if available (unless we're in jit) or float32
    """

    if torch.is_autocast_enabled() and torch.get_autocast_gpu_dtype() == torch.float16:
        if torch.jit.is_scripting() or torch.jit.is_tracing():
            return torch.amp.autocast('cuda', dtype=torch.float32)

        if torch.cuda.is_bf16_supported():
            return torch.amp.autocast('cuda', dtype=torch.bfloat16)
        else:
            return torch.amp.autocast('cuda', dtype=torch.float32)
    else:
        return nullcontext()


# copied from Nemo: nemo/collections/common/data/lhotse/dataloader.py#maybe_set_cuda_expandable_segments
def maybe_set_cuda_expandable_segments(enabled: bool):
    """
    Configures PyTorch memory allocator to expand existing allocated segments
    instead of re-allocating them when tensor shape grows.
    This can help speed up the training when sequence length and/or batch size change often,
    and makes GPU more robust towards OOM.

    See here for more details:
    https://pytorch.org/docs/stable/notes/cuda.html#optimizing-memory-usage-with-pytorch-cuda-alloc-conf
    """
    if enabled and torch.cuda.is_available():
        if (
            (value := os.environ.get("PYTORCH_CUDA_ALLOC_CONF")) is not None
            and len(value) > 0
            and "expandable_segments:True" not in value
        ):
            warnings.warn(
                "You have set PYTORCH_CUDA_ALLOC_CONF without expandable_segments:True option. We're setting that option anyway. To disable it, set cuda_expandable_segments=False in NeMo dataloader configuration."
            )

        try:
            torch.cuda.memory._set_allocator_settings("expandable_segments:True")
        except RuntimeError:
            logger.warning(
                "Failed to set expandable_segments:True for PyTorch CUDA allocator. You may get training speed improvements if you enable this"
            )


def set_to_cuda(modules):
    """Send modules to gpu.

    Args:
        modules: nn.module or a list of module
    """

    if isinstance(modules, (list, tuple)) and len(modules) > 1:
        ret = []
        for model in modules:
            ret.append(model.to(get_current_device()))
        return ret
    elif isinstance(modules, (list, tuple)):
        return [modules[0].to(get_current_device())]
    else:
        return modules.to(get_current_device())


def get_current_device() -> torch.device:
    """
    Returns currently selected device (gpu/cpu).
    If cuda available, return gpu, otherwise return cpu.
    """
    if is_cuda_available():
        return torch.device(f"cuda:{torch.cuda.current_device()}")
    else:
        return torch.device("cpu")


class _TransferableDtype(ABC):
    """
    A type with method ``.to()`` means it can be transferred to torch device.

    Example:
        >>> isinstance(dict, _TransferableDtype)
        False
        >>> isinstance(torch.rand(2, 3), _TransferableDtype)
        True
        >>> class CustomObject:
        ...     def __init__(self):
        ...         self.x = torch.rand(2, 2)
        ...     def to(self, device):
        ...         self.x = self.x.to(device)
        ...         return self
        >>> isinstance(CustomObject(), _TransferableDtype)
        True
    """

    @classmethod
    def __subclasshook__(cls, subclass: Any) -> Union[bool, Any]:
        if cls is _TransferableDtype:
            to = getattr(subclass, "to", None)
            return callable(to)
        return NotImplemented


_BLOCKING_DEVICE_TYPES = ("cpu", "mps")


def to_device(
    data: Any,
    device_object: Union[torch.device, str, int, torch.Tensor, torch.nn.Module] = None,
) -> Any:
    r"""
    Move a tensor or collection of tensors to a specified device by inferring the device from another object.

    Args:
        data (Any):
            A tensor, collection of tensors, or anything with a ``.to(...)`` method.
        device_object (Union[torch.device, str, int, torch.Tensor, torch.nn.Module], optional):
            The target device. Can be one of the following:

                -   ``torch.device``, ``str``, ``int``, The target device.
                -   ``torch.nn.Module``: Infer the device from a module.
                -   ``torch.Tensor``: Infer the device from a tensor.
                    (default: the current defualt cuda device.)

    Returns:
        The same collection with all contained tensors residing on the target device.
    """

    device_object = device_object or get_current_device()
    if isinstance(device_object, torch.nn.Module):
        device = next(device_object.parameters()).device
    elif isinstance(device_object, torch.Tensor):
        device = device_object.device
    else:
        device = torch.device(device_object)

    def batch_to(data: Any) -> Any:
        kwargs = {}
        # Don't issue non-blocking transfers to CPU
        # Same with MPS due to a race condition bug: https://github.com/pytorch/pytorch/issues/83015
        if (
            isinstance(data, torch.Tensor)
            and isinstance(device, torch.device)
            and device.type not in _BLOCKING_DEVICE_TYPES
        ):
            kwargs["non_blocking"] = True
        data_output = data.to(device, **kwargs)
        if data_output is not None:
            return data_output
        # user wrongly implemented the `_TransferableDataType` and forgot to return `self`.
        return data

    return apply_to_collection(data, dtype=_TransferableDtype, function=batch_to)


def parse_gpus(gpus: Union[str, int, Sequence[int]]) -> Optional[List[int]]:
    """
    Parse gpus option.

    Args:
        gpus:
            GPUs used for training on this machine.

            -1: all;
            N: [0,N);
            "1,2": comma-seperated; "0" means invalid while "0," specify 1 gpu with id:1.
    """
    _check_data_type(gpus)
    if gpus is None:
        return None

    # normalize str to int or list of int
    gpus = _gpus_str2int(gpus)
    gpus = _gpus_int2list(gpus)
    if not gpus:
        return None
    if len(gpus) != len(set(gpus)):
        raise ConfigurationException(f"Got duplicated gpus: {gpus!r}.")
    valid_gpus = [gpu for gpu in range(num_gpus())]
    for gpu in gpus:
        if gpu not in valid_gpus:
            raise ConfigurationException(f"Requested gpu:{gpu}, but it is invalid.")
    return gpus


def parse_gpus_opt(gpus: Optional[Union[str, int]]) -> Optional[List[int]]:
    """Similar to `parse_gpus` but combines auto choose a single GPU.

    Args:
        gpus (Optional[Union[str, int]]): What GPUs should be used:

            -   case 0: comma-separated list, e.g., "1," or "0,1" means specified id(s).
            -   case 1: a single int (str) negative number (-1) means all visible devices ``[0, N-1]``.
            -   case 2: '' or None or 0 returns None means no GPU.
            -   case 3: a single int (str) number equals 1 means auto choose a spare GPU.
            -   case 4: a single int (str) number n greater than 1 returns ``[0, n-1]``.

    Returns:
        Optional[List[int]]: A list of GPU IDs or None.
    """
    _check_data_type(gpus)
    if gpus is None:
        return None

    # normalize str to int or list of int
    gpus = _gpus_str2int(gpus)

    # auto single
    if isinstance(gpus, int) and gpus == 1:
        return [GPUManager.detect()]

    gpus = _gpus_int2list(gpus)
    if not gpus:
        return None
    if len(gpus) != len(set(gpus)):
        raise ConfigurationException(f"Got duplicated gpus: {gpus!r}.")
    valid_gpus = [gpu for gpu in range(num_gpus())]
    for gpu in gpus:
        if gpu not in valid_gpus:
            raise ConfigurationException(f"Requested gpu:{gpu}, but it is invalid.")
    return gpus


def parse_gpu_id(
    gpu_id: Optional[Union[str, int]] = 'auto'
) -> Optional[Union[str, int]]:
    """
    Parse single gpu id option.

    Args:
        gpu_id(Optional[Union[str, int]]): select which GPU:

            -   case 0: "auto", auto select spare gpu.
            -   case 1: a single int (str) negative number (-1) means cpu.
            -   case 2: a single int (str) positive number means specified id.
            -   case 3: '' or None returns None, which means defualt behaviour in same case, e.g., torch.load(...)
            -   case 4: other strings, e.g., "cuda:1"

    """
    if isinstance(gpu_id, str):
        gpu_id = gpu_id.lower().strip()
        try:
            gpu_id = int(gpu_id)
        except Exception as exc:  # noqa
            pass
    if gpu_id == 'auto':
        if not is_cuda_available():
            logger.warning(
                f'Try to auto select gpu but no gpu is available, fallback to cpu.'
            )
            return 'cpu'
        return GPUManager.detect()
    if isinstance(gpu_id, int) and gpu_id < 0:
        return 'cpu'
    elif isinstance(gpu_id, int):
        valid_gpus = [gpu for gpu in range(num_gpus())]
        if gpu_id not in valid_gpus:
            raise ConfigurationException(f"Requested gpu:{gpu_id}, but it is invalid.")
        return gpu_id
    else:
        # '' => None
        return gpu_id or None


def _gpus_str2int(gpus: Union[str, int, Sequence[int]]) -> Union[int, List[int]]:
    if isinstance(gpus, str):
        gpus = gpus.strip()
        if gpus == "-1":
            return -1
        gpus = gpus.replace("-", ",")
        if "," in gpus:
            return [int(x.strip()) for x in gpus.split(",") if len(x.strip()) > 0]
        return int(gpus)
    elif isinstance(gpus, collections.abc.Sequence):
        return [int(str(x).strip()) for x in gpus]
    elif isinstance(gpus, int):
        return gpus
    else:
        raise TypeError(f"Expected str, int or list/tuple, bug got {gpus!r}.")


def _gpus_int2list(gpus: Union[int, Sequence]) -> Optional[List[int]]:
    """
    Normalize int gpus to 0-base list and handle -1 case.
    """
    assert gpus is not None
    if isinstance(gpus, collections.abc.Sequence):
        return [gpu for gpu in gpus]

    # must be an int
    if not gpus:  # gpus==0
        return None
    if gpus == -1:
        return list(range(num_gpus()))

    return list(range(gpus))


def _check_data_type(device_ids: Union[str, int, Sequence]) -> None:
    """Checks that the device_ids argument: int, string, or sequence of integers.

    Args:
        device_ids:
            Device location.
    """
    msg = "Device IDs must be an int, a string, a sequence of ints, but got"
    if device_ids is None:
        return
    if isinstance(device_ids, (collections.abc.MutableSequence, tuple)):
        for id_ in device_ids:
            if type(id_) is not int:
                raise TypeError(f"{msg} a sequence of {type(id_).__name__}.")
    elif type(device_ids) not in (int, str):
        raise TypeError(f"{msg} {device_ids!r}.")


def synchronize():
    """Similar to cuda.synchronize().
    Waits for all kernels in all streams on a CUDA device to complete.
    """
    if torch.cuda.is_available():
        torch.cuda.synchronize()


def peak_cuda_memory() -> Tuple[float]:
    """
    Return the peak gpu memory statistics.

    Returns:
        max_alloc (float): the allocated CUDA memory
        max_cached (float): the cached CUDA memory
    """

    max_alloc = torch.cuda.max_memory_allocated() / (1024**3)
    max_cached = torch.cuda.max_memory_reserved() / (1024**3)
    torch.cuda.reset_peak_memory_stats()
    torch.cuda.empty_cache()

    return max_alloc, max_cached


# https://github.com/Lightning-AI/lightning/blob/master/src/lightning/pytorch/utilities/memory.py
def release_memory(*objects):
    """
    Triggers garbage collection and Releases cuda cache memory.

    This function sets the inputs to `None`, triggers garbage collection to release
    CPU memory references, and attempts to clear GPU memory cache.

    Args:
        *objects: Variable number of objects to release.

    Returns:
        List[None]: A list of `None` values, with the same length as the input objects.

    Example:
        >>> import torch
        >>> a = torch.ones(1024, 1024).cuda()
        >>> b = torch.ones(1024, 1024).cuda()
        >>> a, b = release_memory(a, b)
        ```
    """
    if not isinstance(objects, list):
        objects = list(objects)
    for i in range(len(objects)):
        if objects[i] is not None:
            objects[i] = None
    gc.collect()
    if is_cuda_available():
        try:
            torch.cuda.empty_cache()
        except RuntimeError as ext:
            if not is_oom_error(ext):
                # Only handle OOM errors
                raise

    return objects


class AutoGPUMode(str, enum.Enum):
    MAX_MEM = "MAX_MEM"
    MAX_MEM_RATE = "MAX_MEM_RATE"
    MIN_POWER = "MIN_POWER"


class GPUManager:
    """This class enables the automated selection of the most available GPU or another based on specified mode.

    Args:
        addtional_qargs (Optional): Additional arguments passed to ``nvidia-smi``.
        mode (AutoGPUMode): mode for GPU selection.
            Defaults to MAX_MEM (max-free) memory.

    Example::

        import os,torch
        os.environ["CUDA_VISIBLE_DEVICES"]="1,2"
        gm = GPUManager()
        torch_device = gm.auto_choice()

        or

        torch_device = GPUManager.detect()


        a = torch.randn(1,1000)
        a.to(torch_device)
    """

    def __init__(self, addtional_qargs=None, mode: AutoGPUMode = AutoGPUMode.MAX_MEM):
        self._check_nvidia_smi()
        self._check_cuda_devices()

        addtional_qargs = self._normalize_qargs(addtional_qargs)
        self.qargs = [
            "index",
            "gpu_name",
            "memory.free",
            "memory.total",
            "power.draw",
            "power.limit",
        ] + list(addtional_qargs)
        self.numberic_args = [
            "memory.free",
            "memory.total",
            "power.draw",
            "power.limit",
        ]
        self.mode = mode
        self.gpu_id2device = {
            device2gpu_id(device_id): device_id for device_id in list(range(num_gpus()))
        }
        self.gpu_num = len(self.gpu_id2device)
        self.gpu_stats = [{"specified": False} for _ in range(self.gpu_num)]

    def _check_nvidia_smi(self):
        nvidia_smi_path = shutil.which("nvidia-smi")
        if nvidia_smi_path is None:
            raise FileNotFoundError("nvidia-smi: command not found")

    def _check_cuda_devices(self):
        if not is_cuda_available():
            raise RuntimeError(
                "GPUManager could only be used to manage NVIDIA GPUs, but no visible GPU found."
            )

    @classmethod
    def _normalize_qargs(cls, qargs: Union[Sequence[str], str, None]) -> List[str]:
        """normalize additional query args to list."""
        original_qargs = qargs
        if isinstance(qargs, str):
            qargs = [qargs]
        qargs = qargs or []
        if not isinstance(qargs, Sequence) or not all(
            isinstance(qarg, str) for qarg in qargs
        ):
            raise ValueError(
                f"Additional qargs for nvidia-smi should be a string or a sequence of strings, "
                f"but received ({original_qargs!r})."
            )
        return list(qargs)

    def new_query(self):
        """
        Running the ``nvidia-smi`` command and organizing the results as a list of dictionaries
        containing information searched from `nvidia-smi`.
        """
        results = subprocess.run(
            [
                "nvidia-smi",
                f"--query-gpu={','.join(self.qargs)}",
                "--format=csv,noheader",
            ],
            encoding="utf-8",
            capture_output=True,
            check=True,
        )
        results = results.stdout.strip().splitlines()
        gpu_stats = [
            {k: v for k, v in zip(self.qargs, result_line.strip().split(","))}
            for result_line in results
        ]

        return gpu_stats

    def _update_stats(self):
        """Update the GPU stats."""
        for past_stat, new_stat in zip(self.gpu_stats, self.new_query()):
            past_stat.update(new_stat)

    @classmethod
    def detect(cls, mode: Optional[AutoGPUMode] = AutoGPUMode.MAX_MEM) -> int:
        """A classmethod calls :meth`auto_choice` method to select a GPU and returns dveice id."""
        gm = cls(mode=mode)
        return gm.auto_choice()

    def auto_choice(self, mode: Optional[str] = None) -> int:
        """Auto choice a GPU ID based on specified mode.

        Args:
            mode (str): The mode for selecting the GPU.

        Returns:
            device id.
        """
        mode = mode.upper() if mode is not None else self.mode
        self._update_stats()

        # Give a chance to choose everyone.
        unspecified_gpu_stats = [
            stat for stat in self.gpu_stats if not stat["specified"]
        ] or self.gpu_stats

        if mode == AutoGPUMode.MAX_MEM:
            chosen_gpu = self._sort_by_memory(unspecified_gpu_stats, by_size=True)[0]
        elif mode == AutoGPUMode.MAX_MEM_RATE:
            chosen_gpu = self._sort_by_memory(unspecified_gpu_stats, by_size=False)[0]
        elif mode == AutoGPUMode.MIN_POWER:
            chosen_gpu = self._sort_by_power(unspecified_gpu_stats)[0]
        else:
            warnings.warn(
                "Invalid mode, defaulting to auto select gpu with max free memory."
            )
            chosen_gpu = self._sort_by_memory(unspecified_gpu_stats)[0]
        chosen_gpu["specified"] = True
        device = self.gpu_id2device[chosen_gpu["index"]]
        logger.info(
            "Auto select GPU: device={device}\n### {info}".format(
                device=device,
                info="\n### ".join(
                    [
                        str(k) + ":" + str(v)
                        for k, v in chosen_gpu.items()
                        if k != "specified"
                    ]
                ),
            ),
            ranks=0,
        )
        return device

    def _sort_by_memory(self, gpu_stats: List[Dict[str, str]], by_size=True):
        """sorted memory descending."""
        return sorted(gpu_stats, key=partial(self._get_memory, by_size=by_size))

    @classmethod
    def _get_memory(cls, gpu_stat: Dict[str, str], by_size: bool = True):
        memorys = (cls._to_float(gpu_stat["power.draw"], "W"),)
        if by_size:
            return memorys[0]
        memorys = memorys + (cls._to_float(gpu_stat["power.limit"], "W"),)

        return float(memorys[0] / memorys[1])

    def _sort_by_power(self, gpu_stats: List[Dict[str, str]]):
        return sorted(gpu_stats, key=self._get_power)

    @classmethod
    def _get_power(cls, gpu_stat: Dict[str, str]):
        powers = (
            cls._to_float(gpu_stat["power.draw"], "W"),
            cls._to_float(gpu_stat["power.limit"], "W"),
        )
        if any(power == 1 for power in powers):
            warnings.warn(f"Power management unable for GPU: {gpu_stat['index']}.")
            return 1
        else:
            return float(powers[0] / powers[1])

    @classmethod
    def _to_float(cls, info: str, unit: str) -> float:
        try:
            return float(info.upper().replace(unit.upper(), "").strip())
        except ValueError:  # 'Not Supported' case
            return 1.0


def device2gpu_id(device_id: int) -> str:
    """Given the device index and get the unmasked real GPU ID."""
    # All devices if `CUDA_VISIBLE_DEVICES` unset
    default = ",".join(str(i) for i in range(num_gpus()))
    cuda_visible_devices = os.getenv("CUDA_VISIBLE_DEVICES", default=default).split(",")
    return cuda_visible_devices[device_id].strip()


@lru_cache(1)
def num_gpus() -> int:
    """Get visible gpu number."""
    with patch_nvml_check_env():
        return (
            torch.cuda.device_count()
            if _TORCH_GREATER_EQUAL_2_0
            else NVMLDeviceCount.device_count()
        )


def is_cuda_available():
    """
    check cuda and leave cuda uninitialized.
    """
    with patch_nvml_check_env():
        return (
            torch.cuda.is_available()
            if _TORCH_GREATER_EQUAL_2_0
            else NVMLDeviceCount.device_count() > 0
        )


@contextmanager
def patch_nvml_check_env():
    """A context manager that patch ``PYTORCH_NVML_BASED_CUDA_CHECK=1``, and restore finally."""
    has_nvml_env = "PYTORCH_NVML_BASED_CUDA_CHECK" in os.environ
    cache_nvml_env = os.environ.get("PYTORCH_NVML_BASED_CUDA_CHECK")
    os.environ["PYTORCH_NVML_BASED_CUDA_CHECK"] = str(1)
    yield
    if has_nvml_env:
        os.environ["PYTORCH_NVML_BASED_CUDA_CHECK"] = cache_nvml_env
    else:
        del os.environ["PYTORCH_NVML_BASED_CUDA_CHECK"]


class NVMLDeviceCount(object):
    """A tool for nvml-based cuda check for torch < 2.0 which won't trigger the drivers and leave cuda
    uninitialized.

    Coppied from pytorch, see:
        https://github.com/pytorch/pytorch/pull/84879
    """

    @staticmethod
    def device_count():
        nvml_count = NVMLDeviceCount._device_count_nvml()
        return torch.cuda.device_count() if nvml_count < 0 else nvml_count

    @staticmethod
    def _device_count_nvml() -> int:
        """Return number of devices as reported by NVML taking CUDA_VISIBLE_DEVICES into account.

        Negative value is returned if NVML discovery or initialization has failed.
        """
        visible_devices = NVMLDeviceCount._parse_visible_devices()
        if not visible_devices:
            return 0
        try:
            if type(visible_devices[0]) is str:
                # Skip MIG parsing
                if visible_devices[0].startswith("MIG-"):
                    return -1
                uuids = NVMLDeviceCount._raw_device_uuid_nvml()
                if uuids is None:
                    return -1
                visible_devices = NVMLDeviceCount._transform_uuid_to_ordinals(
                    cast(List[str], visible_devices), uuids
                )
            else:
                raw_cnt = NVMLDeviceCount._raw_device_count_nvml()
                if raw_cnt <= 0:
                    return raw_cnt
                # Trim the list up to a maximum available device
                for idx, val in enumerate(visible_devices):
                    if cast(int, val) >= raw_cnt:
                        return idx
        except (OSError, AttributeError):
            return -1
        return len(visible_devices)

    @staticmethod
    def _parse_visible_devices() -> Union[List[int], List[str]]:
        """Parse CUDA_VISIBLE_DEVICES environment variable."""
        var = os.getenv("CUDA_VISIBLE_DEVICES")
        if var is None:
            return list(range(64))

        def _strtoul(s: str) -> int:
            """Return -1 or positive integer sequence string starts with,"""
            if not s:
                return -1
            for idx, c in enumerate(s):
                if not (c.isdigit() or (idx == 0 and c in "+-")):
                    break
                if idx + 1 == len(s):
                    idx += 1
            return int(s[:idx]) if idx > 0 else -1

        def parse_list_with_prefix(lst: str, prefix: str) -> List[str]:
            rcs: List[str] = []
            for elem in lst.split(","):
                # Repeated id results in empty set
                if elem in rcs:
                    return cast(List[str], [])
                # Anything other but prefix is ignored
                if not elem.startswith(prefix):
                    break
                rcs.append(elem)
            return rcs

        if var.startswith("GPU-"):
            return parse_list_with_prefix(var, "GPU-")
        if var.startswith("MIG-"):
            return parse_list_with_prefix(var, "MIG-")
        # CUDA_VISIBLE_DEVICES uses something like strtoul
        # which makes `1gpu2,2ampere` is equivalent to `1,2`
        rc: List[int] = []
        for elem in var.split(","):
            x = _strtoul(elem.strip())
            # Repeated ordinal results in empty set
            if x in rc:
                return cast(List[int], [])
            # Negative value aborts the sequence
            if x < 0:
                break
            rc.append(x)
        return rc

    @staticmethod
    def _raw_device_uuid_nvml() -> Optional[List[str]]:
        """Return list of device UUID as reported by NVML
        or None if NVM discovery/initialization failed."""
        from ctypes import CDLL, byref, c_int, c_void_p, create_string_buffer

        nvml_h = CDLL("libnvidia-ml.so.1")
        rc = nvml_h.nvmlInit()
        if rc != 0:
            warnings.warn("Can't initialize NVML")
            return None
        dev_count = c_int(-1)
        rc = nvml_h.nvmlDeviceGetCount_v2(byref(dev_count))
        if rc != 0:
            warnings.warn("Can't get nvml device count")
            return None
        uuids: List[str] = []
        for idx in range(dev_count.value):
            dev_id = c_void_p()
            rc = nvml_h.nvmlDeviceGetHandleByIndex_v2(idx, byref(dev_id))
            if rc != 0:
                warnings.warn("Can't get device handle")
                return None
            buf_len = 96
            buf = create_string_buffer(buf_len)
            rc = nvml_h.nvmlDeviceGetUUID(dev_id, buf, buf_len)
            if rc != 0:
                warnings.warn("Can't get device UUID")
                return None
            uuids.append(buf.raw.decode("ascii").strip("\0"))
        del nvml_h
        return uuids

    @staticmethod
    def _transform_uuid_to_ordinals(
        candidates: List[str], uuids: List[str]
    ) -> List[int]:
        """Given the set of partial uuids and list of known uuids builds
        a set of ordinals excluding ambiguous partials IDs"""

        def uuid_to_orinal(candidate: str, uuids: List[str]) -> int:
            best_match = -1
            for idx, uuid in enumerate(uuids):
                if not uuid.startswith(candidate):
                    continue
                # Ambiguous candidate
                if best_match != -1:
                    return -1
                best_match = idx
            return best_match

        rc: List[int] = []
        for candidate in candidates:
            idx = uuid_to_orinal(candidate, uuids)
            # First invalid ordinal stops parsing
            if idx < 0:
                break
            # Duplicates result in empty set
            if idx in rc:
                return cast(List[int], [])
            rc.append(idx)
        return rc

    @staticmethod
    def _raw_device_count_nvml() -> int:
        """Return number of devices as reported by NVML
        or negative value if NVML discovery/initialization failed."""
        from ctypes import CDLL, byref, c_int

        nvml_h = CDLL("libnvidia-ml.so.1")
        rc = nvml_h.nvmlInit()
        if rc != 0:
            warnings.warn("Can't initialize NVML")
            return -1
        dev_count = c_int(-1)
        rc = nvml_h.nvmlDeviceGetCount_v2(byref(dev_count))
        if rc != 0:
            warnings.warn("Can't get nvml device count")
            return -1
        del nvml_h
        return dev_count.value


def is_oom_error(exception: BaseException) -> bool:
    return (
        is_cuda_out_of_memory(exception)
        or is_cudnn_snafu(exception)
        or is_out_of_cpu_memory(exception)
    )


# based on https://github.com/BlackHC/toma/blob/master/toma/torch_cuda_memory.py
def is_cuda_out_of_memory(exception: BaseException) -> bool:
    return (
        isinstance(exception, RuntimeError)
        and len(exception.args) == 1
        and "CUDA" in exception.args[0]
        and "out of memory" in exception.args[0]
    )


# based on https://github.com/BlackHC/toma/blob/master/toma/torch_cuda_memory.py
def is_cudnn_snafu(exception: BaseException) -> bool:
    # For/because of https://github.com/pytorch/pytorch/issues/4107
    return (
        isinstance(exception, RuntimeError)
        and len(exception.args) == 1
        and "cuDNN error: CUDNN_STATUS_NOT_SUPPORTED." in exception.args[0]
    )


# based on https://github.com/BlackHC/toma/blob/master/toma/cpu_memory.py
def is_out_of_cpu_memory(exception: BaseException) -> bool:
    return (
        isinstance(exception, RuntimeError)
        and len(exception.args) == 1
        and "DefaultCPUAllocator: can't allocate memory" in exception.args[0]
    )
