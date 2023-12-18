# -*- coding:utf-8 -*-
# Copyright xmuspeech (Author: Leo 2023-06)

import functools
import importlib
import operator
import types
from typing import Callable, Optional

import packaging
import pkg_resources


def is_package_available(*modules: str) -> bool:
    """
    Returns if a top-level module with :attr:`name` exists *without**
    importing it. This is generally safer than try-catch block around a
    `import X`. It avoids third party libraries breaking assumptions of some of
    our tests, e.g., setting multiprocessing start method when imported
    (see librosa/#747, torchvision/#544).
    """
    return all(importlib.util.find_spec(m) is not None for m in modules)


@functools.lru_cache
def is_module_available(module_path: str) -> bool:
    """Check if a module path is available in your environment.
    This will try to import it.

    >>> is_module_available('torch')
    True
    >>> is_module_available('fake')
    False
    >>> is_module_available('torch.utils')
    True
    >>> is_module_available('torch.util')
    False
    """
    module_names = module_path.split(".")
    if not is_package_available(module_names[0]):
        return False
    try:
        importlib.import_module(module_path)
    except ImportError:
        return False
    return True


def compare_version(
    package: str, op: Callable, ver: str, use_base_version: bool = False
) -> bool:
    """Compare package version with some requirements.

    >>> compare_version("torch", operator.ge, "0.1")
    True
    >>> compare_version("does_not_exist", operator.ge, "0.0")
    False
    """
    try:
        pkg = importlib.import_module(package)
    except (ImportError, pkg_resources.DistributionNotFound):
        return False
    try:
        if hasattr(pkg, "__version__"):
            pkg_version = packaging.version.parse(pkg.__version__)
        else:
            # try pkg_resources to infer version
            pkg_version = packaging.version.parse(
                pkg_resources.get_distribution(package).version
            )
    except TypeError:
        # this is mocked by Sphinx, so it should return True to generate all summaries
        return True
    if use_base_version:
        pkg_version = packaging.version.parse(pkg_version.base_version)
    return op(pkg_version, packaging.version.parse(ver))


@functools.lru_cache()
class RequirementCache:
    """Boolean-like class to check for requirement and module availability.

    Avoid overhead import, copied from:
        https://github.com/Lightning-AI/utilities/blob/main/src/lightning_utilities/core/imports.py#lazy_import

    Args:
        requirement: The requirement to check, version specifiers are allowed.
        module: The optional module to try to import if the requirement check fails.

    >>> RequirementCache("torch>=0.1")
    Requirement 'torch>=0.1' met
    >>> bool(RequirementCache("torch>=0.1"))
    True
    >>> bool(RequirementCache("torch>100.0"))
    False
    >>> RequirementCache("torch")
    Requirement 'torch' met
    >>> bool(RequirementCache("torch"))
    True
    >>> bool(RequirementCache("unknown_package"))
    False
    >>> bool(RequirementCache(module="torch.utils"))
    True
    >>> bool(RequirementCache(module="unknown_package"))
    False
    >>> bool(RequirementCache(module="unknown.module.path"))
    False

    """

    def __init__(
        self, requirement: Optional[str] = None, module: Optional[str] = None
    ) -> None:
        if not (requirement or module):
            raise ValueError("At least one arguments need to be set.")
        self.requirement = requirement
        self.module = module

    def _check_requirement(self) -> None:
        assert self.requirement  # noqa: S101; needed for typing
        try:
            # first try the pkg_resources requirement
            pkg_resources.require(self.requirement)
            self.available = True
            self.message = f"Requirement {self.requirement!r} met"
        except Exception as ex:
            self.available = False
            self.message = f"{ex.__class__.__name__}: {ex}. HINT: Try running `pip install -U {self.requirement!r}`"
            req_include_version = any(c in self.requirement for c in "=<>")
            if not req_include_version or self.module is not None:
                module = self.requirement if self.module is None else self.module
                # sometimes `pkg_resources.require()` fails but the module is importable
                self.available = is_module_available(module)
                if self.available:
                    self.message = f"Module {module!r} available"

    def _check_module(self) -> None:
        assert self.module  # noqa: S101; needed for typing
        self.available = is_module_available(self.module)
        if self.available:
            self.message = f"Module {self.module!r} available"
        else:
            self.message = f"Module not found: {self.module!r}. HINT: Try running `pip install -U {self.module}`"

    def _check_available(self) -> None:
        if hasattr(self, "available"):
            return
        if self.requirement:
            self._check_requirement()
        if getattr(self, "available", True) and self.module:
            self._check_module()

    def __bool__(self) -> bool:
        """Format as bool."""
        self._check_available()
        return self.available

    def __str__(self) -> str:
        """Format as string."""
        self._check_available()
        return self.message

    def __repr__(self) -> str:
        """Format as string."""
        return self.__str__()


_PL_AVAILABLE = is_package_available("lightning")
_TRANSFORMERS_AVAILABLE = is_package_available("transformers")
_H5PY_AVAILABLE = is_package_available("h5py")
_KALDI_NATIVE_IO_AVAILABLE = is_package_available("kaldi_native_io")

_TORCH_GREATER_EQUAL_2_0 = RequirementCache("torch>=2.0")
_TORCH_GREATER_EQUAL_1_9 = RequirementCache("torch>=1.9")


@functools.lru_cache
def torch_dist_is_available():
    import torch

    return torch.distributed.is_available()


@functools.lru_cache(1)
def torchaudio_ge_2_1():
    return is_module_available("torchaudio") and compare_version(
        "torchaudio", operator.ge, "2.1"
    )


def lazy_import(module_name, callback=None):
    """Returns a proxy module object that will lazily import the given module the first time it is used.

    Copied from:
        https://github.com/Lightning-AI/utilities/blob/main/src/lightning_utilities/core/imports.py#lazy_import

    Example usage::

        # Lazy version of `import tensorflow as tf`
        tf = lazy_import("tensorflow")

        # Other commands

        # Now the module is loaded
        tf.__version__

    Args:
        module_name: the fully-qualified module name to import
        callback (None): a callback function to call before importing the
            module

    Returns:
        a proxy module object that will be lazily imported when first used

    """
    return LazyModule(module_name, callback=callback)


class LazyModule(types.ModuleType):
    """Proxy module that lazily imports the underlying module the first time it is actually used.

    Args:
        module_name: the fully-qualified module name to import
        callback (None): a callback function to call before importing the
            module

    """

    def __init__(self, module_name, callback=None):
        super().__init__(module_name)
        self._module = None
        self._callback = callback

    def __getattr__(self, item):
        if self._module is None:
            self._import_module()

        return getattr(self._module, item)

    def __dir__(self):
        if self._module is None:
            self._import_module()

        return dir(self._module)

    def _import_module(self):
        # Execute callback, if any
        if self._callback is not None:
            self._callback()

        # Actually import the module
        module = importlib.import_module(self.__name__)
        self._module = module

        # Update this object's dict so that attribute references are efficient
        # (__getattr__ is only called on lookups that fail)
        self.__dict__.update(module.__dict__)
