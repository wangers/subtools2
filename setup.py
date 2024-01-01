# -*- coding:utf-8 -*-

# Copyright xmuspeech (Author: Leo 2023-03-11)

import functools
import glob
import itertools
import os
import re
import sys

import pkg_resources
from setuptools import find_namespace_packages, find_packages, setup

if sys.version_info < (3, 8):
    print("Python > 3.8 is required.")
    sys.exit(-1)

MIN_PYTORCH_VERSION_MAJOR = 1
MIN_PYTORCH_VERSION_MINOR = 13

ROOT_DIR = os.path.realpath(os.path.dirname(__file__))
PATH_REQUIRE = os.path.join(ROOT_DIR, "requirements")
BASE_REQUIRE = os.path.join(PATH_REQUIRE, "base.txt")


def load_requirements(path_dir: str, file_name: str = "base.txt"):
    """Load requirements from a file.

    >>> load_requirements(PATH_REQUIRE)
    ['rich...', 'lightning..."]

    """
    path_f = os.path.join(path_dir, file_name)
    if not os.path.exists(path_f):
        raise ValueError(
            f"Path {path_f} not found for input dir {path_dir} and filename {file_name}."
        )
    with open(path_f, "r") as f:
        req = []
        for line in f.readlines():
            try:
                (req_,) = pkg_resources.parse_requirements(line)
                req.append(str(req_))
            except ValueError:
                pass
    return req


def get_pytorch_version():
    """
    This functions finds the PyTorch version.

    Returns:
        A tuple of integers in the form of (major, minor, patch).
    """
    import torch

    torch_version = torch.__version__.split("+")[0]
    TORCH_MAJOR = int(torch_version.split(".")[0])
    TORCH_MINOR = int(torch_version.split(".")[1])
    TORCH_PATCH = int(torch_version.split(".")[2], 16)
    return TORCH_MAJOR, TORCH_MINOR, TORCH_PATCH


def check_pytorch_version(min_major_version, min_minor_version) -> bool:
    # get pytorch version
    torch_major, torch_minor, _ = get_pytorch_version()

    # if the
    if torch_major < min_major_version or (
        torch_major == min_major_version and torch_minor < min_minor_version
    ):
        raise RuntimeError(
            f"Requires Pytorch {min_major_version}.{min_minor_version} or newer.\n"
            "refer https://pytorch.org/ to install torch."
        )


def _prepare_extras(skip_pattern: str = "^_", skip_files=("base.txt",)):
    """Preparing extras for the package listing requirements.

    Args:
        skip_pattern: ignore files with this pattern, by default all files starting with _
        skip_files: ignore some additional files, by default base requirements

    Note, particular domain test requirement are aggregated in single "_tests" extra (which is not accessible).

    """
    # find all extra requirements
    _load_req = functools.partial(load_requirements, path_dir=PATH_REQUIRE)
    found_req_files = sorted(
        os.path.basename(p) for p in glob.glob(os.path.join(PATH_REQUIRE, "*.txt"))
    )

    # filter unwanted files
    found_req_files = [n for n in found_req_files if not re.match(skip_pattern, n)]
    found_req_files = [n for n in found_req_files if n not in skip_files]
    found_req_names = [os.path.splitext(req)[0] for req in found_req_files]

    # define basic and extra extras
    extras_req = {"_tests": []}
    for name, fname in zip(found_req_names, found_req_files):
        if name.endswith("_test"):
            extras_req["_tests"] += _load_req(file_name=fname)
        else:
            extras_req[name] = _load_req(file_name=fname)

    # filter the uniques
    extras_req = {n: list(set(req)) for n, req in extras_req.items()}

    # create an 'all' keyword that install all possible dependencies
    extras_req["all"] = list(
        itertools.chain(
            [pkgs for k, pkgs in extras_req.items() if k not in ("_test", "_tests")]
        )
    )
    extras_req["test"] = extras_req["all"] + extras_req["_tests"]

    return extras_req


with open("README.md", encoding="utf8") as f:
    readme = f.read().split("--------------------")[-1]


def get_version() -> str:
    """
    This function reads the VERSION and generates the egrecho/version.py file.

    Returns:
        The library version stored in VERSION.
    """
    project_path = ROOT_DIR
    version_txt_path = os.path.join(project_path, "VERSION")
    version_py_path = os.path.join(project_path, "egrecho/version.py")

    with open(version_txt_path) as f:
        version = f.read().strip()

    # write version into version.py
    with open(version_py_path, "w") as f:
        f.write(f"__version__ = '{version}'\n")
    return version


if __name__ == "__main__":
    try:
        check_pytorch_version(MIN_PYTORCH_VERSION_MAJOR, MIN_PYTORCH_VERSION_MINOR)
        # If the user already installed PyTorch, make sure he has torchaudio too.
        import torchaudio
    except ImportError as e:
        raise ImportError(
            f"{e}\n#### Please refer https://pytorch.org/ to install torch and torchaudio."
        )

    version = get_version()
    reqs = load_requirements(PATH_REQUIRE)
    setup(
        version=version,
        packages=find_packages(include=("egrecho*",))
        + find_namespace_packages(include=("egrecho_cli*",)),
        install_requires=reqs,
        extras_require=_prepare_extras(),
        long_description=readme,
        long_description_content_type="text/markdown",
    )
