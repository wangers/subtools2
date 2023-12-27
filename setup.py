# -*- coding:utf-8 -*-

# Copyright xmuspeech (Author: Leo 2023-03-11)

import sys

from setuptools import find_namespace_packages, find_packages, setup

VERSION = 0.1
MIN_PYTORCH_VERSION_MAJOR = 1
MIN_PYTORCH_VERSION_MINOR = 13


if sys.version_info < (3, 8):
    print("Python > 3.8 is required.")
    sys.exit(-1)


def check_pytorch_version(min_major_version, min_minor_version) -> bool:
    def get_pytorch_version():
        """
        This functions finds the PyTorch version.

        Returns:
            A tuple of integers in the form of (major, minor, patch).
        """
        torch_version = torch.__version__.split("+")[0]
        TORCH_MAJOR = int(torch_version.split(".")[0])
        TORCH_MINOR = int(torch_version.split(".")[1])
        TORCH_PATCH = int(torch_version.split(".")[2], 16)
        return TORCH_MAJOR, TORCH_MINOR, TORCH_PATCH

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


with open("requirements.txt") as f:
    reqs = []
    for line in f:
        line = line.strip()
        reqs.append(line)

try:
    # If the user already installed PyTorch, make sure he has torchaudio too.
    import torch

    check_pytorch_version(MIN_PYTORCH_VERSION_MAJOR, MIN_PYTORCH_VERSION_MINOR)
    try:
        import torchaudio

    except ImportError as e:
        raise ValueError("need torchdata")
except ImportError as e:
    raise ImportError(
        f"{e}\n#### Please refer https://pytorch.org/ to install torch and torchaudio."
    )

package_name = "egrecho"

setup(
    name=package_name,
    version=VERSION,
    author="Dexin Liao",
    packages=find_packages(include=("egrecho*",))
    + find_namespace_packages(include=("egrecho_cli*",)),
    entry_points={
        "console_scripts": [
            "egrecho=egrecho_cli.__main__:main",
        ]
    },
    license="Apache-2.0 License",
    install_requires=reqs,
    python_requires=">=3.8",
)
