# -*- coding:utf-8 -*-

# Copyright xmuspeech (Author: Leo 2023-03-11)

import sys

from setuptools import find_namespace_packages, find_packages, setup

VERSION = 0.1

if sys.version_info < (3, 8):
    print("Python > 3.8 is required.")
    sys.exit(-1)

with open("requirements.txt") as f:
    reqs = []
    for line in f:
        line = line.strip()
        reqs.append(line)

try:
    # If the user already installed PyTorch, make sure he has torchaudio too.
    import torch

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
