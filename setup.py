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


def get_version() -> str:
    """
    This function reads the pyproject.toml and generates the egrecho/version.py file.

    Returns:
        The library version stored in pyproject.toml.
    """
    project_path = ROOT_DIR
    toml_path = os.path.join(project_path, "VERSION")
    version_py_path = os.path.join(project_path, "egrecho/version.py")

    with open(toml_path) as f:
        version = f.read().strip()
    # write version into version.py
    with open(version_py_path, "w") as f:
        f.write(f"__version__ = '{version}'\n")
    return version


if __name__ == "__main__":

    with open("README.md", encoding="utf8") as f:
        readme = f.read().split("--------------------")[-1]

    version = get_version()
    reqs = load_requirements(PATH_REQUIRE)
    setup(
        name='egrecho',
        version=version,
        packages=find_packages(include=("egrecho*",))
        + find_namespace_packages(include=("egrecho_cli*",)),
        entry_points={
            "console_scripts": [
                "egrecho = egrecho_cli.__main__:main",
            ]
        },
        install_requires=reqs,
        extras_require=_prepare_extras(),
        zip_safe=False,
        include_package_data=True,
        long_description=readme,
        long_description_content_type="text/markdown",
        author="Dexin Liao",
        url='https://github.com/wangers/subtools2',
        license="Apache-2.0 License",
        classifiers=[
            "Programming Language :: Python :: 3",
            "License :: OSI Approved :: Apache Software License",
            "",
        ],
    )
