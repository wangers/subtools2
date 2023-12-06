# -*- coding:utf-8 -*-
# Copyright xmuspeech (Author: Leo 2023-10)

from pathlib import Path
from typing import Literal, Optional

from egrecho.utils import constants
from egrecho.utils.io.files import is_remote_url, resolve_file
from egrecho.utils.io.utils import SerializationFn, repr_dict


def resolve_ckpt(
    checkpoint: str = "last.ckpt",
    dirpath: Optional[str] = None,
    version: str = "version",
    best_k_fname: str = constants.BEST_K_MAP_FNAME,
    best_k_mode: Literal["max", "min"] = "min",
    ckpt_subdir: str = constants.CHECKPOINT_DIR_NAME,
) -> str:
    """
    Resolve checkpoint path from local or remote.

    Automatically search checkpoint, parameters except `checkpoint` is for local fs,
    checkpoint can be either:

    - remote url (e.g., startwith "http"): return it directly, otherwise change to local mode.
    - absolute file path: return it if exists, otherwise raise a FileExistError.
    - relative file name: rel to `dirpath`, return it if exist, otherwise change mode to auto-matching.
    - auto-matching: `checkpoint` must a parameter of one level rel path (recommand one file name)
    to avoid messing:

        - `dirpath`: base dir.
        - `ckpt_subdir`: defaults to name as "checkpoints"
        - `version`: version subdir, if specified will check it otherwise find the max version number (means latest training).
        - See :function::`resolve_version_ckpt` and :function::`resolve_version_ckpt` for details.

        The minist matching unit is like::

        ./dirpath/
        ├── best_k_models.yaml
        ├── last.ckpt
        └── abc.ckpt

        or

        ./dirpath/
        ├── hparams.yaml
        └── checkpoints
            ├── best_k_models.yaml
            ├── last.ckpt
            └── abc.ckpt

    With version subdir structure can be::

        ./dirpath/version_1
                ├── hparams.yaml
                └── checkpoints
                    ├── best_k_models.yaml
                    ├── last.ckpt
                    └── abc.ckpt

    Args:
        checkpoint (str, optional):
            The file name of checkpoint to resolve, local file needs a suffix like ".ckpt" / ".pt",
            While checkpoint="best" is a preseved key means it will find `best_k_fname` which is
            a file contains `Dict[BEST_K_MODEL_PATH, BEST_K_SCORE]`, and sort by its score to
            match a best ckpt. Defaults to "last.ckpt".
        dirpath (Path or str, optional):
            The root path. Defaults to None, which means the current directory.
        version (str, optional):
            The versioned subdir name. Conmmonly subdir is named as "version_0/version_1", if you specify
            the version name with a version num, it will search that version dir, otherwise choose the max number
            of version (above "version_1"). Defaults to "version".
        best_k_fname (str, optional):
            The filename for the best_k map file. Note that the best model path in best map file may
            not in this directory since it is stored in training stage, so we assume that its basename
            can matching ckpts in the same level. Defaults to best_k_models.yaml.
        best_k_mode (Literal["max", "min"], optional):
            The mode for selecting the best_k checkpoint. Defaults to "min".
        ckpt_subdir (str, optional):
            The name of the checkpoints subdir. Defaults to "checkpoints".
    """
    if is_remote_url(checkpoint):
        return checkpoint
    ckpt = Path(resolve_file(checkpoint, dirpath))
    if ckpt.is_file():
        return str(ckpt.resolve())
    base_dir = Path(dirpath) if dirpath is not None else Path().resolve()
    assert base_dir.is_dir()
    if ckpt := resolve_version_ckpt(
        dirpath=str(base_dir),
        checkpoint=checkpoint,
        version=version,
        best_k_fname=best_k_fname,
        best_k_mode=best_k_mode,
        ckpt_subdir=ckpt_subdir,
    ):
        return ckpt
    else:
        param_msg = repr_dict(
            dict(
                checkpoint=checkpoint,
                dirpath=dirpath,
                version=version,
                best_k_fname=best_k_fname,
                best_k_mode=best_k_mode,
                ckpt_subdir=ckpt_subdir,
            )
        )
        raise ValueError(
            f"Failed to resolve a valid checkpoint file from:\n{param_msg}"
            "#### Fix your parameter or use a absolute path for `checkpoint`."
        )


def resolve_version_ckpt(
    dirpath: Optional[str] = None,
    checkpoint: str = "last.ckpt",
    version: str = "version",
    best_k_fname: str = constants.BEST_K_MAP_FNAME,
    best_k_mode: Literal["max", "min"] = "min",
    ckpt_subdir: str = constants.CHECKPOINT_DIR_NAME,
) -> Optional[str]:
    """Search for a local version directory.

    Cares about structure like::

        ./dirpath/version_1
                ├── hparams.yaml
                └── checkpoints
                    ├── best_k_models.yaml
                    ├── last.ckpt
                    └── abc.ckpt

    Note: Truly matching see :function::``resolve_rel_ckpt`` for more details.

    Args:
        version (str, optional):
            The versioned subdir name. Conmmonly subdir is named as "version_0/version_1", if you specify
            the version name with a version num, it will search that version dir, otherwise choose the max number
            of version (above "version_1"). Defaults to "version".
        dirpath (Path or str, optional):
            The root path. Defaults to None, which means the current directory.
        checkpoint (str, optional):
            The file name of checkpoint to resolve, needs a suffix like ".ckpt" / ".pt",
            While checkpoint="best" is a preseved key means it will find `best_k_fname` which is
            a file contains `Dict[BEST_K_MODEL_PATH, BEST_K_SCORE]`, and sort by its score to
            match a best ckpt. Defaults to "last.ckpt".
        best_k_fname (str, optional):
            The filename for the best_k map file. Note that the best model path in best map file may
            not in this directory since it is stored in training stage, so we assume that its basename
            can matching ckpts in the same level. Defaults to best_k_models.yaml.
        best_k_mode (Literal["max", "min"], optional):
            The mode for selecting the best_k checkpoint. Defaults to "min".
        ckpt_subdir (str, optional):
            The name of the checkpoints subdir. Defaults to "checkpoints".
    """
    # Check this dir
    if ckpt := resolve_rel_ckpt(
        dirpath=dirpath,
        checkpoint=checkpoint,
        best_k_fname=best_k_fname,
        best_k_mode=best_k_mode,
        ckpt_subdir=ckpt_subdir,
    ):
        return ckpt

    # Check specified version subdir
    ver_name = None
    if version and (Path(dirpath) / version).is_dir():
        ver_name = version
    # Detect max version
    elif version:
        exist_versions = []
        for d in Path(dirpath).iterdir():
            if d.is_dir() and d.name.startswith(f"{version+'_'}"):
                exist_versions.append(int(d.name.split("_")[1]))
        if len(exist_versions) > 0:
            max_ver = max(exist_versions)
            ver_name = version + f"_{max_ver}"
    else:
        ver_name = None
    if ver_name:
        if ckpt := resolve_rel_ckpt(
            dirpath=Path(dirpath) / ver_name,
            checkpoint=checkpoint,
            best_k_fname=best_k_fname,
            best_k_mode=best_k_mode,
            ckpt_subdir=ckpt_subdir,
        ):
            return ckpt
    return None


def resolve_rel_ckpt(
    dirpath: Optional[str] = None,
    checkpoint: str = "last.ckpt",
    best_k_fname: str = constants.BEST_K_MAP_FNAME,
    best_k_mode: Literal["max", "min"] = "min",
    ckpt_subdir: str = constants.CHECKPOINT_DIR_NAME,
) -> Optional[str]:
    """Resolve checkpoint path rel to dirpath.

    Automatically search checkpoint in a directory's checkpoints subdir, normally names as `checkpoints`.
    The `dirpath` may has such default structure::

        ./dirpath/
        ├── best_k_models.yaml
        ├── last.ckpt
        └── abc.ckpt

        or

        ./dirpath/
        ├── hparams.yaml
        └── checkpoints
            ├── best_k_models.yaml
            ├── last.ckpt
            └── abc.ckpt

    First search dirpath , then fallback to its `ckpt_subdir` (checkpoints) subdir match the valid checpoint path
    . If all failed, return None.

    Note: ``checkpoint`` must a parameter of one level rel path to avoid mess matching. and deep rel path matching
    should in top level function but not here.

    - valid: (last.ckpt, best, ./last.ckpt)

    - invalid: (/last.ckpt, mypath/last.ckpt)

    Args:
        dirpath (Path or str, optional):
            The root path. Defaults to None, which means the current directory.
        checkpoint (str, optional):
            The file name of checkpoint to resolve, needs a suffix like ".ckpt" / ".pt",
            While checkpoint="best" is a preseved key means it will find `best_k_fname` which is
            a file contains `Dict[BEST_K_MODEL_PATH, BEST_K_SCORE]`, and sort by its score to
            match a best ckpt. Defaults to "last.ckpt".
        best_k_fname (str, optional):
            The filename for the best_k map file. Note that the best model path in best map file may
            not in this directory since it is stored in training stage, so we assume that its basename
            can matching ckpts in the same level. Defaults to best_k_models.yaml.
        best_k_mode (Literal["max", "min"], optional):
            The mode for selecting the best_k checkpoint. Defaults to "min".
        ckpt_subdir (str, optional):
            The name of the checkpoints subdir. Defaults to "checkpoints".


    Returns:
        Optional[str]: The resolved checkpoint path or None.

    Examples:
        >>> resolve_rel_ckpt('./dirpath', checkpoint='best')
        '/path/to/xxxl.ckpt'
    """
    orig_ckpt = checkpoint  # record orig to recurve
    dirpath = Path(dirpath) if dirpath is not None else Path().resolve()

    if not dirpath.is_dir():
        raise ValueError(f"dirpath=({dirpath}) is not a dir.")

    if len(Path(checkpoint).parts) > 1:
        raise ValueError(
            f"Resolve ckpt rel to dirpath requires on level file name, but got "
            f"checkpoint={checkpoint} with level ({len(Path(checkpoint).parts)})."
        )
    # best key case
    if checkpoint == "best":
        best_k_fpath = dirpath / best_k_fname
        if not best_k_fpath.is_file():
            return None
        best_k_maps = SerializationFn.load_file(best_k_fpath)
        _op = min if best_k_mode == "min" else max
        best_model_path_record = _op(best_k_maps, key=best_k_maps.get)  # type: ignore[arg-type]
        best_model_name = Path(best_model_path_record).name

        # the best model path in best map file may not in this directory, fetch its file base name
        best_model_path = dirpath / best_model_name
        if not best_model_path.is_file():
            raise ValueError(
                f"Resolved best model name {best_model_name} in best k map {best_k_fpath} "
                f"but failed to find ckpt name in dirpath ({best_model_path})."
            )
        return str(best_model_path)

    # can directly resolve
    checkpoint = dirpath / orig_ckpt

    checkpoint = str(Path(checkpoint)) if checkpoint.is_file() else None

    # Recurve ckpt_subdir on 1-depth.
    if checkpoint is None:
        if ckpt_subdir and (dirpath / ckpt_subdir).is_dir():
            checkpoint = resolve_rel_ckpt(
                str(dirpath / ckpt_subdir),
                checkpoint=orig_ckpt,
                best_k_fname=best_k_fname,
                best_k_mode=best_k_mode,
                ckpt_subdir=None,
            )
    return checkpoint
