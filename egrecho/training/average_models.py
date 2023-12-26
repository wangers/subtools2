# -*- coding:utf-8 -*-
# Copyright xmuspeech (Author: Leo 2023-11)

from pathlib import Path
from typing import List, Literal, Union

import torch

from egrecho.utils.constants import BEST_K_MAP_FNAME, CHECKPOINT_DIR_NAME
from egrecho.utils.cuda_utils import release_memory, to_device
from egrecho.utils.io.utils import SerializationFn


def average_best_models(
    dirpath: str,
    avg_num: int,
    version: str = "version",
    best_k_fname: str = BEST_K_MAP_FNAME,
    best_k_mode: Literal["max", "min"] = "min",
    ckpt_subdir: str = CHECKPOINT_DIR_NAME,
):
    """Averages models in directory.

    Structure may like::

        ./dirpath/version_1
                ├── hparams.yaml
                └── checkpoints
                    ├── best_k_models.yaml
                    ├── last.ckpt
                    └── abc.ckpt

    Args:
        dirpath (Path or str, optional):
            The root path. None means the current directory.
        avg_num:
            number models in best_k_models map files to be averaged.
        version (str, optional):
            The versioned subdir name. Conmmonly subdir is named as "version_0/version_1", if you specify
            the version name with a version num, it will search that version dir, otherwise choose the max number
            of version (above "version_1"). Defaults to "version". None means ignore it.
        best_k_fname (str, optional):
            The filename for the best_k map file. Note that the best model path in best map file may
            not in this directory since it is stored in training stage, so we assume that its basename
            can matching ckpts in the same level. Defaults to best_k_models.yaml.
        best_k_mode (Literal["max", "min"], optional):
            The mode for selecting the best_k checkpoint. Defaults to "min".
        ckpt_subdir (str, optional):
            The name of the checkpoints subdir. Defaults to "checkpoints". None means ignore it.

    """
    assert avg_num > 0
    dirpath = Path(dirpath) if dirpath is not None else Path().resolve()
    if not dirpath.is_dir():
        raise ValueError(f"dirpath=({dirpath}) is not a dir.")

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
        dirpath = dirpath / ver_name

    # search checkpoint dir
    if ckpt_subdir and (dirpath / ckpt_subdir).is_dir():
        dirpath = dirpath / ckpt_subdir
    best_k_fpath = dirpath / best_k_fname
    if not best_k_fpath.is_file():
        raise FileExistsError(f"Invalid best_k_models map file: {best_k_fpath}")

    best_k_maps = SerializationFn.load_file(best_k_fpath)
    if avg_num > len(best_k_maps):
        raise ValueError(
            f"Intend to average {avg_num} number of models, but only got {len(best_k_maps)} "
            f"number of models in best_k_models map file {best_k_fpath}"
        )
    sorted_model_paths = sorted(
        best_k_maps, key=best_k_maps.get, reverse=(best_k_mode == "max")
    )

    # the best model path in best map file may not in this directory, fetch its file base name
    soted_model_names = [
        Path(sorted_model_path).name for sorted_model_path in sorted_model_paths
    ]
    sorted_model_paths = [
        (dirpath / soted_model_name) for soted_model_name in soted_model_names
    ]

    # top avg_num
    ready2avg = sorted_model_paths[:avg_num]
    dest_path = dirpath / f"average_{avg_num}.ckpt"
    average_models(ready2avg, dest_path=dest_path)
    release_memory()  # gc
    return dest_path


def average_models(model_paths: List[Union[str, Path]], dest_path: Union[str, Path]):
    if isinstance(model_paths, (str, Path)):
        model_paths = (model_paths,)
    assert isinstance(model_paths, (list, tuple))
    num = len(model_paths)

    print(f"#### Processing {model_paths[0]}")
    first_model = torch.load(model_paths[0])
    is_pl_model = "state_dict" in first_model
    avg_states = first_model["state_dict"] if is_pl_model else first_model
    device = next(
        (t for t in avg_states.values() if isinstance(t, torch.Tensor)),
        torch.tensor(0),
    ).device

    avg_states = to_device(avg_states, "cpu")
    for path in model_paths[1:]:
        print(f"#### Processing {path}")
        states = torch.load(path, map_location=torch.device("cpu"))
        states = states["state_dict"] if "state_dict" in states else states
        for k in avg_states.keys():
            avg_states[k] += states[k]
    # average
    for k in avg_states.keys():
        if avg_states[k] is not None:
            # pytorch 1.6 use true_divide instead of /=
            avg_states[k] = torch.true_divide(avg_states[k], num)
    avg_states = to_device(avg_states, device)
    if is_pl_model:
        first_model["state_dict"] = avg_states
    else:
        first_model = avg_states
    torch.save(first_model, dest_path)
    print(f"#### Saving to {dest_path} done.")
