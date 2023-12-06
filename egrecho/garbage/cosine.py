# -*- coding:utf-8 -*-
# Copyright xmuspeech (Author: Leo 2023-09)

from functools import lru_cache
from pathlib import Path
from typing import List, Optional, Union

import numpy as np
import torch
from tqdm import tqdm

from egrecho.utils.io import (
    KaldiVectorReader,
    buf_count_newlines,
    close_cached_kaldi_handles,
    get_filename,
)


class CosineScore:
    """Computes cosine similarity.

    Args:
        eval_scp:
            kaldi style xvector, i.e, xvector.scp.
        trial_files:
            pairwise file(s), 2 or 3 items per line::

            enroll_name test_name
            enroll_name test_name target/nontarget

        storage_dir:
            Where stores score result, score file name will be the `trial_file`'s name with `".score"` suffix.
            score number insert in the 3rd column of `trials_file`.
        submean_vec:
            A numpy (.npy) stores mean stats to be subtracted. if None, disable submean.
            Defaults to None.
        cache_size (int or float):
            If greater than 0, caches embed when reading `eval_scp`.

            If float, cache size is controlled  by
            `cache_size * total_len`, where `total_len` is the size of `eval_scp`.

            If int, directly set cache size.
            Defaults to 1.0.
    """

    def __init__(
        self,
        eval_scp: Union[str, Path],
        trial_files: List[Union[str, Path]],
        storage_dir: Optional[Union[str, Path]] = "",
        submean_vec: Optional[Union[str, Path]] = None,
        cache_size: Union[int, float] = 1.0,
    ) -> None:
        if isinstance(trial_files, str):
            trial_files = (trial_files,)
        elif isinstance(trial_files, (tuple, list)):
            pass
        self.trial_files = trial_files
        self.eval_scp = eval_scp
        self.storage_dir = storage_dir
        self.check_files()
        self.submean_vec = np.load(submean_vec) if submean_vec is not None else 0.0
        self.vec_reader = self.get_embed_reader()

        self.enable_cache = False
        if cache_size > 0:
            if isinstance(cache_size, float):
                assert 0.0 <= cache_size <= 1.0
                total_len = buf_count_newlines(self.eval_scp)
                cache_size = int(total_len * cache_size)
            self.get_embed = lru_cache(maxsize=cache_size)(self.get_embed)
            self.enable_cache = True

    def check_files(self):
        for trial in self.trial_files:
            if not Path(trial).is_file():
                raise FileExistsError(f"Trial: ({trial}) is not a valid file.")
        if not Path(self.eval_scp).is_file():
            raise FileExistsError(
                f"Eval xvecotr: ({self.eval_scp}) is not a valid file."
            )

    def compute_scores(self):
        self.vec_reader = self.get_embed_reader()  # update reader

        for trial in self.trial_files:
            trial_name = str(get_filename(trial))
            store_path = Path(self.storage_dir) / (trial_name + ".score")
            trial_len = buf_count_newlines(trial)
            msg = trial_name.split(".")[0]

            with open(trial, "r") as trial_r, open(store_path, "w") as score_w, tqdm(
                total=trial_len,
                leave=False,
                unit=" pairs",
                desc=f"Score [{msg}]",
            ) as pbar:
                for line in trial_r:
                    segs = line.strip().split()
                    emb1, emb2 = self.get_embed(segs[0]), self.get_embed(segs[1])
                    score = torch.nn.functional.cosine_similarity(
                        emb1.reshape(1, -1), emb2.reshape(1, -1)
                    )[0].item()
                    ready_line = f"{segs[0]} {segs[1]} {score:.5f}"
                    if len(segs) == 3:  # enroll_name test_name target/nontarget
                        ready_line += f" {segs[2]}"
                    score_w.write(ready_line + "\n")
                    pbar.update()
            print(f"#### Scoring {msg} done.")
        close_cached_kaldi_handles()
        if self.enable_cache:
            self.get_embed.cache_clear()

    def get_embed(self, utt: str):
        embed = self.vec_reader.read(utt)
        return torch.from_numpy(embed - self.submean_vec)

    def get_embed_reader(self):
        return KaldiVectorReader(self.eval_scp)


class CosineScorer:
    """Computes cosine similarity.

    Args:
        eval_scp:
            kaldi style xvector, i.e, xvector.scp.
        submean_vec:
            A numpy (.npy) stores mean stats to be subtracted. if None, disable submean.
            Defaults to None.
        cache_size (int or float):
            If greater than 0, caches embed when reading `eval_scp`.

            If float, cache size is controlled  by
            `cache_size * total_len`, where `total_len` is the size of `eval_scp`.

            If int, directly set cache size.
            Defaults to 1.0.
    """

    def __init__(
        self,
        eval_scp: Union[str, Path],
        submean_vec: Optional[Union[str, Path]] = None,
        cache_size: Union[int, float] = 1.0,
    ) -> None:
        self.eval_scp = eval_scp

        if not Path(self.eval_scp).is_file():
            raise FileExistsError(
                f"Eval xvecotr: ({self.eval_scp}) is not a valid file."
            )
        self.submean_vec = np.load(submean_vec) if submean_vec is not None else 0.0
        self.vec_reader = self.get_embed_reader()

        self.enable_cache = False
        if cache_size > 0:
            if isinstance(cache_size, float):
                assert 0.0 <= cache_size <= 1.0
                total_len = buf_count_newlines(self.eval_scp)
                cache_size = int(total_len * cache_size)
            self.get_embed = lru_cache(maxsize=cache_size)(self.get_embed)
            self.enable_cache = True

    def score(
        self,
        *trial_files: Union[str, Path],
        storage_dir: Optional[Union[str, Path]] = "",
    ):
        """
        trial_files:
            pairwise file(s), 2 or 3 items per line::

            enroll_name test_name
            enroll_name test_name target/nontarget

        storage_dir:
            Where stores score result, score file name will be the `trial_file`'s name with `".score"` suffix.
            score number insert in the 3rd column of `trials_file`.
        """
        success_files = []
        for trial in trial_files:
            trial_name = str(get_filename(trial))
            store_path = Path(storage_dir) / (trial_name + ".score")
            trial_len = buf_count_newlines(trial)
            msg = trial_name.split(".")[0]

            with open(trial, "r") as trial_r, open(store_path, "w") as score_w, tqdm(
                total=trial_len,
                leave=False,
                unit=" pairs",
                desc=f"Score [{msg}]",
            ) as pbar:
                for line in trial_r:
                    segs = line.strip().split()
                    emb1, emb2 = self.get_embed(segs[0]), self.get_embed(segs[1])
                    score = torch.nn.functional.cosine_similarity(
                        emb1.reshape(1, -1), emb2.reshape(1, -1)
                    )[0].item()
                    ready_line = f"{segs[0]} {segs[1]} {score:.5f}"
                    if len(segs) == 3:  # enroll_name test_name target/nontarget
                        ready_line += f" {segs[2]}"
                    score_w.write(ready_line + "\n")
                    pbar.update()
            success_files.append(store_path)
            print(f"#### Scoring {msg} done.")
        return success_files

    def get_embed(self, utt: str):
        embed = self.vec_reader.read(utt)
        return torch.from_numpy(embed - self.submean_vec)

    def get_embed_reader(self):
        return KaldiVectorReader(self.eval_scp)

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()

    def close(self):
        close_cached_kaldi_handles()
        if self.enable_cache:
            self.get_embed.cache_clear()

    def __del__(self):
        self.close()
