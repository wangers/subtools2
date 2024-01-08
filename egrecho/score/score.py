# -*- coding:utf-8 -*-
# Copyright xmuspeech (Author: Leo 2023-09)

from functools import lru_cache
from itertools import chain
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

import numpy as np
import torch
from kaldi_native_io import SequentialFloatVectorReader
from tqdm.auto import tqdm

from egrecho.score.binary_metrics import compute_metrics
from egrecho.score.utils import compute_mean_stats
from egrecho.utils.io import (
    KaldiVectorReader,
    buf_count_newlines,
    close_cached_kaldi_handles,
    get_filename,
    read_lists_lazy,
)


def vector_mean(
    embed_scp: Union[str, Path],
    output_file: Optional[Union[str, Path]] = None,
):
    """
    Computes spk mean vectors.

    It will first filt utts in spk2utt but not exists in embedding scp.

    Args:
        embed_scp:
            kaldi style xvector, i.e, xvector.scp.
        spk2utt:
            spk2utt file, per line::

            spk_name utt_id1 utt_id2 utt_id3 ...

        output_file:
            If not provide, will be saved in the same dir of ``embed_scp``
            with suffix ``".mean.npy"`` (e.g., ``"spk_xvector.scp"``).

    Returns:
        The success saved vector path.
    """

    if not output_file:
        output_file = str(embed_scp) + ".mean.npy"
    rspecifier = f"scp:{embed_scp}"
    with SequentialFloatVectorReader(rspecifier) as ki:
        it = (vec for _, vec in ki)
        mean = compute_mean_stats(it)
    np.save(output_file, mean)
    return output_file


class CosineScore:
    """This class is designed to compute cosine similarity.

    Args:
        eval_scp:
            kaldi style xvector, i.e, xvector.scp.
        submean_vec:
            A numpy array or file(.npy) as mean stats to be subtracted.
            Defaults to None, which means will be 0.0.
        cache_size (int or float):
            If greater than 0, caches embed when reading ``eval_scp``.

            If float, cache size is controlled  by
            ``cache_size * total_len``, where ``total_len`` is the size of ``eval_scp``.

            If int, directly set cache size.
            Defaults to 1.0.

    Example::

        scorer = CosineScore('./xvector.scp')
        # score_file will be `"./some_set.trials.score"`.
        score_file = scorer.score('./some_set.trials', storage_dir='./')

        or

        with CosineScore('./xvector.scp') as scorer:
            score_file = scorer.score('./some_set.trials')
            score_files = scorer.score('./some_set1.trials', './some_set2.trials')
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
        if (
            not isinstance(submean_vec, np.ndarray)
            and not Path(self.eval_scp).is_file()
        ):
            raise FileExistsError(
                f"submean np vector: ({submean_vec}) is not a valid file."
            )
        self.submean_vec = submean_vec
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
        storage_dir: Optional[Union[str, Path]] = None,
        submean_vec: Optional[Union[str, Path, np.ndarray]] = None,
    ) -> List[Path]:
        """Computes cosine similarity.

        Args:
            trial_files:
                pairwise file(s), 2 or 3 items per line::

                enroll_name test_name
                enroll_name test_name target/nontarget

            storage_dir:
                Where stores score result, score file name will be the ``trial_file``'s name with ``".score"`` suffix.
                score number insert in the 3rd column of ``trials_file``.
            submean_vec:
                A numpy array or file(.npy) as mean stats to be subtracted.
                can overwrite the instance's ``submean_vec`` attribute.
                Defaults to None, which means use ``self.submean_vec``.

        Returns:
            success saved score files path.
        """

        submean_vec = submean_vec if submean_vec is not None else self.submean_vec
        submean = self.load_mean_vec(submean_vec)

        success_files = []
        for trial in trial_files:
            if storage_dir is None:
                storage_dir = Path(trial).parent
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
                    emb1, emb2 = (
                        torch.from_numpy(self.get_embed(segs[0]) - submean),
                        torch.from_numpy(self.get_embed(segs[1]) - submean),
                    )
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

    @staticmethod
    def load_mean_vec(vec: Union[np.ndarray, str, Path, None]):
        if isinstance(vec, np.ndarray):
            return vec
        else:
            return np.load(vec) if vec is not None else 0.0

    def get_embed(self, utt: str):
        if self.vec_reader is None:
            self.vec_reader = self.get_embed_reader()
        return self.vec_reader.read(utt)

    def get_embed_reader(self):
        return KaldiVectorReader(self.eval_scp)

    def close(self):
        """clear caches"""
        close_cached_kaldi_handles()
        self.vec_reader = None
        if self.enable_cache:
            self.get_embed.cache_clear()

    def clear(self):
        """clear caches"""
        self.close()

    def __enter__(self):
        """Gives a context manager."""
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()

    def __del__(self):
        self.close()


def metrics_from_score_file(
    trial_score_file: Union[str, Path],
    p_target: float = 0.01,
    c_miss: float = 1.0,
    c_fa: float = 1.0,
) -> Dict[str, Any]:
    """Computes metrics (EER & minDCF) from score file.

    Args:
        trial_score_file: cosine score trial file.
        p_target: Prior probability for target speakers (default is 0.01).
        c_miss: Cost associated with a missed detection (default is 1).
        c_fa: Cost associated with a false alarm (default is 1).

    Returns:
        a dict containing EER, minDCF, and threshould corresponding to eer.
    """
    it = read_lists_lazy(trial_score_file, vector=True)
    it_ = iter(it)

    try:
        first = next(it_)
    except StopIteration:
        raise ValueError(f"Got a empty trial score file:{trial_score_file}")

    if len(first) != 4:
        raise ValueError(
            f"Trial score file should be 4 column formmated as (enroll_key, test_key, score, label). Got the first item={first}."
        )

    it_ = chain([first], it_)
    scores, labels = zip(*((float(data[2]), data[3] == "target") for data in it_))

    scores = np.asarray(scores).astype(np.float64)
    labels = np.asarray(labels).astype(np.int64)

    eer, min_dcf, thres = compute_metrics(
        scores, labels, p_target=p_target, c_miss=c_miss, c_fa=c_fa
    )

    return {
        "Score name": str(get_filename(trial_score_file)),
        "EER": round(eer * 100, 3),
        "minDCF": round(min_dcf, 4),
        "EER Threshould": round(thres, 5),
        "p_target": p_target,
        "c_miss": c_miss,
        "c_fa": c_fa,
    }
