# -*- coding:utf-8 -*-

# Copyright xmuspeech (Author: Leo 2023-11)

import random
import warnings
from collections import defaultdict
from functools import lru_cache
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union

import numpy as np
from tqdm.auto import tqdm

from egrecho.score.utils import compute_mean_stats
from egrecho.utils.io import (
    KaldiVectorReader,
    KaldiVectorWriter,
    close_cached_kaldi_handles,
    get_filename,
    read_key_first_lists,
    read_key_first_lists_lazy,
    read_lists_lazy,
)
from egrecho.utils.logging import get_logger

logger = get_logger()


def spk_vector_mean(
    embed_scp: Union[str, Path],
    spk2utt: Union[str, Path],
    output_scp: Optional[Union[str, Path]] = None,
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

        output_fname:
            If not provide, will be saved in the same dir of `embed_scp`
            as the `embed_scp`'s name with `"spk_"` preffix (e.g., `"spk_xvector.scp"`).

    Returns:
        The success saved vector path.
    """

    def _get_spk2utt_map(embed_scp, spk2utt) -> Dict[str, List[str]]:
        seen_utts = {utt for utt, _ in read_key_first_lists_lazy(embed_scp)}
        utts_without_emb = set()
        spks_without_emb = set()
        spk2utt_map = defaultdict(set)
        for spk, utts in read_key_first_lists(spk2utt, vector=True):
            for utt in utts:
                if utt not in seen_utts:
                    utts_without_emb.add(utt)
                else:
                    spk2utt_map[spk].add(utt)
            if spk not in spk2utt_map:
                spks_without_emb.add(spk)
            else:
                spk2utt_map[spk] = list(spk2utt_map[spk])
        if len(utts_without_emb) > 0 or len(spks_without_emb) > 0:
            warnings.warn(
                f"Find utts in spk2utt but not in embedding scp, reduced utts number: ({len(utts_without_emb)}), "
                f"spks number: ({len(spks_without_emb)}). Ignore this warning if don't mind about this case.",
                UserWarning,
                stacklevel=1,
            )
        return spk2utt_map

    for f in (embed_scp, spk2utt):
        if not Path(f).is_file():
            raise FileExistsError(f"({f}) is not a valid file.")

    # filts invalid utts in spk2utt
    spk2utt_map = _get_spk2utt_map(embed_scp, spk2utt)

    if not output_scp:
        embed_fname = get_filename(embed_scp)
        if embed_fname.endswith(".scp"):
            embed_fname = embed_fname[:-4]
        output_scp = (Path(embed_scp).parent) / ("spk_" + embed_fname)
    vec_reader = KaldiVectorReader(embed_scp)
    with KaldiVectorWriter(Path(output_scp).parent, Path(output_scp).name) as vec_w:
        for spk, utts in tqdm(spk2utt_map.items(), desc="Cohort vec mean"):
            utt_embeds = (vec_reader.read(utt) for utt in utts)
            mean_vec = compute_mean_stats(utt_embeds)
            vec_w.write(spk, mean_vec)
    return output_scp


def compute_cohort_stats(
    embed: np.ndarray,
    cohort_embed: np.ndarray,
    top_n: int,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Computes closest cohort stats (mean, std).

    This function calculates the cosine similarity of test/enroll embeddings to the cohort set embeddings.
    It selects the top `n` scores of each sample to compute mean and standard deviation.

    NOTE:
        Matrix calculating to speedup computes cosine similarity, too many cohort samples need to be subset
        to save memory.

    Args:
        embed: ndarray of shape (n_samples, n_dim)
            test/enroll embeddings.
        cohort_embed: ndarray of shape (m_samples, n_dim)
            embeddings of cohort set.
        top_n:
            Number of top positive scores to compute statistics.

    Returns:
        Tuple containing mean and std arrays (n_samples,) of test/enroll cohort scores.
    """
    # Calculate cosine similarity
    embed_normalized = embed / np.linalg.norm(embed, axis=1, keepdims=True)
    cohort_embed_normalized = cohort_embed / np.linalg.norm(
        cohort_embed, axis=1, keepdims=True
    )
    emb_cohort_score = embed_normalized @ cohort_embed_normalized.T

    # Select top_n
    top_n = min(emb_cohort_score.shape[1], max(top_n, 1))
    emb_cohort_score_topn = np.partition(emb_cohort_score, -top_n, axis=1)[:, -top_n:]

    # Compute mean and std
    emb_cohort_mean = np.mean(emb_cohort_score_topn, axis=1)
    emb_cohort_std = np.std(emb_cohort_score_topn, axis=1)

    return emb_cohort_mean, emb_cohort_std


class ScoreNorm:
    """This class is designed for asnorm (Adaptive Symmetrical Normalization).

    Workflow is defined as a sequence of the following operations:

        Loads test/enroll embed and cohort set embed -> Computes consine score of test/enroll relative to cohort ->
        Select topn score to get cohort stats -> Normalize test/enroll raw scores -> Save score files

    Args:
        eval_scp:
            kaldi style xvector, i.e, xvector.scp.
        cohort_scp:
            kaldi style xvector of cohort set, e.g, spk_xvector.scp.
        top_n:
            Number of top positive scores to compute statistics. Recommad to 200 ~ 400.
        output_prefix:
            output score file name is raw score trial file name with this prefix.
            Defaults to `'cohort'`. (e.g., cohort_somset.trials.score)
        storage_dir:
            Where stores score result,if not set, defaults to the dir of trial files.
        cohort_sub:
            int or float, float means the proportion of cohort set. If not set, use the whole cohort set.
            Defaults to None.
        submean_vec:
            A numpy array or file(.npy) as mean stats to be subtracted.
            Defaults to None, which means will be 0.0.

    NOTE:
        Score trial files is passed in call function of instance.

        In order to speedup processing, we load all embeddings of test/enrolls keys and use matrix when
        calculating scores of test/enroll to cohort set.

        Thus, the embeddings of test/enrolls is not cached for different score trial files (i.e., each calling of
        asnorm for one trial file needs reload embeddings), which leaves memory for matrix cosine.

        What's more, if test set is really huge, the trials file may need to be splitted and scores multi times manually.

        Also, the given cohort_scp can be subset via parameter: `cohort_sub`
        and reduce the matrix row number, but the results may be slightly infulenced cause the top n score is related
        to sample number.

    Reference:
        Matejka, P., Novotný, O., Plchot, O., Burget, L., Sánchez, M. D., & Cernocký, J. (2017). Analysis of
        Score Normalization in Multilingual Speaker Recognition. Paper presented at the Interspeech.

        https://github.com/wenet-e2e/wespeaker/blob/master/wespeaker/bin/score_norm.py

    Example::

        sn = ScoreNorm('./xvector.scp',cohort_scp='./spk_xvector.scp', top_n=100)
        # score_file will be `"./cohort_some_set.trials.score"`.
        score_file = sn.norm('./some_set.trials.score')

        or

        with ScoreNorm('./xvector.scp',cohort_scp='./spk_xvector.scp', top_n=100) as sn:
            score_file = sn.norm('./some_set.trials')
            score_files = sn.norm('./some_set1.trials.score', './some_set2.trials.score')
    """

    def __init__(
        self,
        eval_scp: Union[str, Path],
        cohort_scp: Union[str, Path],
        top_n: int = 300,
        output_prefix: str = "cohort",
        storage_dir: Optional[Union[str, Path]] = None,
        cohort_sub: Union[float, int, None] = None,
        submean_vec: Optional[Union[str, Path]] = None,
    ) -> None:
        self.eval_scp = eval_scp
        self.cohort_scp = cohort_scp
        self.top_n = top_n
        self.output_prefix = output_prefix
        self.storage_dir = storage_dir
        self.submean_vec = submean_vec
        self.check_files()
        self.cohort_sub = cohort_sub

        self.eval_reader = KaldiVectorReader(eval_scp)
        self.get_cohort_vectors()  # cache

    def check_files(self):
        if not Path(self.eval_scp).is_file():
            raise FileExistsError(
                f"Eval xvecotr: ({self.eval_scp}) is not a valid file."
            )

        if not Path(self.cohort_scp).is_file():
            raise FileExistsError(
                f"Cohort xvecotr: ({self.eval_scp}) is not a valid file."
            )
        if (
            not isinstance(self.submean_vec, np.ndarray)
            and not Path(self.submean_vec).is_file()
        ):
            raise FileExistsError(
                f"submean np vector: ({self.submean_vec}) is not a valid file."
            )

    @lru_cache
    def get_cohort_vectors(self, seed=42) -> np.ndarray:
        """
        Get cohort vectors of shape (n_samples, embed_dim).
        """
        cohort_keys = [utt for utt, _ in read_key_first_lists_lazy(self.cohort_scp)]
        # subset
        cohort_sub = self.cohort_sub
        if cohort_sub:
            total_len = len(cohort_keys)
            if isinstance(cohort_sub, float):
                if not 0 < cohort_sub <= 1:
                    raise ValueError(
                        f"Require float between `(0, 1]` or int for parameter: `cohort_sub`, but got {cohort_sub}."
                    )
                sub_size = int(total_len * cohort_sub)
            elif isinstance(cohort_sub, int):
                sub_size = max(min(cohort_sub, 1), total_len)
            else:
                raise ValueError(
                    f"Require float or int for parameter:`cohort_sub`, but got invalid type {type(cohort_sub)}."
                )
            cohort_keys = random.Random(seed=seed).sample(cohort_keys, sub_size)
        cohort_keys = list(set(cohort_keys))
        cohort_reader = KaldiVectorReader(self.cohort_scp)

        return np.array([cohort_reader.read(utt) for utt in cohort_keys])

    def norm(
        self,
        *trial_score_files: Union[str, Path],
        storage_dir: Optional[Union[str, Path]] = None,
        output_prefix: Optional[Union[str, Path]] = None,
        submean_vec: Optional[Union[str, Path]] = None,
    ) -> List[Path]:
        """Applys asnorm.

        See doc of this class for detail. Workflow is defined as a sequence of the following operations:

        Loads test/enroll embed and cohort set embed -> Computes consine score of test/enroll relative to cohort ->
        Select topn score to get cohort stats -> Normalize test/enroll raw scores -> Save score files

        Args:
            trial_score_file(s) (str or Path, positional):
                cosine score trial file(s)
            storage_dir:
                overwrite `self.storage_dir`.
            output_prefix:
                overwrite `self.output_prefix`.
            submean_vec:
                overwrite `self.submean_vec`.

        Returns:
            success saved score files path.
        """
        submean_vec = submean_vec if submean_vec is not None else self.submean_vec
        submean = self.load_mean_vec(submean_vec)
        output_prefix = (
            output_prefix if output_prefix is not None else self.output_prefix
        )
        storage_dir = storage_dir if storage_dir is not None else self.storage_dir

        cohort_vecs = self.get_cohort_vectors() - submean
        success_files = []
        for trial in trial_score_files:
            if storage_dir is None:
                storage_dir = Path(trial).parent
            trial_name = str(get_filename(trial))
            store_path = Path(storage_dir) / (output_prefix + "_" + trial_name)

            e_t_gen = (
                (e_t_score[0], e_t_score[1])
                for e_t_score in read_lists_lazy(trial, vector=True)
            )
            e_lst, t_lst = zip(*e_t_gen)
            e_lst, t_lst = sorted(list(set(e_lst))), sorted(list(set(t_lst)))
            e_utt2idx = {utt: idx for idx, utt in enumerate(e_lst)}
            t_utt2idx = {utt: idx for idx, utt in enumerate(t_lst)}
            e_vecs = self.get_eval_vectors(e_lst) - submean
            t_vecs = self.get_eval_vectors(t_lst) - submean
            logger.info(
                f"Start compute cohort stats, got enroll number of ({e_vecs.shape[0]}),  "
                f"test number of ({t_vecs.shape[0]}), cohort number of ({cohort_vecs.shape[0]})."
            )
            e_cohort_mean, e_cohort_std = compute_cohort_stats(
                e_vecs, cohort_vecs, top_n=self.top_n
            )
            t_cohort_mean, t_cohort_std = compute_cohort_stats(
                t_vecs, cohort_vecs, top_n=self.top_n
            )
            logger.debug("Computes cohort stats done.")

            with open(store_path, "w") as score_w:
                for e_t_score in read_lists_lazy(trial, vector=True):
                    e_t_score = list(e_t_score)
                    e, t, score = e_t_score[0], e_t_score[1], float(e_t_score[2])
                    score_normalized = 0.5 * (
                        (score - e_cohort_mean[e_utt2idx[e]])
                        / e_cohort_std[e_utt2idx[e]]
                        + (score - t_cohort_mean[t_utt2idx[t]])
                        / t_cohort_std[t_utt2idx[t]]
                    )
                    e_t_score[2] = f"{score_normalized:.5f}"

                    score_w.write(" ".join(e_t_score) + "\n")

            success_files.append(store_path)
            logger.info(f"Normalize {trial_name} done.")

        return success_files

    def get_eval_vectors(self, utts: List[str]) -> np.ndarray:
        """
        Get the corresponding embedding array for the utts in `eval_scp`.

        Args:
            utts (List[str]): List of utterance keys.

        Returns:
            np.ndarray: Array of embeddings with shape `(nsamples, emb_dim)`.
        """
        return np.array([self.eval_reader.read(utt) for utt in utts])

    @staticmethod
    def load_mean_vec(vec: Union[np.ndarray, str, Path, None]):
        if isinstance(vec, np.ndarray):
            return vec
        else:
            return np.load(vec) if vec is not None else 0.0

    def close(self):
        """clear caches"""
        close_cached_kaldi_handles()
        self.get_cohort_vectors.cache_clear()

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
