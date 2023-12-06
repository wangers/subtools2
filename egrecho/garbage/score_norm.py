# -*- coding:utf-8 -*-

# Copyright xmuspeech (Author: Leo 2023-11)

import warnings
from collections import defaultdict
from functools import lru_cache
from pathlib import Path
from typing import Dict, List, Optional, Union

from tqdm.auto import tqdm

from egrecho.score.utils import compute_mean_stats
from egrecho.utils.io import (
    KaldiVectorReader,
    KaldiVectorWriter,
    buf_count_newlines,
    close_cached_kaldi_handles,
    get_filename,
    read_key_first_lists,
    read_key_first_lists_lazy,
)
import numpy as np


def norm_rows(x: np.ndarray):
    norms = np.einsum("ij,ij->i", x, x)
    return np.sqrt(norms)


def spk_vector_mean(
    embed_scp: Union[str, Path],
    spk2utt: Union[str, Path],
    output_scp: Optional[Union[str, Path]] = None,
):
    """
    Computes spk mean vector.

    It will first filt utts in spk2utt but not exists in embedding scp.

    Args:
        embed_scp:
            kaldi style xvector, i.e, xvector.scp.
        spk2utt:
            spk2utt file, per line::

            spk_name utt_id1 utt_id2 utt_id3 ...

        output_scp:
            If not provide, will be saved in the same dir of `embed_scp`
            as the `embed_scp`'s name with `"spk_"` preffix (e.g., `"spk_xvector.scp"`).

    Returns:
        The success saved vector path.
    """

    def _get_spk2utt_map(embed_scp, spk2utt) -> Dict[str, List[str]]:
        seen_utts = {utt for utt, _ in read_key_first_lists_lazy(embed_scp)}
        utts_without_emb = set()
        spks_without_emb = set()
        spk2utt = defaultdict(set)
        for spk, utts in read_key_first_lists(spk2utt, vector=True):
            for utt in utts:
                if utt not in seen_utts:
                    utts_without_emb.add(utt)
                else:
                    spk2utt[spk].add(utt)
            if spk not in spk2utt:
                spks_without_emb.add(spk)
            else:
                spk2utt[spk] = list(spk2utt[spk])
        if len(utts_without_emb) > 0 or len(spks_without_emb) > 0:
            warnings.warn(
                f"Find utts in spk2utt but not in embedding scp, filteted utts number: ({len(utts_without_emb)}), "
                f"spks number: ({len(spks_without_emb)}). Ignore this warning if don't mind about this case.",
                UserWarning,
                stacklevel=2,
            )
        return spk2utt

    for f in (embed_scp, spk2utt):
        if not Path(f).is_file():
            raise FileExistsError(f"({f}) is not a valid file.")

    # filts invalid utts in spk2utt
    spk2utt_map = _get_spk2utt_map(embed_scp, spk2utt)

    if not output_scp:
        embed_fname = get_filename(embed_scp)
        output_scp = (Path(embed_scp).parent) / ("spk_" + embed_fname)
    vec_reader = KaldiVectorReader(embed_scp)
    with KaldiVectorWriter(Path(output_scp).parent, Path(output_scp).name) as vec_w:
        for spk, utts in tqdm(
            spk2utt_map.items(),
        ):
            utt_embeds = (vec_reader.read(utt) for utt in utts)
            mean_vec = compute_mean_stats(utt_embeds)
            vec_w.write(spk, mean_vec)
    return output_scp


class SpkMeanComputer:
    """Computes spk mean vector.

    It will first filt utts in spk2utt but not exists in embedding files.

    Args:
        embed_scp:
            kaldi style xvector, i.e, xvector.scp.
        spk2utt:
            spk2utt file, per line::

            spk_name utt_id1 utt_id2 utt_id3 ...

        output_scp:
            If not provide, will be saved in the same dir of `embed_scp`
            as the `embed_scp`'s name with `"spk_"` preffix (e.g., `"spk_xvector.scp"`).

        cache_size (int or float):
            If greater than 0, caches embed when reading `embed_scp`.

            If float, cache size is controlled  by
            `cache_size * total_len`, where `total_len` is the size of `eval_scp`.

            If int, directly set cache size.
            Defaults to 1.0.
    """

    def __init__(
        self,
        embed_scp: Union[str, Path],
        spk2utt: Union[str, Path],
        output_scp: Optional[Union[str, Path]] = None,
        cache_size: Union[int, float] = 1.0,
    ) -> None:
        self.spk2utt = spk2utt
        self.embed_scp = embed_scp
        self.check_files()

        if not output_scp:
            embed_fname = get_filename(embed_scp)
            output_scp = (Path(embed_scp).parent) / ("spk_" + embed_fname)
        self.output_scp = output_scp

        self.spk2utt_map = self.get_spk2utt_map()

        self.vec_reader = self.get_embed_reader()
        self.enable_cache = False
        if cache_size > 0:
            if isinstance(cache_size, float):
                assert 0.0 <= cache_size <= 1.0
                total_len = buf_count_newlines(self.embed_scp)
                cache_size = int(total_len * cache_size)
            self.get_embed = lru_cache(maxsize=cache_size)(self.get_embed)
            self.enable_cache = True

    def check_files(self):
        for f in (self.embed_scp, self.spk2utt):
            if not Path(f).is_file():
                raise FileExistsError(f"({f}) is not a valid file.")

    def get_spk2utt_map(self) -> Dict[str, List[str]]:
        seen_utts = {utt for utt, _ in read_key_first_lists_lazy(self.embed_scp)}
        utts_without_emb = set()
        spks_without_emb = set()
        spk2utt = defaultdict(set)
        for spk, utts in read_key_first_lists(self.spk2utt, vector=True):
            for utt in utts:
                if utt not in seen_utts:
                    utts_without_emb.add(utt)
                else:
                    spk2utt[spk].add(utt)
            if spk not in spk2utt:
                spks_without_emb.add(spk)
            else:
                spk2utt[spk] = list(spk2utt[spk])
        if len(utts_without_emb) > 0 or len(spks_without_emb) > 0:
            warnings.warn(
                f"Find utts in spk2utt but not in embedding scp, filteted utts number: ({len(utts_without_emb)}), "
                f"spks number: ({len(spks_without_emb)}). Ignore this warning if don't mind about this case.",
                UserWarning,
            )
        return spk2utt

    def compute_spk_mean(self):
        self.vec_reader = self.get_embed_reader()  # update reader
        with KaldiVectorWriter(
            Path(self.output_scp).parent, Path(self.output_scp).name
        ) as vec_w:
            for spk, utts in tqdm(
                self.spk2utt_map.items(),
            ):
                utt_embeds = (self.get_embed(utt) for utt in utts)
                mean_vec = compute_mean_stats(utt_embeds)
                vec_w.write(spk, mean_vec)
        close_cached_kaldi_handles()
        if self.enable_cache:
            self.get_embed.cache_clear()

    def get_embed(self, utt: str):
        embed = self.vec_reader.read(utt)
        return embed

    def get_embed_reader(self):
        return KaldiVectorReader(self.embed_scp)


class ScoreNormer:
    def __init__(
        self,
        eval_scp: Union[str, Path],
        cohort_scp: Union[str, Path],
        trial_score_files: List[Union[str, Path]],
        top_n: int,
        storage_dir: Optional[Union[str, Path]] = "",
        submean_vec: Optional[Union[str, Path]] = None,
    ) -> None:
        self.trial_score_files = trial_score_files
        self.eval_scp = eval_scp
        self.cohort_scp = cohort_scp
        self.storage_dir = storage_dir
        self.check_files()
        # self.submean_vec = np.load(submean_vec) if submean_vec is not None

    def check_files(self):
        for trial in self.trial_score_files:
            if not Path(trial).is_file():
                raise FileExistsError(f"Trial: ({trial}) is not a valid file.")
        if not Path(self.eval_scp).is_file():
            raise FileExistsError(
                f"Eval xvecotr: ({self.eval_scp}) is not a valid file."
            )

        if not Path(self.cohort_scp).is_file():
            raise FileExistsError(
                f"Cohort xvecotr: ({self.eval_scp}) is not a valid file."
            )
