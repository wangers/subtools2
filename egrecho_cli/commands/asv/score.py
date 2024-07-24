# -*- coding:utf-8 -*-
# Copyright xmuspeech (Author: Leo 2023-09)

import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

from egrecho.core.parser import BaseCommand, CommonParser
from egrecho.score import (
    CosineScore,
    ScoreNorm,
    metrics_from_score_file,
    spk_vector_mean,
    vector_mean,
)
from egrecho.utils.io import DataFilesList, get_filename, resolve_file, save_json
from egrecho.utils.logging import get_logger
from egrecho.utils.misc import ConfigurationException
from egrecho_cli.register import register_command

logger = get_logger()


def score_set(
    eval_scp: str,
    trial_fpatterns: Union[str, List[str]],
    scp_dir: Optional[str] = None,
    trial_dir: Optional[str] = None,
    helper_scp: Optional[str] = None,
    skip_mean: bool = False,
    skip_metric: bool = False,
    spk2utt: Optional[str] = None,
    submean: bool = True,
    score_norm: bool = True,
    sn_kwds: Optional[Dict[str, Any]] = None,
    metric_kwds: Optional[Dict[str, Any]] = None,
    result_file: Optional[str] = None,
    verbose: bool = True,
):
    """Scores sv.

    Args:
        eval_scp:
            eval xvector scp.
        trial_fpatterns:
            trial filename patterns, support unix matching.
        scp_dir:
            Where to match scp (eval_scp, help_scp) when given scp is relative.
            Special case to read `scp_dir` when passed is a file, which means a dir is saved in this file.
        trial_dir:
            Path (e.g. `"./"`) directory have trial files.
            Where to match when given `trial_fpatterns` is relative.
        helper_scp:
            xvector scp for submean & scorenorm.
            submean -> `helper_scp + '.mean.npy'`, cohorset -> add `'spk_'` prefix to `helper_scp` filename.
        skip_mean:
            Whether skip prepare mean vector such as spk_xvector.scp, xvector.mean.npy.
        skip_metric:
            Might we just want to get score file. defualts to False.
        sn_kwds:
            Some kwargs for score norm.
        verbose:
            whether std out the metric results.
        result_file:
            If specify set, metrics dicts will be saved as json style.
            defaults save to the dir `scp_dir` with name eer.results.json.
    """
    if Path(scp_dir).is_file():
        with open(scp_dir, "r", encoding="utf-8") as f:
            scp_dir = f.read()

    eval_scp = resolve_file(eval_scp, scp_dir)

    if not Path(eval_scp).is_file():
        raise FileNotFoundError(f"Invalid eval file {eval_scp}.")
    trial_files = DataFilesList.from_local_or_remote(
        trial_fpatterns, base_path=trial_dir
    )
    need_help_scp = submean + score_norm
    mean_vec_path, spk_vec_path = None, None
    if need_help_scp:
        if not helper_scp:
            raise ConfigurationException(
                f"Need a xvec scp such as trainset to apply submean={submean}, score_norm={score_norm}."
            )
        helper_scp = resolve_file(helper_scp, scp_dir)
        if not skip_mean:
            if submean:
                vector_mean(helper_scp)
            if score_norm:
                if not (spk2utt and Path(spk2utt).is_file()):
                    raise ConfigurationException(
                        f"Need a valid spk2utt to get coroset xcector, but got {spk2utt}."
                    )
                spk_vector_mean(helper_scp, spk2utt)

        mean_vec_path = str(helper_scp) + ".mean.npy" if submean else None
        spk_vec_path = (Path(helper_scp).parent) / ("spk_" + get_filename(helper_scp))
    if verbose:
        print(f"#### Score cosine: submean {mean_vec_path}.")
    with CosineScore(eval_scp=eval_scp, submean_vec=mean_vec_path) as cs:
        score_files = cs.score(*trial_files, storage_dir=scp_dir)
    metrics = []
    metric_kwds = metric_kwds or {}
    if not skip_metric:
        for f in score_files:
            metr = metrics_from_score_file(f, **metric_kwds)
            metrics.append(metr)
            if verbose:
                print(json.dumps(metr, indent=4))

    if score_norm:
        sn_kwds = sn_kwds or {}
        if verbose:
            print(
                f"#### Score Norm: submean {mean_vec_path}, \n#### cohort: {spk_vec_path}"
            )
        with ScoreNorm(
            eval_scp, cohort_scp=spk_vec_path, submean_vec=mean_vec_path, **sn_kwds
        ) as sn:
            score_files = sn.norm(*score_files)
        if not skip_metric:
            for f in score_files:
                metr = metrics_from_score_file(f, **metric_kwds)
                metrics.append(metr)
                if verbose:
                    print(json.dumps(metr, indent=4))
    if metrics:
        save_json(metrics, result_file) if result_file else save_json(
            metrics, Path(scp_dir) / "eer.results.json"
        )

    return metrics


DESCRIPTION = """Scores sv."""


@register_command(name="score-sv", help=DESCRIPTION)
class Score(BaseCommand):
    @classmethod
    def get_dummy_parser(cls) -> CommonParser:
        return CommonParser(description=DESCRIPTION)

    @classmethod
    def setup_parser(cls, parser: CommonParser):
        parser.add_cfg_flag()

        parser.add_function_arguments(score_set, as_positional=True)
        parser.set_defaults(
            sn_kwds={"top_n": 300, "cohort_sub": None}, metric_kwds={"p_target": 0.01}
        )

        return parser

    @staticmethod
    def run_from_args(args, parser: Optional[CommonParser] = None):
        args.pop("cfg", None)
        score_set(**args)


if __name__ == "__main__":
    parser = Score.get_dummy_parser()
    parser = Score.setup_parser(parser)
    args = parser.parse_args()
    Score.run_from_args(args, parser)
