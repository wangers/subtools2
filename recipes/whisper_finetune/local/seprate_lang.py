# (Author: Leo 2024-06-04)

import logging
import os
from pathlib import Path
from typing import Optional

import torch
from jsonargparse import CLI
from lhotse import (
    CutSet,
    Fbank,
    FbankConfig,
    LilcomChunkyWriter,
    WhisperFbank,
    WhisperFbankConfig,
)


def seperate_lang(
    src_cut: Path = './data/manifests/olr21_test/cuts.jsonl.gz',
    langs: str = 'vi',
    feats_dir: Optional[str] = None,
    num_mel_bins: int = 80,
    perturb_speed: bool = False,
    whisper_fbank: bool = False,
):
    """Seprate multi-lang cuts to singles.

    Args:
        src_cut:
            source file.
        langs:
            Comma or blank sep langs.
            e.g., ``"vi,yue"``, ``"vi yue"``.
        feats_dir:
            If given (e.g., data/fbank), It will extract fbank and results a dir includes feats and its manifests.
        num_mel_bins:
            The number of mel bins for Fbank.
        perturb_speed:
            Enable 0.9 and 1.1 speed perturbation for data augmentation. Default: False.
        whisper_fbank:
            Use WhisperFbank instead of Fbank. Default: False.
    """
    src_cut = Path(src_cut)
    tgt_langs = langs.replace(',', ' ').strip().split()
    cuts: CutSet = CutSet.from_file(src_cut)

    cuts_to_extract = {}  # {SRC_FILE: TGT_FILE}
    for lang in tgt_langs:
        lang_cut_f = src_cut.parent / (f'{lang}_' + src_cut.name)
        tgt_cut_f = (
            Path(feats_dir) / (f'{lang}_feats_' + src_cut.name) if feats_dir else None
        )
        if lang_cut_f.is_file():
            logging.warning(
                f'{tgt_langs} exists, skip it. or delete it manually and re-run.'
            )
            # already have src cuts, need to extract feature.
            if tgt_cut_f and not tgt_cut_f.is_file():
                cuts_to_extract[lang_cut_f] = tgt_cut_f
        else:

            cuts = cuts.filter_supervisions(
                lambda s: (s.language or '') == lang
            ).filter(lambda c: bool(c.supervisions))
            cuts.to_file(lang_cut_f)
            logging.info(f'Generated {lang_cut_f}.')
            # always extract feats as src cuts is generated.
            if tgt_cut_f:
                cuts_to_extract[lang_cut_f] = tgt_cut_f

    if not cuts_to_extract:
        return

    num_jobs = min(15, os.cpu_count())
    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)

    if whisper_fbank:
        extractor = WhisperFbank(
            WhisperFbankConfig(num_filters=num_mel_bins, device="cpu")
        )
    else:
        extractor = Fbank(FbankConfig(num_mel_bins=num_mel_bins))
    logging.info(f"Ready to Extract from {list(cuts_to_extract.items())}.")
    for lang_cut_f, tgt_f in cuts_to_extract.items():
        lang_cuts: CutSet = CutSet.from_file(lang_cut_f)
        logging.info(f"Extracting:\n({lang_cut_f}) -> ({tgt_f}).")
        if perturb_speed:
            logging.info("Doing speed perturb")
            lang_cuts = (
                lang_cuts + lang_cuts.perturb_speed(0.9) + lang_cuts.perturb_speed(1.1)
            )

        lang_cuts = lang_cuts.compute_and_store_features(
            extractor=extractor,
            storage_path=f"{feats_dir}/{str(Path(tgt_f).name).split('.')[0]}",
            # when an executor is specified, make more partitions
            num_jobs=num_jobs,
            storage_type=LilcomChunkyWriter,
        )
        lang_cuts.to_file(tgt_f)


if __name__ == "__main__":
    formatter = "%(asctime)s %(levelname)s [%(filename)s:%(lineno)d] %(message)s"

    logging.basicConfig(format=formatter, level=logging.INFO)
    CLI(seperate_lang)
