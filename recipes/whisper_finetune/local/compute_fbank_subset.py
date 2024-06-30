#!/usr/bin/env python3
# Copyright    2021  Xiaomi Corp.        (authors: Fangjun Kuang)
#
# See ../../../../LICENSE for clarification regarding multiple authors
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

# (Author: Leo 2024-06-04)
"""
This file computes fbank features.
It looks for manifests in the directory data/manifests.
"""

import logging
import os
from pathlib import Path

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
from lhotse.recipes.utils import read_manifests_if_cached

# Torch's multithreaded behavior needs to be disabled or
# it wastes a lot of CPU and slow things down.
# Do this outside of main() in case it needs to take effect
# even when we are not invoking the main (e.g. when spawning subprocesses).
torch.set_num_threads(1)
torch.set_num_interop_threads(1)


def compute_fbank_subset(
    num_mel_bins: int = 80,
    perturb_speed: bool = False,
    whisper_fbank: bool = False,
    output_dir: str = "data/fbank",
    prefix: str = 'egs',
    parts: str = 'all',
    src_suffix: str = "jsonl",
    tgt_suffix: str = "jsonl.gz",
    device: str = 'cpu',
):
    """
    Computes offline fbank from data/manifests.
    The manifests are searched for using the pattern ``f'{prefix}_{manifest}_{part}.jsonl'``,
    where `manifest` is one of ``["recordings", "supervisions"]`` and ``part`` is specified by ``parts``.

    Args:
        num_mel_bins:
            The number of mel bins for Fbank.
        perturb_speed:
            Enable 0.9 and 1.1 speed perturbation for data augmentation. Default: False.
        whisper_fbank:
            Use WhisperFbank instead of Fbank. Default: False.
        output_dir:
            Output directory. Includes feats and its manifests.
        prefix:
            Prefix which can identify dataset name.
        parts:
            Comma or blank sep splits.
            e.g., ``"train,valid,test"``, ``"train valid test"``.
        src_suffix:
            Source manifest format.
            e.g, ``"jsonl"``, ``"jsonl.gz"``.
        tgt_suffix:
            Control out manifest format.
            e.g, ``"jsonl"``, ``"jsonl.gz"``.
    """
    src_dir = Path("data/manifests")
    output_dir = Path(output_dir)
    num_jobs = min(15, os.cpu_count())
    parts = parts or 'all'
    prefix = prefix or 'egs'
    dataset_parts = parts.replace(',', ' ').strip().split()
    manifests = read_manifests_if_cached(
        dataset_parts=dataset_parts,
        output_dir=src_dir,
        prefix=prefix,
        suffix=src_suffix,
    )
    assert manifests is not None

    assert len(manifests) == len(dataset_parts), (
        len(manifests),
        len(dataset_parts),
        list(manifests.keys()),
        dataset_parts,
    )
    if whisper_fbank:
        extractor = WhisperFbank(
            WhisperFbankConfig(num_filters=num_mel_bins, device=device)
        )
    else:
        extractor = Fbank(FbankConfig(num_mel_bins=num_mel_bins, device=device))

    for partition, m in manifests.items():

        if (output_dir / f"{prefix}_cuts_{partition}.{tgt_suffix}").is_file():
            logging.info(f"{partition} already exists - skipping.")
            continue
        logging.info(f"Processing {partition}")
        cut_set = CutSet.from_manifests(
            recordings=m["recordings"],
            supervisions=m["supervisions"],
        )

        if ('train' in partition or 'all' in partition) and perturb_speed:
            logging.info("Doing speed perturb")
            cut_set = cut_set + cut_set.perturb_speed(0.9) + cut_set.perturb_speed(1.1)
        cut_set = cut_set.compute_and_store_features(
            extractor=extractor,
            storage_path=f"{output_dir}/{prefix}_feats_{partition}",
            # when an executor is specified, make more partitions
            num_jobs=num_jobs,
            storage_type=LilcomChunkyWriter,
        )
        cut_set.to_file(output_dir / f"{prefix}_cuts_{partition}.{tgt_suffix}")


if __name__ == "__main__":
    formatter = "%(asctime)s %(levelname)s [%(filename)s:%(lineno)d] %(message)s"

    logging.basicConfig(format=formatter, level=logging.INFO)
    CLI(compute_fbank_subset)
