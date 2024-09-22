#!/usr/bin/env python3
# Copyright    2023  Xiaomi Corp.        (authors: Yifan Yang)
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


# (Author: Leo 2024-08)

import logging
import re
from pathlib import Path
from pprint import pprint

from jsonargparse import CLI
from lhotse import CutSet
from lhotse.recipes.utils import read_manifests_if_cached


def normalize_text(utt: str, language: str = '') -> str:
    utt = re.sub(r"[{0}]+".format("-"), " ", utt)
    utt = re.sub("’", "'", utt)
    utt = re.sub(r"\s+", " ", utt)
    utt = re.sub(r"\s+\.\s+", ".", utt)
    # utt = re.sub(r"[\.\,\?\:\-!;()«»…\]\[/\*–‽+&_\\½√>€™$•¼}{~—=“\"”″‟„]", "", utt)
    return utt


def preprocess_cuts_simple(
    outdir: str = "data/libritts/simplecuts_tts",
    srcdir: str = "data/libritts/manifests",
    prefix: str = "egs",
    parts: str = "train",
):
    """Simple cuts"""
    src_dir = Path(srcdir)
    outdir: Path = Path(outdir)
    outdir.mkdir(exist_ok=True, parents=True)

    parts = parts or "all"

    parts = parts.replace(",", " ").strip().split()

    pprint(locals())
    logging.info("Loading manifest")

    suffix = "jsonl.gz"
    manifests = read_manifests_if_cached(
        dataset_parts=parts,
        output_dir=src_dir,
        suffix=suffix,
        prefix=prefix,
    )
    assert manifests is not None

    assert len(manifests) == len(parts), (
        len(manifests),
        len(parts),
        list(manifests.keys()),
        parts,
    )

    for partition, m in manifests.items():
        logging.info(f"Processing {partition}")
        raw_cuts_path = outdir / f"{prefix}_cuts_{partition}.{suffix}"
        if raw_cuts_path.is_file():
            logging.info(f"{partition} already exists - skipping")
            continue

        logging.info(f"Normalizing text in {partition}")
        for sup in m["supervisions"]:
            if sup.has_custom('normalized_text'):
                orig_text = str(sup.normalized_text)
            else:
                orig_text = str(sup.text)
            normalized_text = normalize_text(orig_text)
            sup.normalized_text = normalized_text
            if len(orig_text) != len(normalized_text):
                logging.info(
                    f"\nOriginal text vs normalized text:\n{orig_text}\n{normalized_text}"
                )

        # Create long-recording cut manifests.
        cut_set = CutSet.from_manifests(
            recordings=m["recordings"],
            supervisions=m["supervisions"],
        )
        # Run data augmentation that needs to be done in the
        # time domain.
        logging.info(f"Saving to {raw_cuts_path}")
        cut_set.to_file(raw_cuts_path)


if __name__ == "__main__":
    formatter = "%(asctime)s %(levelname)s [%(filename)s:%(lineno)d] %(message)s"
    logging.basicConfig(format=formatter, level=logging.INFO)

    CLI(preprocess_cuts_simple)
    logging.info("Done")
