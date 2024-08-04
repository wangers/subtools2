# (Author: Leo 2024-06)
"""
Extract encodec codes and phoneme tokens.
"""
import logging
import os
from pathlib import Path
from typing import Literal, Optional

import torch
from jsonargparse import CLI
from lhotse import CutSet, NumpyHdf5Writer, load_manifest
from lhotse.recipes.utils import read_manifests_if_cached
from tokenizer_utils import EncodecTokeConfig, EncodecTokenExtractor, G2PModel
from tqdm.auto import tqdm

# Torch's multithreaded behavior needs to be disabled or
# it wastes a lot of CPU and slow things down.
# Do this outside of main() in case it needs to take effect
# even when we are not invoking the main (e.g. when spawning subprocesses).
torch.set_num_threads(1)
torch.set_num_interop_threads(1)


def prepare_encodec(
    outdir: str = "data/libritts/codes24k",
    srcdir: str = "data/libritts/manifests",
    mvdir: Optional[str] = None,
    prefix: str = "egs",
    parts: str = "train",
    suffix="jsonl.gz",
    batch_duration: float = 400.0,
    device: str = "auto",
):
    src_dir = Path(srcdir)
    outdir: Path = Path(outdir)
    outdir.mkdir(exist_ok=True, parents=True)
    if mvdir is not None and Path(mvdir).resolve() != outdir.resolve():
        mvdir = Path(mvdir)
        mvdir.mkdir(exist_ok=True, parents=True)
    num_jobs = min(16, os.cpu_count())
    parts = parts or "all"

    parts = parts.replace(",", " ").strip().split()

    manifests = read_manifests_if_cached(
        dataset_parts=parts,
        output_dir=src_dir,
        prefix=prefix,
        suffix=suffix,
    )

    assert len(manifests) == len(parts), (parts, manifests)

    extractor = EncodecTokenExtractor(EncodecTokeConfig(device=device))
    device = extractor.device
    for partition, m in manifests.items():

        logging.info(f"Extracting codes {partition}")

        cut_set = CutSet.from_manifests(
            recordings=m["recordings"],
            supervisions=m["supervisions"],
        )
        cut_set = cut_set.resample(24000)
        with torch.no_grad():
            if device != "cpu":

                cut_set = cut_set.compute_and_store_features_batch(
                    extractor=extractor,
                    storage_path=f"{outdir}/{prefix}_encodec_{partition}",
                    num_workers=num_jobs,
                    batch_duration=batch_duration,
                    collate=False,
                    overwrite=True,
                    storage_type=NumpyHdf5Writer,
                )
            else:
                cut_set = cut_set.compute_and_store_features(
                    extractor=extractor,
                    storage_path=f"{outdir}/{prefix}_encodec_{partition}",
                    num_jobs=num_jobs,
                    storage_type=NumpyHdf5Writer,
                )
            cut_set.to_file(outdir / f"{prefix}_cuts_{partition}.{suffix}")
            if mvdir is not None:
                cut_set.to_file(mvdir / f"{prefix}_cuts_{partition}.{suffix}")
        logging.info(f"Extracting codes {partition} Done.")


def prepare_tokens(
    outdir: Optional[str] = "data/libritts/codes24k",
    mvdir: Optional[str] = "data/libritts/tokenized",
    prefix: str = "egs",
    parts: str = "train",
    suffix="jsonl.gz",
    language="en-us",
    backend: Literal["pypinyin_initials_finals", "espeak", "pypinyin"] = 'espeak',
):

    outdir = outdir or Path("data/codes24k")
    outdir = Path(outdir)
    if mvdir is not None:
        if Path(mvdir).resolve() != outdir.resolve():
            mvdir = Path(mvdir)
            mvdir.mkdir(exist_ok=True, parents=True)
        else:
            raise ValueError(
                f'Ilegal move src <{outdir}> to itself {outdir}, [HINT] fix ``mvdir``.'
            )

    parts = parts or "all"
    parts = parts.replace(",", " ").strip().split()

    cuts_manis = {}
    for part in parts:
        cuts_manis[part] = load_manifest(outdir / f"{prefix}_cuts_{part}.{suffix}")

    g2p = G2PModel(language=language, backend=backend)
    for part, cuts in cuts_manis.items():
        logging.info(f"Tokenize phoneme codes {part}")
        new_cuts = []
        for cut in tqdm(cuts):
            # Each cut only contains one supervision
            assert len(cut.supervisions) == 1, (len(cut.supervisions), cut)
            text = cut.supervisions[0].text
            # Convert to phonemes
            tokens_list = g2p(text)
            tokens = []
            for t in tokens_list:
                tokens.extend(t)
            cut.tokens = tokens
            new_cuts.append(cut)

        new_cut_set = CutSet.from_cuts(new_cuts)
        new_cut_set.to_file(outdir / f"{prefix}_cuts_with_tokens_{part}.{suffix}")
        if mvdir is not None:
            new_cut_set.to_file(mvdir / f"{prefix}_cuts_{part}.{suffix}")


if __name__ == "__main__":
    formatter = "%(asctime)s %(levelname)s [%(filename)s:%(lineno)d] %(message)s"
    logging.basicConfig(format=formatter, level=logging.INFO)

    subcommands = {
        "encodec": prepare_encodec,
        "tokenize": prepare_tokens,
    }
    CLI(subcommands)
