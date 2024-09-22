# (Author: Leo 2024-06)

import json
import logging
import shutil
from collections import defaultdict
from pathlib import Path
from typing import List, Optional, Union

from jsonargparse import CLI
from lhotse import CutSet
from tqdm.auto import tqdm


def libritts_prompt_target(
    path_or_paths: Union[str, List[str]] = "./exp/egs/libritts/cuts_test.jsonl.gz",
    outdir='./data/prompt_test/libritts',
    max_prompt_length=3.1,
    min_prompt_length=2.0,
    max_sample_len=10,
    min_sample_len=4,
    overwrite: bool = True,
    nj: Optional[int] = None,
):
    """libritts"""
    out_dir = Path(outdir)
    if out_dir.is_dir():
        if overwrite:
            logging.warning(f"Force clearing ({out_dir}) ...")
            shutil.rmtree(out_dir)
        else:
            try:
                next(out_dir.iterdir())
            except StopIteration:
                pass
            else:
                raise ValueError(
                    f"{out_dir} already exists, clean it or set overwrite=True."
                )
    out_dir.mkdir(parents=True, exist_ok=True)
    # spkid-chapter
    # 909_131041_000013_000002
    # 909_131041_000013_000003
    if isinstance(path_or_paths, str):
        path_or_paths = [path_or_paths]
    cuts = CutSet.from_files(path_or_paths).trim_to_supervisions(keep_overlapping=False)
    speaker2utts = defaultdict(lambda: [])

    utt2cut = {}

    for cut in cuts:

        speaker = cut.supervisions[0].speaker
        speaker2utts[speaker].append(cut.id)
        utt2cut[cut.id] = cut
    spk2validcuts = {}
    spk_prompts_pairs = defaultdict(lambda: [])

    for spk in speaker2utts:
        uttids = sorted(speaker2utts[spk])
        lens = len(uttids)
        valid_utts = set()
        spk_utt2cut = {}

        for cur in range(1, lens):
            if (
                prev_dur := utt2cut[uttids[cur - 1]].supervisions[0].duration
            ) < max_prompt_length and prev_dur > min_prompt_length:
                if (
                    min_sample_len
                    <= utt2cut[uttids[cur]].supervisions[0].duration
                    <= max_sample_len
                ):
                    spk_prompts_pairs[spk].append([uttids[cur - 1], uttids[cur]])
                    valid_utts.add(uttids[cur - 1])
                    valid_utts.add(uttids[cur])
        valid_cuts = CutSet(list(utt2cut[cut_id] for cut_id in valid_utts))
        spk2validcuts[spk] = valid_cuts

    prompts_pair_cuts = ([], [])
    for spk_, valid_cuts_ in tqdm(spk2validcuts.items(), desc='Process spks...'):
        new_valid_cuts = valid_cuts_.save_audios(
            out_dir / 'audio' / spk_, num_jobs=nj, progress_bar=False
        )
        for cut in new_valid_cuts:
            spk_utt2cut[cut.id] = cut
        for pro, tgt in spk_prompts_pairs[spk_]:
            prompts_pair_cuts[0].append(spk_utt2cut[pro])
            prompts_pair_cuts[1].append(spk_utt2cut[tgt])
    cuts_pro = CutSet(prompts_pair_cuts[0])
    cuts_tgt = CutSet(prompts_pair_cuts[1])
    cuts_pro.to_file(out_dir / 'prompt_cuts.jsonl')
    cuts_tgt.to_file(out_dir / 'target_cuts.jsonl')
    with open(out_dir / 'egs_test_prompt.jsonl', 'w') as fw:
        for cut_pro, cut_tgt in zip(prompts_pair_cuts[0], prompts_pair_cuts[1]):
            item = {}
            item['id'] = cut_tgt.id
            item['prompt_audio'] = cut_pro.recording.sources[0].source
            item['prompt_text'] = cut_pro.supervisions[0].text
            item['audio'] = cut_tgt.recording.sources[0].source
            item['text'] = cut_tgt.supervisions[0].text
            print(json.dumps(item, ensure_ascii=False), file=fw)
    logging.info(
        f"Done ({out_dir}) (prompt_cuts.jsonl|target_cuts.jsonl|egs_test_prompt.jsonl) "
    )


if __name__ == "__main__":
    formatter = "%(asctime)s %(levelname)s [%(filename)s:%(lineno)d] %(message)s"
    logging.basicConfig(format=formatter, level=logging.INFO)
    subcommands = {
        "libritts": libritts_prompt_target,
    }
    CLI(subcommands)
