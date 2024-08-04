# (Author: Leo 2024-06)
import logging
from itertools import chain
from pathlib import Path
from typing import Optional

from jsonargparse import CLI
from tqdm.auto import tqdm

from egrecho.data.datasets.audio.samples import ASVSamples, AudioDew
from egrecho.data.dew import DewSamples
from egrecho.data.processors import unbatch


def prepare_acoustic_trials(
    tts_egfile: str = 'data/infer/libritts/valle/test/egs_infer_tts.jsonl',
    outdir: Optional[str] = None,
):

    outdir = Path(tts_egfile).parent if outdir is None else Path(outdir)
    dew_egs = DewSamples.from_files(tts_egfile)

    dews = []
    prompt_dews = []
    sv_prompt_trials = []
    for eg in tqdm(dew_egs, desc='Formate egs', leave=True):

        dew = eg
        meta = dew.pop('meta')
        text = meta.pop('text')
        ref = meta.get('ref_audio', None)
        ref_item = {'ref_audio': ref} if ref else {}
        extras = {
            'prompt_text': meta['prompt_text'],
            'prompt_audio': meta['prompt_audio'],
            **ref_item,
        }
        tts_id = eg.id
        prompt_id = f'prompt-{tts_id}'
        dews.append(
            AudioDew(
                id=tts_id,
                audio_path=eg.hyp_audio,
                text=text,
                extras=extras,
            )
        )
        prompt_dews.append(
            AudioDew(
                id=prompt_id,
                audio_path=meta['prompt_audio'],
                text=meta['prompt_text'],
            )
        )

        sv_prompt_trials.append([prompt_id, tts_id])

    if sv_prompt_trials:
        with open(outdir / 'sv_prompt_trials.txt', 'w', encoding='utf-8') as ftrial:
            for prompt, hyp in sv_prompt_trials:
                print(prompt, hyp, file=ftrial)
    dewsets = ASVSamples.from_dews(dews)
    dewsets.to_file(outdir / 'egs_tts_metric.jsonl')

    ASVSamples.from_dews(chain.from_iterable(zip(prompt_dews, dews))).to_file(
        outdir / 'egs_prompt_tts.jsonl'
    )


if __name__ == "__main__":
    formatter = "%(asctime)s %(levelname)s [%(filename)s:%(lineno)d] %(message)s"
    logging.basicConfig(format=formatter, level=logging.INFO)
    CLI(prepare_acoustic_trials)
