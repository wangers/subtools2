# (Author: Leo 2024-06-04)
"""
convert kaldi (multilingual) asr dir to manifest.
"""

import logging
from collections import defaultdict
from pathlib import Path
from typing import Dict, Optional, Union

from jsonargparse import CLI
from lhotse import validate_recordings_and_supervisions
from lhotse.audio import Recording, RecordingSet
from lhotse.qa import fix_manifests
from lhotse.supervision import SupervisionSegment, SupervisionSet
from lhotse.utils import Pathlike
from tqdm.auto import tqdm


def text_normalize(line: str):
    """
    sed 's/ａ/a/g' | sed 's/ｂ/b/g' |\
    sed 's/ｃ/c/g' | sed 's/ｋ/k/g' |\
    sed 's/ｔ/t/g' > $dir/transcripts.t

    """
    line = line.replace("ａ", "a")
    line = line.replace("ｂ", "b")
    line = line.replace("ｃ", "c")
    line = line.replace("ｋ", "k")
    line = line.replace("ｔ", "t")
    line = line.replace("", "")
    line = line.replace("！", "")
    line = line.replace("，", "")
    line = line.replace("０", "0")
    line = line.replace("１", "1")
    line = line.replace("２", "2")
    line = line.replace("３", "3")
    line = line.replace("４", "4")
    line = line.replace("５", "5")
    line = line.replace("８", "8")
    line = line.replace("９", "9")
    line = line.replace("；", "")
    line = line.replace("？", "")
    line = line.replace("%", "")
    line = line.replace("(", "")
    line = line.replace(")", "")
    line = line.replace("-", "")
    line = line.replace(":", "")
    line = line.replace("*", "")
    # line = line.upper()
    return line


def prepare_multi_lang_from_kaldi(
    corpus_dir: Pathlike,
    output_dir: Optional[Pathlike] = None,
    prefix: str = '',
    part='all',
) -> Dict[str, Dict[str, Union[RecordingSet, SupervisionSet]]]:
    """
    Returns the manifests which consist of the Recordings and Supervisions.

    Cares files as below:

    - wav.scp
    - utt2lang
    - text

    :param corpus_dir: Pathlike, the path of the data dir.
    :param output_dir: Pathlike, the path where to write the manifests. e.g., {output_dir}/{prefix}_"supervisions"_{part}.jsonl.
    :param prefix: str, the prefix name of manifests.
    :param part: str, the split name of manifests.
    :return: a Dict whose key is the dataset part, and the value is Dicts with the keys 'recordings' and 'supervisions'.
    """
    corpus_dir = Path(corpus_dir)
    part = part or 'all'
    prefix = prefix or 'egs'

    assert corpus_dir.is_dir(), f"No such directory: {corpus_dir}"
    if output_dir is not None:
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

    transcript_path = corpus_dir / "text"
    transcript_dict = {}
    with open(transcript_path, "r", encoding="utf-8") as f:
        for line in f.readlines():
            idx_transcript = line.split()
            content = " ".join(idx_transcript[1:])
            content = text_normalize(content)
            transcript_dict[idx_transcript[0]] = content

    wav2lang = corpus_dir / "utt2lang"
    lang_dict = {}
    with open(wav2lang, "r", encoding="utf-8") as f:
        for line in f.readlines():
            uid, lang = line.split()
            lang_dict[uid] = lang

    manifests = defaultdict(dict)

    logging.info(f"Process {corpus_dir}, it takes about ? seconds.")

    # Generate a mapping: utt_id -> (audio_path, audio_info, speaker, text)
    recordings = []
    supervisions = []
    wav_scp = corpus_dir / "wav.scp"

    nospace = {
        "zh",
        "ja",
        "th",
        "lo",
        "my",
        "yue",
        "chinese",
        "japanese",
        "thai",
        "lao",
        "myanmar",
        "cantonese",
    }
    nospace_olr = {"minnan", "sichuan", "shanghai"}  # olr special
    nospace |= nospace_olr

    with open(wav_scp, "r", encoding="utf-8") as f:
        lines = f.readlines()
        logging.info(len(lines))
        for line in tqdm(lines, desc='Parse utts...'):
            uid, audio_path = line.split()

            audio_path = Path(audio_path)

            if uid not in transcript_dict:
                logging.warning(f"No transcript: {uid}")
                logging.warning(f"{audio_path} has no transcript.")
                continue

            if uid not in lang_dict:
                lang = "zh"
                # logging.warning(f"No lang_id: {uid}")
                # logging.warning(f"{audio_path} has no transcript.")
                # continue
            else:
                lang = lang_dict[uid]  # "Chinese"

            text = transcript_dict[uid].strip()

            if lang.lower() in nospace:
                text = text.replace(" ", "")

            if not audio_path.is_file():
                logging.warning(f"No such file: {audio_path}")
                continue

            recording = Recording.from_file(audio_path, recording_id=uid)

            segment = SupervisionSegment(
                id=uid,
                recording_id=uid,
                start=0.0,
                duration=recording.duration,
                channel=0,
                language=lang,
                text=text,
            )

            recordings.append(recording)
            supervisions.append(segment)
            if recording.duration > 30.0:
                logging.warning(
                    f"The audio duration is too long: {recording.id,recording.duration}"
                )

        recording_set = RecordingSet.from_recordings(recordings)
        supervision_set = SupervisionSet.from_segments(supervisions)
        recording_set, supervision_set = fix_manifests(recording_set, supervision_set)
        validate_recordings_and_supervisions(recording_set, supervision_set)

        if output_dir is not None:
            prefix = prefix + '_' if prefix and not prefix.endswith("_") else ''
            supervision_set.to_file(output_dir / f"{prefix}supervisions_{part}.jsonl")
            recording_set.to_file(output_dir / f"{prefix}recordings_{part}.jsonl")

        manifests = {"recordings": recording_set, "supervisions": supervision_set}

    return manifests


if __name__ == "__main__":
    formatter = "%(asctime)s %(levelname)s [%(filename)s:%(lineno)d] %(message)s"

    logging.basicConfig(format=formatter, level=logging.INFO)
    CLI(prepare_multi_lang_from_kaldi)
