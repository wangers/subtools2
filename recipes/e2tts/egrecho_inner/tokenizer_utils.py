#!/usr/bin/env python3
# Copyright    2023                            (authors: Feiteng Li)
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

# (Author: Leo 2024-06)
# Modified from: https://github.com/lifeiteng/vall-e/blob/main/valle/data/tokenizer.py

import re
from dataclasses import asdict, dataclass
from typing import List, Optional, Tuple, Union

from egrecho.utils.text import (
    basic_cleaners,
    chinese_mandarin_cleaners,
    english_cleaners,
)

try:
    from pypinyin import Style, pinyin
    from pypinyin.style._utils import get_finals, get_initials
except Exception:
    pass

try:
    from piper_phonemize import phonemize_espeak
except Exception as ex:
    raise RuntimeError(
        f"{ex}\nPlease run\n"
        "pip install piper_phonemize -f https://k2-fsa.github.io/icefall/piper_phonemize.html"
    )

_DEFAULT_MARKS = ';:,.!?¡¿—…"«»“”'


@dataclass(frozen=True)
class Separator:
    phone: str = "|"
    syllable: str = "-"
    word: str = " "


class G2PModel:
    """Phonemize Text."""

    def __init__(
        self,
        language="en-us",
        backend="espeak",
        punctuation_marks=_DEFAULT_MARKS,
    ) -> None:
        if backend == "espeak":
            phonemizer = PiperEspeakBackend(
                language,
            )
        elif backend in ["pypinyin", "pypinyin_initials_finals"]:
            separator = Separator()
            if language not in {"cmn", "zh"}:
                raise ValueError(
                    f"{language} is not supported for pypinyin and pypinyin_initials_finals."
                )
            phonemizer = PypinyinBackend(
                backend=backend,
                punctuation_marks=punctuation_marks + separator.word,
                separator=separator,
            )
        else:
            raise NotImplementedError(f"{backend}")

        self.backend = phonemizer

    def __call__(
        self, text, language: Optional[Union[str, List[str]]] = None, **kwargs
    ) -> List[List[str]]:
        if isinstance(text, str):
            text = [text]

        phonemized = self.backend.phonemize(text, language)
        return phonemized


class PiperEspeakBackend:
    def __init__(
        self,
        language="en-us",
    ) -> None:
        self.language = language

    def phonemize(
        self,
        text: List[str],
        language: Optional[Union[str, List[str]]] = None,
        **kwargs,
    ) -> List[str]:
        assert isinstance(text, List)
        phonemized = []
        language = language or self.language
        for _text in text:
            _text = english_cleaners(_text)
            tokens_list = phonemize_espeak(_text, language)
            tokens = []
            for t in tokens_list:
                tokens.extend(t)
            phonemized.append(tokens)
        return phonemized

    @classmethod
    def support_langs(cls):
        return ("en-us",)


class PypinyinBackend:
    """PypinyinBackend for Chinese. Most codes is referenced from espnet.
    There are two types pinyin or initials_finals, one is
    just like "ni1 hao3", the other is like "n i1 h ao3".
    """

    def __init__(
        self,
        backend="pypinyin_initials_finals",
        punctuation_marks=_DEFAULT_MARKS,
        separator=None,
    ) -> None:
        self.backend = backend
        self.punctuation_marks = punctuation_marks
        self.separator = separator or Separator()

    def phonemize(self, text: List[str], separator=None, **kwargs) -> List[str]:
        assert isinstance(text, List)
        phonemized = []
        separator = separator or self.separator
        for _text in text:
            _text = basic_cleaners(_text)
            _text = chinese_mandarin_cleaners(_text)
            _text = _text.replace(" ", separator.word)
            phones = []
            if self.backend == "pypinyin":
                for n, py in enumerate(
                    pinyin(_text, style=Style.TONE3, neutral_tone_with_five=True)
                ):
                    if all([c in self.punctuation_marks for c in py[0]]):
                        if len(phones):
                            assert phones[-1] == separator.syllable
                            phones.pop(-1)

                        phones.extend(list(py[0]))
                    else:
                        phones.extend([py[0], separator.syllable])
            elif self.backend == "pypinyin_initials_finals":
                for n, py in enumerate(
                    pinyin(_text, style=Style.TONE3, neutral_tone_with_five=True)
                ):
                    if all([c in self.punctuation_marks for c in py[0]]):
                        if len(phones):
                            assert phones[-1] == separator.syllable
                            phones.pop(-1)
                        phones.extend(list(py[0]))
                    else:
                        if py[0][-1].isalnum():
                            initial = get_initials(py[0], strict=False)
                            if py[0][-1].isdigit():
                                final = get_finals(py[0][:-1], strict=False) + py[0][-1]
                            else:
                                final = get_finals(py[0], strict=False)
                            phones.extend(
                                [
                                    initial,
                                    separator.phone,
                                    final,
                                    separator.syllable,
                                ]
                            )
                        else:
                            assert ValueError
            else:
                raise NotImplementedError

            tokens = py_to_list(
                "".join(phones).rstrip(f"{separator.word}{separator.syllable}"),
                separator,
            )
            phonemized.append(tokens)
        return phonemized


def py_to_list(phonemized: str, separator=None) -> List[str]:

    separator = separator or Separator()
    fields = []
    for word in phonemized.split(separator.word):

        pp = re.findall(r"\w+|[^\w\s]", word, re.UNICODE)
        fields.extend([p for p in pp if p != separator.phone] + [" "])
    assert len("".join(fields[:-1])) == len(phonemized) - phonemized.count(
        separator.phone
    )
    return fields[:-1]


if __name__ == "__main__":

    g2p = G2PModel(backend="pypinyin_initials_finals", language="zh")
    pybk = PypinyinBackend()
    cn = ["我爱 你 中国"]
    pyrs = pybk.phonemize(cn)
    pyrs1 = g2p(cn)
    esp = phonemize_espeak(
        "KNOT one point one five miles per hour To get up and running quickly just follow the steps below.",
        "en-us",
    )
    print(pyrs)
    print(pyrs1)
    print(esp)
    print(len(esp[0]))
