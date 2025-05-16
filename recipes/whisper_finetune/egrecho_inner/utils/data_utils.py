# (Author: Leo 2024-06)

import re
from dataclasses import dataclass
from typing import Any, Dict, List, Union

import torch
from transformers.models.whisper.english_normalizer import (
    BasicTextNormalizer,
    EnglishTextNormalizer,
)
from transformers.models.whisper.tokenization_whisper import TO_LANGUAGE_CODE
from zhconv import convert

CN_LANGS = ['shanghai', 'sichuan', 'minnan', 'mandarin', 'zh', 'yue']


# 删除标点符号
def remove_punctuation(text: Union[str, List[str]]):
    punctuation = '!,.;:?、！，。；：？'
    if isinstance(text, str):
        text = re.sub(r'[{}]+'.format(punctuation), '', text).strip()
        return text
    elif isinstance(text, list):
        result_text = []
        for t in text:
            t = re.sub(r'[{}]+'.format(punctuation), '', t).strip()
            result_text.append(t)
        return result_text
    else:
        raise Exception(f'不支持该类型{type(text)}')


# 将繁体中文总成简体中文
def to_simple(text: Union[str, List[str]]):
    if isinstance(text, str):
        text = convert(text, 'zh-cn')
        return text
    elif isinstance(text, list):
        result_text = []
        for t in text:
            t = convert(t, 'zh-cn')
            result_text.append(t)
        return result_text
    else:
        raise Exception(f'不支持该类型{type(text)}')


class XNormlizer:
    def __init__(self, tokenizer=None) -> None:
        english_spelling_normalizer = (
            tokenizer.english_spelling_normalizer if tokenizer else {}
        )
        self.en_normalizer = EnglishTextNormalizer(english_spelling_normalizer)
        self.basic_normalizer = BasicTextNormalizer()

    def __call__(self, text: Union[str, List[str]], language: Union[str, List[str]]):
        """Auto normalize text according to provided language."""
        if text_is_str := isinstance(text, str):
            text = [text]
        if isinstance(language, str):
            language = [language]

        language = [self._get_lang(lang) for lang in language]
        if not (isinstance(text, (list, tuple)) and isinstance(text, (list, tuple))):
            raise ValueError(
                f'Intut text and language shuold be type of str of list of str(s), but got {type(text)}, {type(language)}.'
            )
        text, language = list(text), list(language)
        if len(text) != len(language) and len(language) == 1:
            language = language * len(text)
        elif len(text) != len(language):
            raise ValueError(
                f'Dismatch text and language num for {len(text)}, {len(language)}.'
            )
        normlized_text = []
        for txt, lang in zip(text, language):
            if lang in CN_LANGS:
                txt = remove_punctuation(txt)
                normlized_text.append(to_simple(txt))
            elif lang == 'en':
                normlized_text.append(self.en_normalizer(txt))
            else:
                normlized_text.append(self.basic_normalizer(txt))

        return normlized_text[0] if text_is_str else normlized_text

    @staticmethod
    def _get_lang(lang: str):
        if lang is not None:
            lang = lang.lower()
            if lang in TO_LANGUAGE_CODE:
                language_id = TO_LANGUAGE_CODE[lang]
            elif lang in TO_LANGUAGE_CODE.values():
                language_id = lang
        language_id = lang
        return language_id


@dataclass
class DataCollatorSpeechSeq2SeqWithPadding:
    processor: Any

    def __call__(
        self, features: List[Dict[str, Union[List[int], torch.Tensor]]]
    ) -> Dict[str, torch.Tensor]:
        # split inputs and labels since they have to be of different lengths and need different padding methods
        # first treat the audio inputs by simply returning torch tensors
        input_features = [
            {"input_features": feature["input_features"][0]} for feature in features
        ]
        batch = self.processor.feature_extractor.pad(
            input_features, return_tensors="pt"
        )

        # get the tokenized label sequences
        label_features = [{"input_ids": feature["labels"]} for feature in features]
        # pad the labels to max length
        labels_batch = self.processor.tokenizer.pad(label_features, return_tensors="pt")

        # replace padding with -100 to ignore loss correctly
        labels = labels_batch["input_ids"].masked_fill(
            labels_batch.attention_mask.ne(1), -100
        )

        # if bos token is appended in previous tokenization step,
        # cut bos token here as it's append later anyways
        if (labels[:, 0] == self.processor.tokenizer.bos_token_id).all().cpu().item():
            labels = labels[:, 1:]

        batch["labels"] = labels

        return batch
