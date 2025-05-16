# -*- encoding: utf-8 -*-
# Copyright xmuspeech (Author: Leo 2024-08)

import re
from dataclasses import dataclass, field

from egrecho.utils.misc import pprint2str

__all__ = sorted(
    [
        "Provider",
        "ProviderInfo",
        "HFProviderInfo",
        'OpenAIWhisperProviderInfo',
    ]
)


@dataclass
class Provider:
    name: str
    url: str

    def __str__(self):
        return f"{self.name} ({self.url})"


_HUGGINGFACE = Provider(
    "Hugging Face/transformers", "https://github.com/huggingface/transformers"
)
_OPENAIWHISPER = Provider("OpenAI/whisper", "https://github.com/openai/whisper")


@dataclass
class ProviderInfo:
    provider: Provider = None

    def __repr__(self):
        base = self.__class__.__name__
        d = {k: v for k, v in self.__dict__.items() if v is not None}
        if prov_str := (d.pop("provider", self.provider)):
            d["provider"] = str(prov_str)

        repr = pprint2str(d, sort_dicts=False).strip()
        repr = re.sub(r"^", " " * 4, repr, 0, re.M)
        return f"{base}(\n{repr}\n)"


# set repr=False to avoid dataclasses overwrite __repr__
@dataclass(repr=False)
class HFProviderInfo(ProviderInfo):
    provider: Provider = field(default=_HUGGINGFACE, init=False)
    repo: str = None
    description: str = None


@dataclass(repr=False)
class OpenAIWhisperProviderInfo(ProviderInfo):
    provider: Provider = field(default=_OPENAIWHISPER, init=False)
    repo: str = None
    description: str = None
