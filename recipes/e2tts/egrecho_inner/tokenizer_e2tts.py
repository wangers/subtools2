# (Author: Leo 2024-06)

from __future__ import annotations

import warnings
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

from tokenizer_utils import G2PModel
from utils import save_array

from egrecho.core.tokenizer import (
    BaseTokenizerConfig,
    TextInput,
    Tokenizer,
    TruncationStrategy,
)
from egrecho.utils.logging import get_logger
from egrecho.utils.types import PaddingStrategy

logger = get_logger(__name__)
VOCAB_FILES_NAMES = {"vocab_file": "tokens.txt"}


@dataclass
class E2TTSTokenizerConfig(BaseTokenizerConfig):
    language: str = "en-us"
    backend: str = "espeak"

    @property
    def extra_files_names(self):
        """
        Defines the extra file names required by the model. Can be either:

        - A dictionary with values being the filenames for saving the associated files (strings).
        - A tuple/list of filenames.
        """
        return VOCAB_FILES_NAMES


class E2TTSTokenizer(Tokenizer):
    """E2TTS phonemize tokenizer."""

    CONFIG_CLS = E2TTSTokenizerConfig
    model_input_names = ["input_ids", "attention_mask"]

    def __init__(self, config: E2TTSTokenizerConfig):
        super().__init__(config)
        self.config: E2TTSTokenizerConfig
        self.g2p = G2PModel(self.config.language, backend=self.config.backend)
        vocab_files = self.config.get_extra_files()

        self.token2id: Dict[str, int] = {}
        with open(vocab_files["vocab_file"], "r", encoding="utf-8") as f:
            for line in f.readlines():
                info = line.rstrip().split()
                if len(info) == 1:
                    # case of space
                    token = " "
                    id = int(info[0])
                else:
                    token, id = info[0], int(info[1])
                assert token not in self.token2id, token
                self.token2id[token] = id

        # Refer to https://github.com/rhasspy/piper/blob/master/TRAINING.md
        self.pad_id = self.token2id["_"]  # padding
        self.bos_id = self.token2id["^"]  # beginning of an utterance (bos)
        self.eos_id = self.token2id["$"]  # end of an utterance (eos)
        self.space_id = self.token2id[" "]  # word separator (whitespace)

    @property
    def vocab_size(self) -> int:
        return len(self.token2id)

    def save_extras(self, savedir: str, filename_prefix=None) -> Tuple[str]:
        self.default_save_extras(savedir, filename_prefix)

    def text2tokens(self, line: str) -> List[str]:
        return self.g2p(line)[0]

    def tokens2ids(self, tokens) -> List[int]:
        token_ids = []
        for t in tokens:
            if t not in self.token2id:
                warnings.warn(f"Skip OOV {t}")
                continue
            token_ids.append(self.token2id[t])
        return token_ids

    def __call__(
        self,
        text,
        add_special_tokens: bool = True,
        padding: bool | TextInput | PaddingStrategy = True,
        truncation: bool | TextInput | TruncationStrategy = None,
        max_length: int | None = None,
        is_split_into_words: bool = False,
        **kwargs,
    ):

        is_batched = self.input_text_batched(
            text=text, text_pair=None, is_split_into_words=is_split_into_words
        )
        (
            padding_strategy,
            truncation_strategy,
            max_length,
            kwargs,
        ) = self._get_padding_truncation_strategies(
            padding=padding,
            truncation=truncation,
            max_length=max_length,
            **kwargs,
        )

        if is_batched:
            input_ids = []
            for t in text:
                first_ids = self.get_input_ids(
                    t, is_split_into_words=is_split_into_words
                )
                input_ids.append((first_ids, None))

            batch_outputs = self._batch_prepare_for_model(
                input_ids,
                add_special_tokens=add_special_tokens,
                padding_strategy=padding_strategy,
                truncation_strategy=truncation_strategy,
                max_length=max_length,
                **kwargs,
            )
            return batch_outputs
        else:
            first_ids = self.get_input_ids(
                text, is_split_into_words=is_split_into_words
            )

            return self.prepare_for_model(
                first_ids,
                pair_ids=None,
                add_special_tokens=add_special_tokens,
                padding=padding_strategy,
                prepend_batch_axis=True,
                truncation=truncation_strategy,
                max_length=max_length,
                **kwargs,
            )

    def build_inputs_with_special_tokens(
        self, token_ids_0: List[int], token_ids_1: Optional[List[int]] = None
    ) -> List[int]:
        """
        Build model inputs from a sequence or a pair of sequence for sequence classification tasks by concatenating and
        adding special tokens.

        This implementation does not add special tokens and this method should be overridden in a subclass.

        Args:
            token_ids_0 (``List[int]``): The first tokenized sequence.
            token_ids_1 (``List[int]``, *optional*): The second tokenized sequence.

        Returns:
            ``List[int]``: The model input with special tokens.
        """
        if token_ids_1 is None:
            return [self.bos_id] + token_ids_0 + [self.eos_id]
        # We don't expect to process pairs, but leave the pair logic for API consistency
        return [self.bos_id] + token_ids_0 + token_ids_1 + [self.eos_id]
