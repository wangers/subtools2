# -*- coding:utf-8 -*-
# Copyright xmuspeech (Author: Leo 2024-09)

"""
HuggingFace Auto tokenizer. Modified from `Nemo
<https://docs.nvidia.com/nemo-framework/user-guide/latest/nemotoolkit/common/tokenizers.html?highlight=autotokenizer#nemo.collections.common.tokenizers.AutoTokenizer>`_.
"""

from __future__ import annotations

from contextlib import contextmanager
from dataclasses import dataclass, field
from typing import List, Optional, Tuple, Union

import numpy as np
import torch

from egrecho.core.tokenizer import BaseTokenizer, BaseTokenizerConfig
from egrecho.utils.common import alt_none
from egrecho.utils.imports import _TRANSFORMERS_AVAILABLE
from egrecho.utils.logging import get_logger

if not _TRANSFORMERS_AVAILABLE:
    raise ImportError(
        "To use hf tokenizer,  please ``pip install transformers`` first."
    )

from transformers import AutoTokenizer as AUTOTOKENIZER
from transformers.tokenization_utils_base import (
    ENCODE_KWARGS_DOCSTRING,
    ENCODE_PLUS_ADDITIONAL_KWARGS_DOCSTRING,
    BatchEncoding,
    PaddingStrategy,
    PreTokenizedInput,
    TensorType,
    TextInput,
    TruncationStrategy,
    add_end_docstrings,
)

logger = get_logger()

__all__ = [
    'AutoTokenizer',
    'AutoTokenizerConfig',
]


@dataclass
class AutoTokenizerConfig(BaseTokenizerConfig):
    """
    Args:
        pretrained_model_name: corresponds to HuggingFace-AutoTokenizer's 'pretrained_model_name_or_path' input argument.
            For more details please refer to https://huggingface.co/transformers/_modules/transformers/tokenization_auto.html#AutoTokenizer.from_pretrained.
            The list of all supported models can be found here: ALL_PRETRAINED_CONFIG_ARCHIVE_MAP
        vocab_file: path to file with vocabulary which consists
            of characters separated by '\n'.
        mask_token: mask token
        bos_token: the beginning of sequence token
        eos_token: the end of sequence token. Usually equal to sep_token
        pad_token: token to use for padding
        sep_token: token used for separating sequences
        cls_token: class token. Usually equal to bos_token
        unk_token: token to use for unknown tokens
        additional_special_tokens: list of other tokens beside standard special tokens (bos, eos, pad, etc.).
            For example, sentinel tokens for T5 (<extra_id_0>, <extra_id_1>, etc.)
        replace_additional_special_tokens (`bool`, *optional*,, defaults to `True`): If `True`,
            the existing list of additional special tokens will be replaced by the list provided in
            `special_tokens_dict`. Otherwise, `self.additional_special_tokens` is just extended. In the former
            case, the tokens will NOT be removed from the tokenizer's full vocabulary - they are only being flagged
            as non-special tokens. Remember, this only affects which tokens are skipped during decoding, not the
            `added_tokens_encoder` and `added_tokens_decoder`. This means that the previous
            `additional_special_tokens` are still added tokens, and will not be split by the model.

        use_fast: whether to use fast HuggingFace tokenizer
    """

    pretrained_model_name: Optional[str] = None
    vocab_file: Optional[str] = None
    merges_file: Optional[str] = None
    mask_token: Optional[str] = None
    bos_token: Optional[str] = None
    eos_token: Optional[str] = None
    pad_token: Optional[str] = None
    sep_token: Optional[str] = None
    cls_token: Optional[str] = None
    unk_token: Optional[str] = None
    use_fast: Optional[bool] = False
    additional_special_tokens: List = field(default_factory=list)
    replace_additional_special_tokens: bool = True
    save_hf: bool = False

    def __post_init__(self):

        self.pretrained_model_name = alt_none(self.pretrained_model_name, self.extradir)
        assert (
            self.pretrained_model_name
        ), f'At least one of pretrained_model_name or extradir should be provided, but got extradir={self.extradir} and pretrained_model_name={self.pretrained_model_name}'


class AutoTokenizer(BaseTokenizer):
    '''
    Wrapper of HuggingFace AutoTokenizer https://huggingface.co/transformers/model_doc/auto.html#autotokenizer.
    '''

    CONFIG_CLS = AutoTokenizerConfig

    def __init__(self, config: AutoTokenizerConfig):
        super().__init__(config)

        try:
            # this logic deals with different huggingface tokenizers having different positional args
            if config.vocab_file is None:
                self.tokenizer = AUTOTOKENIZER.from_pretrained(
                    pretrained_model_name_or_path=config.pretrained_model_name,
                    use_fast=config.use_fast,
                )
            elif config.merges_file is None:
                self.tokenizer = AUTOTOKENIZER.from_pretrained(
                    pretrained_model_name_or_path=config.pretrained_model_name,
                    vocab_file=config.vocab_file,
                    use_fast=config.use_fast,
                )
            else:
                self.tokenizer = AUTOTOKENIZER.from_pretrained(
                    pretrained_model_name_or_path=config.pretrained_model_name,
                    vocab_file=config.vocab_file,
                    merges_file=config.merges_file,
                    use_fast=config.use_fast,
                )
        except Exception as e:
            raise ValueError(
                f'Unable to instantiate HuggingFace AUTOTOKENIZER for {config.pretrained_model_name}. Exception: {e}'
            )

        self.original_vocab_size = len(self.tokenizer)
        special_tokens_dict = {}

        # # setting special tokens, by default the default model's special tokens will be preserved
        # # unless passes new values to the special tokens
        if config.unk_token is not None:
            special_tokens_dict["unk_token"] = config.unk_token
        if config.mask_token is not None:
            special_tokens_dict["mask_token"] = config.mask_token
        if config.pad_token is not None:
            special_tokens_dict["pad_token"] = config.pad_token

        # if the model does not have eos_token but has sep_token,
        # set eos_token = sep_token, and vice versa
        if config.sep_token is not None:
            special_tokens_dict["sep_token"] = config.sep_token
        elif self.tokenizer.sep_token is None and self.tokenizer.eos_token:
            special_tokens_dict["sep_token"] = self.tokenizer.eos_token
        if config.eos_token is not None:
            special_tokens_dict["eos_token"] = config.eos_token
        elif self.tokenizer.eos_token is None and self.tokenizer.sep_token:
            special_tokens_dict["eos_token"] = self.tokenizer.sep_token

        # if the model does not have bos_token but has cls_token,
        # set bos_token = cls_token, and vice versa
        if config.bos_token is not None:
            special_tokens_dict["bos_token"] = config.bos_token
        elif self.tokenizer.bos_token is None and self.tokenizer.cls_token:
            special_tokens_dict["bos_token"] = self.tokenizer.cls_token
        if config.cls_token is not None:
            special_tokens_dict["cls_token"] = config.cls_token
        elif self.tokenizer.cls_token is None and self.tokenizer.bos_token:
            special_tokens_dict["cls_token"] = self.tokenizer.bos_token

        # add additional special tokens (not standard special tokens such as bos, eod, sep)
        if config.additional_special_tokens:
            special_tokens_dict[
                "additional_special_tokens"
            ] = config.additional_special_tokens

        new_tokens_in_vocab = []
        for token in [
            config.mask_token,
            config.bos_token,
            config.eos_token,
            config.pad_token,
            config.sep_token,
            config.cls_token,
            config.unk_token,
        ]:
            if token is not None and token not in self.tokenizer.get_vocab():
                new_tokens_in_vocab.append(token)
        for token in config.additional_special_tokens:
            if token is not None and token not in self.tokenizer.get_vocab():
                new_tokens_in_vocab.append(token)
        if len(new_tokens_in_vocab) > 0:
            """
            Special tokens that were not previously included in the tokenizer's vocabulary file will be added to
            the vocabulary and, as a result, the model should be resized, for example:

            # define your model
            from transformers import AutoModel
            pretrained_model_name = 'roberta'
            model = AutoModel.from_pretrained(pretrained_model_name)

            # define pretrained tokenizer
            from egrecho.data.tokenizers.auto_tokenizer import AutoTokenizer, AutoTokenizerConfig
            t_cfg = AutoTokenizerConfig(pretrained_model_name=pretrained_model_name)
            tokenizer_default = AutoTokenizer(t_cfg)

            special_tokens = {'bos_token': '<BOS>',
                              'cls_token': '<CSL>',
                              'additional_special_tokens': ['<MY_NER_TOKEN>', '<ANOTHER_TOKEN>']}
            tokenizer_default.add_special_tokens(special_tokens_dict=special_tokens)

            # resize your model so that the embeddings for newly added tokens are updated during training/finetuning
            model.resize_token_embeddings(len(tokenizer_default))
            """
            logger.warning(
                f'{new_tokens_in_vocab} \n will be added to the vocabulary.\n'
                f'Please resize your model accordingly, ',
                ranks=0,
            )
        self.add_special_tokens(special_tokens_dict)
        self.space_sensitive = self.text2tokens('x y') != self.text2tokens(
            'x'
        ) + self.text2tokens('y')

    def add_special_tokens(
        self, special_tokens_dict: dict, replace_additional_special_tokens: bool = True
    ) -> int:
        """
        Adds a dictionary of special tokens (eos, pad, cls...). If special tokens are NOT in the vocabulary, they are added
        to it (indexed starting from the last index of the current vocabulary).
        Args:
            special_tokens_dict: dict of string. Keys should be in the list of predefined special attributes:
                [``bos_token``, ``eos_token``, ``unk_token``, ``sep_token``, ``pad_token``, ``cls_token``, ``mask_token``,
                ``additional_special_tokens``].
                Tokens are only added if they are not already in the vocabulary.
            replace_additional_special_tokens (`bool`, *optional*,, defaults to `True`): If `True`,
                the existing list of additional special tokens will be replaced by the list provided in
                `special_tokens_dict`. Otherwise, `self.additional_special_tokens` is just extended. In the former
                case, the tokens will NOT be removed from the tokenizer's full vocabulary - they are only being flagged
                as non-special tokens. Remember, this only affects which tokens are skipped during decoding, not the
                `added_tokens_encoder` and `added_tokens_decoder`. This means that the previous
                `additional_special_tokens` are still added tokens, and will not be split by the model.

        Returns:
            Number of tokens added to the vocabulary.
        """
        num_tokens_added = self.tokenizer.add_special_tokens(
            special_tokens_dict, replace_additional_special_tokens
        )

        if num_tokens_added > 0:
            logger.info(
                f'{num_tokens_added} special tokens added, resize your model accordingly.',
                ranks=0,
            )
        for k in self.tokenizer.SPECIAL_TOKENS_ATTRIBUTES:
            setattr(self, k, getattr(self.tokenizer, k, None))
        return num_tokens_added

    @property
    def additional_special_tokens_ids(self):
        """Returns a list of the additional special tokens (excluding bos, eos, pad, unk). Used to return sentinel tokens for e.g. T5."""
        return [self.token_to_id(token) for token in self.additional_special_tokens]

    @property
    def additional_special_tokens(self) -> List[str]:
        """
        `List[str]`: All the additional special tokens you may want to use. Log an error if used while not having been
        set.
        """
        return self.tokenizer.additional_special_tokens

    @additional_special_tokens.setter
    def additional_special_tokens(self, value):
        self.tokenizer.additional_special_tokens = value

    def token_to_id(self, token):
        return self.tokens2ids([token])[0]

    def text2tokens(self, text):
        tokens = self.tokenizer.tokenize(text)
        return tokens

    def tokens2text(self, tokens):
        text = self.tokenizer.convert_tokens_to_string(tokens)
        return text

    def tokens2ids(self, tokens):
        ids = self.tokenizer.convert_tokens_to_ids(tokens)
        return ids

    def ids2tokens(self, ids):
        tokens = self.tokenizer.convert_ids_to_tokens(ids)
        return tokens

    def text2ids(self, text) -> List[int]:
        return super().text2ids(text)

    def ids2text(self, ids):
        tokens = self.ids2tokens(ids)
        tokens_clean = [t for t in tokens if t not in self.tokenizer.all_special_tokens]
        text = self.tokens2text(tokens_clean)
        return text

    @property
    def all_special_ids(self) -> List[int]:
        """
        Returns:
            `List[int]`: List the ids of the special tokens(`'<unk>'`, `'<cls>'`, etc.) mapped to class attributes.
        """
        return self.tokenizer.all_special_ids

    @property
    def vocab_size(self):
        """
        Returns:
            ``int``: Size of the base vocabulary (without the added tokens).
        """
        return self.tokenizer.vocab_size

    @property
    def vocab(self):
        id2vocab = {v: k for k, v in self.tokenizer.vocab.items()}
        return [id2vocab[i] for i in range(len(id2vocab))]

    @property
    def pad_token_type_id(self) -> int:
        return self.tokenizer.pad_token_type_id

    @property
    def pad(self):
        """Returns pad_id."""
        return self.tokenizer.pad_token_id

    @property
    def pad_token_id(self):
        return self.tokens2ids([getattr(self, 'pad_token')])[0]

    @pad_token_id.setter
    def pad_token_id(self, value):
        self.tokenizer.pad_token_id = value
        setattr(self, 'pad_token', getattr(self.tokenizer, 'pad_token', None))

    @property
    def bos(self):
        return self.tokens2ids([getattr(self, 'bos_token')])[0]

    @property
    def eod(self):
        return self.eos

    @property
    def eos(self):
        return self.tokens2ids([getattr(self, 'eos_token')])[0]

    @property
    def sep(self):
        return self.tokens2ids([getattr(self, 'sep_token')])[0]

    @property
    def cls(self):
        return self.tokens2ids([getattr(self, 'cls_token')])[0]

    @property
    def unk(self):
        return self.tokens2ids([getattr(self, 'unk_token')])[0]

    @property
    def mask(self):
        return self.tokens2ids([getattr(self, 'mask_token')])[0]

    @property
    def name(self):
        return type(self.tokenizer).__name__

    @add_end_docstrings(
        ENCODE_KWARGS_DOCSTRING, ENCODE_PLUS_ADDITIONAL_KWARGS_DOCSTRING
    )
    def __call__(
        self,
        text: Union[
            TextInput, PreTokenizedInput, List[TextInput], List[PreTokenizedInput]
        ] = None,
        text_pair: Optional[
            Union[
                TextInput, PreTokenizedInput, List[TextInput], List[PreTokenizedInput]
            ]
        ] = None,
        text_target: Union[
            TextInput, PreTokenizedInput, List[TextInput], List[PreTokenizedInput]
        ] = None,
        text_pair_target: Optional[
            Union[
                TextInput, PreTokenizedInput, List[TextInput], List[PreTokenizedInput]
            ]
        ] = None,
        add_special_tokens: bool = True,
        padding: Union[bool, str, PaddingStrategy] = False,
        truncation: Union[bool, str, TruncationStrategy] = None,
        max_length: Optional[int] = None,
        stride: int = 0,
        is_split_into_words: bool = False,
        pad_to_multiple_of: Optional[int] = None,
        return_tensors: Optional[Union[str, TensorType]] = None,
        return_token_type_ids: Optional[bool] = None,
        return_attention_mask: Optional[bool] = None,
        return_overflowing_tokens: bool = False,
        return_special_tokens_mask: bool = False,
        return_offsets_mapping: bool = False,
        return_length: bool = False,
        verbose: bool = True,
        **kwargs,
    ) -> BatchEncoding:
        """
        Main method to tokenize and prepare for the model one or several sequence(s) or one or several pair(s) of
        sequences.

        Args:
            text (`str`, `List[str]`, `List[List[str]]`, *optional*):
                The sequence or batch of sequences to be encoded. Each sequence can be a string or a list of strings
                (pretokenized string). If the sequences are provided as list of strings (pretokenized), you must set
                `is_split_into_words=True` (to lift the ambiguity with a batch of sequences).
            text_pair (`str`, `List[str]`, `List[List[str]]`, *optional*):
                The sequence or batch of sequences to be encoded. Each sequence can be a string or a list of strings
                (pretokenized string). If the sequences are provided as list of strings (pretokenized), you must set
                `is_split_into_words=True` (to lift the ambiguity with a batch of sequences).
            text_target (`str`, `List[str]`, `List[List[str]]`, *optional*):
                The sequence or batch of sequences to be encoded as target texts. Each sequence can be a string or a
                list of strings (pretokenized string). If the sequences are provided as list of strings (pretokenized),
                you must set `is_split_into_words=True` (to lift the ambiguity with a batch of sequences).
            text_pair_target (`str`, `List[str]`, `List[List[str]]`, *optional*):
                The sequence or batch of sequences to be encoded as target texts. Each sequence can be a string or a
                list of strings (pretokenized string). If the sequences are provided as list of strings (pretokenized),
                you must set `is_split_into_words=True` (to lift the ambiguity with a batch of sequences).
        """
        return self.tokenizer(
            text=text,
            text_pair=text_pair,
            text_target=text_target,
            text_pair_target=text_pair_target,
            add_special_tokens=add_special_tokens,
            padding=padding,
            truncation=truncation,
            max_length=max_length,
            stride=stride,
            is_split_into_words=is_split_into_words,
            pad_to_multiple_of=pad_to_multiple_of,
            return_tensors=return_tensors,
            return_token_type_ids=return_token_type_ids,
            return_attention_mask=return_attention_mask,
            return_overflowing_tokens=return_overflowing_tokens,
            return_special_tokens_mask=return_special_tokens_mask,
            return_offsets_mapping=return_offsets_mapping,
            return_length=return_length,
            verbose=verbose,
            **kwargs,
        )

    def batch_decode(
        self,
        sequences: Union[List[int], List[List[int]], "np.ndarray", "torch.Tensor"],
        skip_special_tokens: bool = False,
        clean_up_tokenization_spaces: bool = None,
        **kwargs,
    ) -> List[str]:
        """
        Convert a list of lists of token ids into a list of strings by calling decode.

        Args:
            sequences (`Union[List[int], List[List[int]], np.ndarray, torch.Tensor, tf.Tensor]`):
                List of tokenized input ids. Can be obtained using the `__call__` method.
            skip_special_tokens (`bool`, *optional*, defaults to `False`):
                Whether or not to remove special tokens in the decoding.
            clean_up_tokenization_spaces (`bool`, *optional*):
                Whether or not to clean up the tokenization spaces. If `None`, will default to
                `self.clean_up_tokenization_spaces`.
            kwargs (additional keyword arguments, *optional*):
                Will be passed to the underlying model specific decode method.

        Returns:
            `List[str]`: The list of decoded sentences.
        """
        return self.tokenizer.batch_decode(
            sequences,
            skip_special_tokens=skip_special_tokens,
            clean_up_tokenization_spaces=clean_up_tokenization_spaces,
            **kwargs,
        )

    def decode(
        self,
        token_ids: Union[int, List[int], "np.ndarray", "torch.Tensor"],
        skip_special_tokens: bool = False,
        clean_up_tokenization_spaces: bool = None,
        **kwargs,
    ) -> str:
        """
        Converts a sequence of ids in a string, using the tokenizer and vocabulary with options to remove special
        tokens and clean up tokenization spaces.

        Similar to doing `self.convert_tokens_to_string(self.convert_ids_to_tokens(token_ids))`.

        Args:
            token_ids (`Union[int, List[int], np.ndarray, torch.Tensor, tf.Tensor]`):
                List of tokenized input ids. Can be obtained using the `__call__` method.
            skip_special_tokens (`bool`, *optional*, defaults to `False`):
                Whether or not to remove special tokens in the decoding.
            clean_up_tokenization_spaces (`bool`, *optional*):
                Whether or not to clean up the tokenization spaces. If `None`, will default to
                `self.clean_up_tokenization_spaces`.
            kwargs (additional keyword arguments, *optional*):
                Will be passed to the underlying model specific decode method.

        Returns:
            `str`: The decoded sentence.
        """
        # Convert inputs to python lists
        return self.tokenizer.decode(
            token_ids,
            skip_special_tokens=skip_special_tokens,
            clean_up_tokenization_spaces=clean_up_tokenization_spaces,
            **kwargs,
        )

    @add_end_docstrings(
        ENCODE_KWARGS_DOCSTRING, ENCODE_PLUS_ADDITIONAL_KWARGS_DOCSTRING
    )
    def prepare_for_model(
        self,
        ids: List[int],
        pair_ids: Optional[List[int]] = None,
        add_special_tokens: bool = True,
        padding: Union[bool, str, PaddingStrategy] = False,
        truncation: Union[bool, str, TruncationStrategy] = None,
        max_length: Optional[int] = None,
        stride: int = 0,
        pad_to_multiple_of: Optional[int] = None,
        return_tensors: Optional[Union[str, TensorType]] = None,
        return_token_type_ids: Optional[bool] = None,
        return_attention_mask: Optional[bool] = None,
        return_overflowing_tokens: bool = False,
        return_special_tokens_mask: bool = False,
        return_offsets_mapping: bool = False,
        return_length: bool = False,
        verbose: bool = True,
        prepend_batch_axis: bool = False,
        **kwargs,
    ) -> BatchEncoding:
        """
        Prepares a sequence of input id, or a pair of sequences of inputs ids so that it can be used by the model. It
        adds special tokens, truncates sequences if overflowing while taking into account the special tokens and
        manages a moving window (with user defined stride) for overflowing tokens. Please Note, for *pair_ids*
        different than `None` and *truncation_strategy = longest_first* or `True`, it is not possible to return
        overflowing tokens. Such a combination of arguments will raise an error.

        Args:
            ids (`List[int]`):
                Tokenized input ids of the first sequence. Can be obtained from a string by chaining the `tokenize` and
                `convert_tokens_to_ids` methods.
            pair_ids (`List[int]`, *optional*):
                Tokenized input ids of the second sequence. Can be obtained from a string by chaining the `tokenize`
                and `convert_tokens_to_ids` methods.
        """
        return self.tokenizer.prepare_for_model(
            ids,
            pair_ids=pair_ids,
            add_special_tokens=add_special_tokens,
            padding=padding,
            truncation=truncation,
            max_length=max_length,
            stride=stride,
            pad_to_multiple_of=pad_to_multiple_of,
            return_tensors=return_tensors,
            return_token_type_ids=return_token_type_ids,
            return_attention_mask=return_attention_mask,
            return_overflowing_tokens=return_overflowing_tokens,
            return_special_tokens_mask=return_special_tokens_mask,
            return_offsets_mapping=return_offsets_mapping,
            return_length=return_length,
            verbose=verbose,
            prepend_batch_axis=prepend_batch_axis,
            **kwargs,
        )

    @contextmanager
    def _temporarily_set_config_attribute(self, attribute, value):
        """Temporarily sets an attribute on the config and restores it on exit."""
        original_value = getattr(self.config, attribute)
        setattr(self.config, attribute, value)
        try:
            yield
        finally:
            setattr(self.config, attribute, original_value)

    def save_to(
        self,
        savedir,
        filename_prefix: Optional[str] = None,
        **kwargs,
    ):
        if self.config.save_hf:
            with self._temporarily_set_config_attribute('pretrained_model_name', None):
                super().save_to(savedir, filename_prefix=filename_prefix, **kwargs)
        else:
            super().save_to(savedir, filename_prefix=filename_prefix, **kwargs)

    def save_extras(
        self, savedir: str, filename_prefix: Optional[str] = None
    ) -> Tuple[str]:
        """Derived classes should overwrite it for special savings."""
        if self.config.save_hf:
            return self.tokenizer.save_pretrained(
                savedir, filename_prefix=filename_prefix
            )

    def save_vocabulary(self, save_directory: str, filename_prefix: str = None):
        """Saves tokenizer's vocabulary and other artifacts to the specified directory"""
        return self.tokenizer.save_vocabulary(
            save_directory=save_directory, filename_prefix=filename_prefix
        )

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(config='{self.config}', \n\ttokenizer={self.tokenizer}"

    def __len__(self):
        return len(self.tokenizer)
