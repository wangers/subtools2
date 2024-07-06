# -*- coding:utf-8 -*-
# Copyright xmuspeech (Author: Leo 2024-03-11)

import itertools
import os
import warnings
from abc import ABC, abstractmethod
from collections import UserDict
from collections.abc import Mapping, Sized
from dataclasses import dataclass, field
from functools import lru_cache, partial
from pathlib import Path
from shutil import copyfile
from typing import TYPE_CHECKING, Any, Dict, List, Optional, Tuple, Union

import numpy as np
import torch

import egrecho.utils.constants as constants
from egrecho.core.config import DataclassConfig
from egrecho.utils.apply import apply_to_collection
from egrecho.utils.common import OrderedDict, alt_none
from egrecho.utils.imports import _TRANSFORMERS_AVAILABLE, lazy_import
from egrecho.utils.io.files import is_remote_url
from egrecho.utils.logging import get_logger
from egrecho.utils.misc import (
    ConfigurationException,
    add_end_docstrings,
    rich_exception_info,
)
from egrecho.utils.torch_utils import is_numpy_array, is_torch_tensor, to_py_obj
from egrecho.utils.types import PaddingStrategy, StrEnum

if _TRANSFORMERS_AVAILABLE:
    if TYPE_CHECKING:
        import transformers
    else:
        transformers = lazy_import("transformers")

logger = get_logger(__name__)

TextInput = str
PreTokenizedInput = List[str]
EncodedInput = List[int]
TextInputPair = Tuple[str, str]
PreTokenizedInputPair = Tuple[List[str], List[str]]
EncodedInputPair = Tuple[List[int], List[int]]

VERY_LARGE_INTEGER = int(
    1e30
)  # This is used to set the max input length for a model with infinite size input
LARGE_INTEGER = int(
    1e20
)  # This is used when we need something big but slightly smaller than VERY_LARGE_INTEGER


class TruncationStrategy(StrEnum):
    """
    Possible values for the ``truncation`` argument in :meth:`Tokenizer.__call__`. Useful for tab-completion in
    an IDE.
    """

    ONLY_FIRST = "only_first"
    ONLY_SECOND = "only_second"
    LONGEST_FIRST = "longest_first"
    DO_NOT_TRUNCATE = "do_not_truncate"


class TensorType(StrEnum):
    """For tab-completion in an IDE."""

    PYTORCH = "pt"
    NUMPY = "np"


@dataclass
class BaseTokenizerConfig(DataclassConfig):
    """
    Base class for the :class:`BaseTokenizer` configuration.

    The path :attr:`extradir` will **not** be serialized in the config file. When deserialized from
    a config file (tokenizer_config.json), it will be set to the directory of the config file by default,
    allowing for the locationing of files defined by :meth:`extra_files_names`.

    Args:
        extradir (Optional[Union[str, Path]]):
            Path to the directory containing vocabulary files defined by :meth:``extra_files_names``.
    """

    extradir: Optional[str] = field(default=None, metadata={"to_dict": False})

    @property
    def extra_files_names(self):
        """
        Defines the extra file names required by the model. Can be either:

        - A dictionary with values being the filenames for saving the associated files (strings).
        - A tuple/list of filenames.
        """
        return OrderedDict()

    def get_extra_files(
        self, extra_files_names: Any = None, check_local_exist: bool = True
    ):
        """Recursively adds prefix dir to locate extra files."""
        extra_files_names = extra_files_names or self.extra_files_names
        if extra_files_names:
            if (extradir := self.extradir) is None:
                raise ConfigurationException(
                    f"Invalid extradir {extradir} where to fetch files {extra_files_names}"
                )

            return self.add_prefix(
                extra_files_names,
                prefix=str(extradir),
                check_local_exist=check_local_exist,
            )

    @classmethod
    def from_cfg_dir(
        cls,
        srcdir: Union[str, os.PathLike],
        **kwargs,
    ) -> "BaseTokenizerConfig":
        r"""
        Instantiate a :class:``BaseTokenizerConfig`` (or a derived class).
        """
        if is_remote_url(srcdir):
            raise NotImplementedError("TO DO, support remote file.")
        resolved_path = srcdir
        assert os.path.isdir(resolved_path), f"{resolved_path} should be a directory."
        tokenizer_config_path = (
            Path(resolved_path) / constants.DEFAULT_TOKENIZER_FILENAME
        )
        tokenizer_kwargs = cls.load_cfg_file(tokenizer_config_path)
        extradir = tokenizer_kwargs.pop("extradir", None)
        extradir = extradir or str(resolved_path)
        tokenizer_kwargs["extradir"] = extradir
        return cls.from_config(tokenizer_kwargs, **kwargs)

    def copy_extras(
        self,
        savedir,
        extra_files_names=None,
        filename_prefix=None,
        excludes: Union[str, List[str], None] = None,
        **kwargs,
    ) -> Tuple[str]:
        """
        Copys the extra files of the tokenizer config.

        Use :meth:``BaseTokenizer.save_to`` to save the whole configuration
        (config file + extra files) of the tokenizer. This method will copy all files
        defined by :meth:``extra_files_names`` by defaults.
        Args:
            savedir (``str``):
                The directory in which to save the extra files.
            extra_files_names:
                If None, use default files defined by class property `extra_files_names`
            filename_prefix (``str``, *optional*):
                An optional prefix to add to the named of the saved files.
            excludes (Union[``str``, ``List[str]``]):
                Excludes what fnames.

        Returns:
            ``Tuple(str)``: Paths to the files saved.
        """
        tgt_files = []

        def _cp_file(fname, srcdir, f_prefix, exs):
            if fname in exs:
                return
            assert os.path.isdir(savedir), f"{savedir} should be a directory"
            if os.path.isdir(Path(srcdir)):
                resolved_file = os.path.join(srcdir, fname)
                if not os.path.isfile(resolved_file):
                    raise FileExistsError(
                        f"{srcdir} does not appear to have a file named {fname}. "
                    )
                save_extra_file = os.path.join(savedir, f_prefix + fname)
                if os.path.abspath(resolved_file) != os.path.abspath(save_extra_file):
                    copyfile(resolved_file, save_extra_file)
                tgt_files.append(str(save_extra_file))

        extra_files_names = alt_none(extra_files_names, self.extra_files_names)
        filename_prefix = filename_prefix + "-" if filename_prefix else ""
        extradir = self.extradir
        # about to copy files
        if extradir is not None and extra_files_names:
            extradir = str(extradir)
            excludes = excludes or []
            if isinstance(excludes, str):
                excludes = [excludes]
            cp_fn = partial(
                _cp_file, srcdir=extradir, f_prefix=filename_prefix, exs=excludes
            )
            apply_to_collection(
                extra_files_names, dtype=(str, os.PathLike), function=cp_fn
            )
        return tuple(tgt_files)

    @staticmethod
    @rich_exception_info
    def add_prefix(
        extra_files_names: Any, prefix: str, check_local_exist: bool = False
    ):
        def _add_prefix(fname: str):
            resolved_file = os.path.join(prefix, fname)
            if os.path.isdir(prefix):
                if check_local_exist and not os.path.isfile(resolved_file):
                    raise FileExistsError(
                        f"{prefix} does not appear to have a file named {fname}. "
                    )
            return resolved_file

        return apply_to_collection(extra_files_names, dtype=str, function=_add_prefix)


class BaseTokenizer(ABC):
    r"""
    A base class offers serialize methods for tokenizer. and derived classes should implement its
    encode/decode methods (:meth:``text2ids``, :meth:``ids2text``, etc..)

    The implementation of the tokenizer method is intended for derived classes.
    Its purpose is to facilitate coordination between model inputs and the frontend data processor.

    Unlike :class:``egrecho.core.feature_extractor.speaker.BaseFeature``, which is designed to save
    its mainly attributes as config itself. :class:``BaseTokenizer`` maintains an inside :attr:``config`` instance.

    Class attributes (overridden by derived classes)

        - **CONFIG_CLS** -- The type of assosiate :class:``BaseTokenizerConfig`` (or a derived class).

    Args:
        config (BaseTokenizerConfig):
            configuration object.
    """
    CONFIG_CLS: BaseTokenizerConfig

    def __init__(self, config: BaseTokenizerConfig):
        if not isinstance(config, self.CONFIG_CLS):
            raise ValueError(
                f"Parameter config in ``{self.__class__.__name__}(config)`` "
                f"should be an {self.CONFIG_CLS.__name__} instance, "
                f"but got {type(config)!r}."
            )

        self._config = config

        self._candidate_pad_id = None
        self._pad_token_type_id = 0

    @property
    def name(self):
        return type(self).__name__

    @property
    def cls(self):
        """Returns cls_id if available."""
        if hasattr(self, "cls_id"):
            return self.cls_id
        raise AttributeError(
            f"{type(self).__name__} has no attribute 'cls' or 'cls_id'"
        )

    @property
    def sep(self):
        """Returns sep_id if available."""
        if hasattr(self, "sep_id"):
            return self.sep_id
        raise AttributeError(
            f"{type(self).__name__} has no attribute 'sep' or 'sep_id'"
        )

    @property
    def pad(self):
        """Returns pad_id if available."""
        if hasattr(self, "pad_id"):
            return self.pad_id
        raise AttributeError(
            f"{type(self).__name__} has no attribute 'pad' or 'pad_id'"
        )

    @property
    def pad_token_id(self):
        """Returns pad_id if available."""
        pad_id = getattr(self, "pad_id", None)
        if pad_id is None:
            pad_id = self._candidate_pad_id
        return pad_id

    @pad_token_id.setter
    def pad_token_id(self, value):
        self._candidate_pad_id = value

    @property
    def pad_token_type_id(self) -> int:
        """
        ``int``: Id of the padding token type in the vocabulary.
        """
        return self._pad_token_type_id

    @property
    def eod(self):
        """Returns eod_id if available."""
        if hasattr(self, "eod_id"):
            return self.eod_id
        if hasattr(self, "eos_id"):
            # Default to end-of-sentence id if end-of-document is not defined.
            return self.eos_id
        raise AttributeError(
            f"{type(self).__name__} has no attribute 'eod', 'eod_id', 'eos', or 'eos_id'"
        )

    @property
    def bos(self):
        """Returns bos_id if available."""
        if hasattr(self, "bos_id"):
            return self.bos_id
        raise AttributeError(
            f"{type(self).__name__} has no attribute 'bos' or 'bos_id'"
        )

    @property
    def eos(self):
        """Returns eos_id if available."""
        if hasattr(self, "eos_id"):
            return self.eos_id
        raise AttributeError(
            f"{type(self).__name__} has no attribute 'eos' or 'eos_id'"
        )

    @property
    def mask(self):
        """Returns mask_id if available."""
        if hasattr(self, "mask_id"):
            return self.mask_id
        raise AttributeError(
            f"{type(self).__name__} has no attribute 'mask' or 'mask_id'"
        )

    def tokenize(self, line: TextInput, **kwargs) -> List[str]:
        """Accept kwargs for tokenize, overwrite it in subclasses."""
        return self.text2tokens(line)

    def text2ids(self, line: TextInput) -> List[int]:
        return self.text2tokens(self.tokens2ids(line))

    def ids2text(self, ids: EncodedInput) -> str:
        return self.ids2tokens(self.tokens2text(ids))

    @abstractmethod
    def text2tokens(self, line: TextInput) -> List[str]:
        raise NotImplementedError("abstract method")

    def tokens2text(self, tokens: PreTokenizedInput) -> str:
        raise NotImplementedError("abstract method")

    @abstractmethod
    def tokens2ids(self, tokens: PreTokenizedInput) -> List[int]:
        raise NotImplementedError("abstract method")

    def ids2tokens(self, ids: EncodedInput) -> List[str]:
        raise NotImplementedError("abstract method")

    def add_special_tokens(self, special_tokens: List[str]):
        raise NotImplementedError("To be implemented")

    @property
    def all_special_ids(self) -> List[int]:
        """
        `List[int]`: List the ids of the special tokens(`'<unk>'`, `'<cls>'`, etc.) mapped to class attributes.
        """
        raise NotImplementedError

    @property
    def vocab_size(self) -> int:
        """
        ``int``: Size of the base vocabulary (without the added tokens).
        """
        raise NotImplementedError

    # alias
    @property
    def config(self):
        """Refs config"""
        return self._config

    @property
    def extradir(self):
        """Refs extra files directory."""
        return self.config.extradir

    @classmethod
    def from_cfg_dir(
        cls,
        srcdir: Union[str, os.PathLike],
        **kwargs,
    ) -> "BaseTokenizer":
        r"""
        Instantiate a :class:``BaseTokenizer`` (or a derived class) from a
        dir has config files.
        """
        config = cls.CONFIG_CLS.from_cfg_dir(srcdir, **kwargs)
        return cls(config)

    def save_to(
        self,
        savedir,
        filename_prefix: Optional[str] = None,
        **kwargs,
    ):
        """Saves the whole configuration (config file + extra files) of the tokenizer

        Args:
            savedir (``str``):
                The directory in which to save the extra files.
            filename_prefix (``str``, *optional*):
                An optional prefix to add to the named of the saved files.
        """
        if os.path.isfile(savedir):
            logger.error(f"Provided path ({savedir}) should be a directory, not a file")
            return

        os.makedirs(savedir, exist_ok=True)
        tok_cfg_fname = kwargs.pop("config_fname", constants.DEFAULT_TOKENIZER_FILENAME)
        tok_cfg_file = os.path.join(savedir, tok_cfg_fname)
        self.save_config(tok_cfg_file)
        self.save_extras(savedir=savedir, filename_prefix=filename_prefix)

    def save_config(self, path: str):
        """
        save the configuration to a file.

        Args:
            path (Union[Path, str]):
                The path of the output file.
        """
        self.config.to_file(path)

    def save_extras(
        self, savedir: str, filename_prefix: Optional[str] = None
    ) -> Tuple[str]:
        """Derived classes should overwrite it for special savings."""
        raise NotImplementedError

    def default_save_extras(
        self,
        savedir: str,
        filename_prefix: Optional[str] = None,
        excludes: Union[str, List[str], None] = None,
        **kwargs,
    ):
        """A default funtion conviniently copies extra files."""
        self.config.copy_extras(
            savedir=savedir,
            filename_prefix=filename_prefix,
            excludes=excludes,
            **kwargs,
        )

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(config='{self.config}', vocab_size={self.vocab_size}"


ENCODE_KWARGS_DOCSTRING = r"""
            add_special_tokens (``bool``, *optional*, defaults to ``True``):
                Whether or not to add special tokens when encoding the sequences. This will use the underlying
                :meth:`Tokenizer.build_inputs_with_special_tokens` function, which defines which tokens are
                automatically added to the input ids. This is usefull if you want to add ``bos`` or ``eos`` tokens
                automatically.
            padding (``bool``, ``str`` or :class:``~egrecho.utils.types.PaddingStrategy``, *optional*, defaults to ``False``):
                Activates and controls padding. Accepts the following values:

            - ``True`` or ``'longest'``: Pad to the longest sequence in the batch (or no padding if only a single
                sequence if provided).
            - ``'max_length'``: Pad to a maximum length specified with the argument ``max_length`` or to the maximum
                acceptable input length for the model if that argument is not provided.
            - ``False`` or ``'do_not_pad'`` (default): No padding (i.e., can output a batch with sequences of different
                lengths).

            truncation (``bool``, ``str`` or :class:``~egrecho.core.tokenizer.TruncationStrategy``, *optional*, defaults to ``False``):
                Activates and controls truncation. Accepts the following values:

            - ``True`` or ``'longest_first'``: Truncate to a maximum length specified with the argument ``max_length`` or
            to the maximum acceptable input length for the model if that argument is not provided. This will
            truncate token by token, removing a token from the longest sequence in the pair if a pair of
            sequences (or a batch of pairs) is provided.
            - ``'only_first'``: Truncate to a maximum length specified with the argument ``max_length`` or to the
            maximum acceptable input length for the model if that argument is not provided. This will only
            truncate the first sequence of a pair if a pair of sequences (or a batch of pairs) is provided.
            - ``'only_second'``: Truncate to a maximum length specified with the argument ``max_length`` or to the
            maximum acceptable input length for the model if that argument is not provided. This will only
            truncate the second sequence of a pair if a pair of sequences (or a batch of pairs) is provided.
            - ``False`` or ``'do_not_truncate'`` (default): No truncation (i.e., can output batch with sequence lengths
            greater than the model maximum admissible input size).

            max_length (``int``, *optional*):
                Controls the maximum length to use by one of the truncation/padding parameters.

                If left unset or set to ``None``, this will use the predefined model maximum length if a maximum length
                is required by one of the truncation/padding parameters. If the model has no specific maximum input
                length (like XLNet) truncation/padding to a maximum length will be deactivated.
            stride (``int``, *optional*, defaults to 0):
                If set to a number along with ``max_length``, the overflowing tokens returned when
                ``return_overflowing_tokens=True`` will contain some tokens from the end of the truncated sequence
                returned to provide some overlap between truncated and overflowing sequences. The value of this
                argument defines the number of overlapping tokens.
            is_split_into_words (``bool``, *optional*, defaults to ``False``):
                Whether or not the input is already pre-tokenized (e.g., split into words). If set to ``True``, the
                tokenizer assumes the input is already split into words (for instance, by splitting it on whitespace)
                which it will tokenize. This is useful for NER or token classification.
            pad_to_multiple_of (``int``, *optional*):
                If set will pad the sequence to a multiple of the provided value. Requires ``padding`` to be activated.
                This is especially useful to enable the use of Tensor Cores on NVIDIA hardware with compute capability
                ``>= 7.5`` (Volta).
            return_tensors (``str`` or :class:`~egrecho.core.tokenizer.TensorType`, *optional*):
                If set, will return tensors instead of list of python integers. Acceptable values are:

            - ``'pt'``: Return PyTorch ``torch.Tensor`` objects.
            - ``'np'``: Return Numpy ``np.ndarray`` objects.
"""

ENCODE_ADDITIONAL_KWARGS_DOCSTRING = r"""
            return_token_type_ids (``bool``, *optional*):
                Whether to return token type IDs. If left to the default, will return the token type IDs according to
                the specific tokenizer's default, defined by the ``return_outputs`` attribute.
            return_attention_mask (``bool``, *optional*):
                Whether to return the attention mask. If left to the default, will return the attention mask according
                to the specific tokenizer's default, defined by the ``return_outputs`` attribute.
            return_overflowing_tokens (``bool``, *optional*, defaults to ``False``):
                Whether or not to return overflowing token sequences. If a pair of sequences of input ids (or a batch
                of pairs) is provided with ``truncation_strategy = longest_first`` or ``True``, an error is raised instead
                of returning overflowing tokens.
            return_special_tokens_mask (``bool``, *optional*, defaults to ``False``):
                Whether or not to return special tokens mask information.
            return_length  (``bool``, *optional*, defaults to ``False``):
                Whether or not to return the lengths of the encoded inputs.
            verbose (``bool``, *optional*, defaults to ``True``):
                Whether or not to print more information and warnings.
            \**kwargs: passed to the ``self.tokenize()`` method

        Return:
            [``BatchEncoding``]: A [``BatchEncoding``] with the following fields:

            - **input_ids** -- List of token ids to be fed to a model.
            - **token_type_ids** -- List of token type ids to be fed to a model (when ``return_token_type_ids=True`` or
              if *"token_type_ids"* is in ``self.model_input_names``).
            - **attention_mask** -- List of indices specifying which tokens should be attended to by the model (when
              ``return_attention_mask=True`` or if *"attention_mask"* is in ``self.model_input_names``).
            - **overflowing_tokens** -- List of overflowing tokens sequences (when a ``max_length`` is specified and
              ``return_overflowing_tokens=True``).
            - **num_truncated_tokens** -- Number of tokens truncated (when a ``max_length`` is specified and
              ``return_overflowing_tokens=True``).
            - **special_tokens_mask** -- List of 0s and 1s, with 1 specifying added special tokens and 0 specifying
              regular sequence tokens (when ``add_special_tokens=True`` and ``return_special_tokens_mask=True``).
            - **length** -- The length of the inputs (when ``return_length=True``)
"""


class Tokenizer(BaseTokenizer):
    """A base class aims to prepare model inputs via __call__ interface , derived from :class:`BaseTokenizer`.
    And offers some useful methods for padding/truncate. Core methods:

    - :meth:`__call__`:
      Abstract method to tokenize and prepare for the model, can handle single or batch inputs.
    - :meth:`prepare_for_model` (**one sample**):
      Prepares a sequence of input id (tokenized by the :meth:`text2ids`), or a pair of sequences of inputs ids so
      that it can be used by the model. Workflows typically follows:

        - Pre-define settings: Get truncate/pad strategy. Computes the total size of the returned encodings
          via :meth:``num_special_tokens_to_add``. Which default hacks building
          empty input ids through :meth:``build_inputs_with_special_tokens``.
        - Truncates: :meth:``truncate_sequences``.
        - Add special tokens like eos/sos, the list method should be overriden in a subclass:

            * :meth:``build_inputs_with_special_tokens``: Build model inputs from given ids.
            * :meth:``create_token_type_ids_from_sequences``: Create the token type IDs corresponding to the sequences.

        - Pad: pad a sample use :meth:``pad``.

    - :meth:`batch_decode` / :meth:`decode`: inverse of :meth:`__call__`, depends on :meth:`_decode` in subclasses.

    Class attributes (overridden by derived classes)

        - **CONFIG_CLS** -- The type of assosiate :class:`BaseTokenizerConfig` (or a derived class).
        - **model_input_names** (``List[str]``) -- A list of inputs expected in the forward pass of the model.
        - **padding_side** (``str``) -- The default value for the side on which the model should have padding applied.
          Should be ``'right'`` or ``'left'``.
        - **truncation_side** (``str``) -- The default value for the side on which the model should have truncation
          applied. Should be ``'right'`` or ``'left'``.

    Note:
        Heavily borrowed and adapted from tokenizer module in `huggingface tokenizer
        <https://github.com/huggingface/transformers/blob/main/src/transformers/tokenization_utils_base.py>`_.

    Args:
        config:
            configuration object derived from :class:`BaseTokenizerConfig`.
    """

    # first name has to correspond to main model input name
    # to make sure ``tokenizer.pad(...)`` works correctly
    model_input_names: List[str] = ["input_ids", "token_type_ids", "attention_mask"]
    padding_side: str = "right"
    truncation_side: str = "right"

    def __init__(self, config: BaseTokenizerConfig):
        model_max_length = getattr(config, "model_max_length", None)
        self.model_max_length = (
            model_max_length if model_max_length is not None else VERY_LARGE_INTEGER
        )
        # Padding and truncation side are right by default and overridden in subclasses. If specified in the config, it
        # is changed.
        self.padding_side = getattr(config, "padding_side", self.padding_side)
        if self.padding_side not in ["right", "left"]:
            raise ValueError(
                f"Padding side should be selected between 'right' and 'left', current value: {self.padding_side}"
            )

        self.truncation_side = getattr(config, "truncation_side", self.truncation_side)
        if self.truncation_side not in ["right", "left"]:
            raise ValueError(
                f"Truncation side should be selected between 'right' and 'left', current value: {self.truncation_side}"
            )

        self.model_input_names = getattr(
            config, "model_input_names", self.model_input_names
        )
        super().__init__(config)

    @staticmethod
    def _check_text_input(text, text_pair=None):
        # Input type checking for clearer error
        def _is_valid_text_input(t):
            if isinstance(t, str):
                # Strings are fine
                return True
            elif isinstance(t, (list, tuple)):
                # List are fine as long as they are...
                if len(t) == 0:
                    # ... empty
                    return True
                elif isinstance(t[0], str):
                    # ... list of strings
                    return True
                elif isinstance(t[0], (list, tuple)):
                    # ... list with an empty list or with a list of strings
                    return len(t[0]) == 0 or isinstance(t[0][0], str)
                else:
                    return False
            else:
                return False

        if not _is_valid_text_input(text):
            raise ValueError(
                "text input must be of type ``str`` (single example), ``List[str]`` (batch or single pretokenized example) "
                "or ``List[List[str]]`` (batch of pretokenized examples)."
            )

        if text_pair is not None and not _is_valid_text_input(text_pair):
            raise ValueError(
                "text input must be of type ``str`` (single example), ``List[str]`` (batch or single pretokenized example) "
                "or ``List[List[str]]`` (batch of pretokenized examples)."
            )

    @staticmethod
    def _check_batched(text, is_split_into_words: bool) -> bool:
        if is_split_into_words:
            is_batched = (
                isinstance(text, (list, tuple))
                and text
                and isinstance(text[0], (list, tuple))
            )
        else:
            is_batched = isinstance(text, (list, tuple))
        return is_batched

    @classmethod
    def input_text_batched(
        cls, text, text_pair=None, is_split_into_words: bool = False
    ) -> bool:
        """Detect inputs text is valid batched."""
        cls._check_text_input(text, text_pair)
        is_batched = cls._check_batched(text, is_split_into_words=is_split_into_words)
        if is_batched:
            if isinstance(text_pair, str):
                raise TypeError(
                    "when tokenizing batches of text, `text_pair` must be a list or tuple with the same length as"
                    " `text`."
                )
            if text_pair is not None and len(text) != len(text_pair):
                raise ValueError(
                    f"batch length of `text`: {len(text)} does not match batch length of `text_pair`:"
                    f" {len(text_pair)}."
                )
        return is_batched

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
        add_special_tokens: bool = True,
        padding: Union[bool, str, PaddingStrategy] = False,
        truncation: Union[bool, str, TruncationStrategy] = None,
        max_length: Optional[int] = None,
        is_split_into_words: bool = False,
        **kwargs,
    ):
        """
        Main abstract method to tokenize and prepare for the model one or several sequence(s) or one or several pair(s) of
        sequences. Below lists a possible format of inputs.

        Tip:
            A preferred paradigm of inputs:

            - ``is_split_into_words=False``, input text as follows:
                - List[List[str]]: list with a list of strings, **batch** of tokenized tokens, i.e., need tokens2ids.
                - List[str]: list of strings, **batch** of strings, i.e., need text2ids.
                - str: **single** string, i.e., need directly text2ids.
            - ``is_split_into_words=True``, input text as follows:
                - List[List[str]]: list with a list of strings, **batch** of pretokenized (not tokenized but splited), i.e., need text2ids in inner list.
                - List[str]: list of strings, **single** pretokenized, i.e., need text2ids one by one.
                - str: **single** string, auto fallback to is_split_into_words=False.

        Args:
            text (``str``, ``List[str]``, ``List[List[str]]``, *optional*):
                The sequence or batch of sequences to be encoded. Each sequence can be a string or a list of strings
                (pretokenized string). If the sequences are provided as list of strings (pretokenized), you must set
                ``is_split_into_words=True`` (to lift the ambiguity with a batch of sequences).
            text_pair (``str``, ``List[str]``, ``List[List[str]]``, *optional*):
                The sequence or batch of sequences to be encoded. Each sequence can be a string or a list of strings
                (pretokenized string). If the sequences are provided as list of strings (pretokenized), you must set
                ``is_split_into_words=True`` (to lift the ambiguity with a batch of sequences).
            add_special_tokens (``bool``, *optional*, defaults to ``True``):
                Whether or not to add special tokens when encoding the sequences. This will use the underlying
                :meth:`Tokenizer.build_inputs_with_special_tokens` function, which defines which tokens are
                automatically added to the input ids. This is usefull if you want to add ``bos`` or ``eos`` tokens
                automatically.
            padding (``bool``, ``str`` or :class:`~egrecho.utils.types.PaddingStrategy`, *optional*, defaults to ``False``):
                Activates and controls padding. Accepts the following values:

            - ``True`` or ``'longest'``: Pad to the longest sequence in the batch (or no padding if only a single
                sequence if provided).
            - ``'max_length'``: Pad to a maximum length specified with the argument ``max_length`` or to the maximum
                acceptable input length for the model if that argument is not provided.
            - ``False`` or ``'do_not_pad'`` (default): No padding (i.e., can output a batch with sequences of different
                lengths).

            truncation (``bool``, ``str`` or :class:`~egrecho.core.tokenizer.TruncationStrategy`, *optional*, defaults to ``False``):
                Activates and controls truncation. Accepts the following values:

            - ``True`` or ``'longest_first'``: Truncate to a maximum length specified with the argument ``max_length`` or
            to the maximum acceptable input length for the model if that argument is not provided. This will
            truncate token by token, removing a token from the longest sequence in the pair if a pair of
            sequences (or a batch of pairs) is provided.
            - ``'only_first'``: Truncate to a maximum length specified with the argument ``max_length`` or to the
            maximum acceptable input length for the model if that argument is not provided. This will only
            truncate the first sequence of a pair if a pair of sequences (or a batch of pairs) is provided.
            - ``'only_second'``: Truncate to a maximum length specified with the argument ``max_length`` or to the
            maximum acceptable input length for the model if that argument is not provided. This will only
            truncate the second sequence of a pair if a pair of sequences (or a batch of pairs) is provided.
            - ``False`` or ``'do_not_truncate'`` (default): No truncation (i.e., can output batch with sequence lengths
            greater than the model maximum admissible input size).

            max_length (``int``, *optional*):
                Controls the maximum length to use by one of the truncation/padding parameters.

                If left unset or set to ``None``, this will use the predefined model maximum length if a maximum length
                is required by one of the truncation/padding parameters. If the model has no specific maximum input
                length (like XLNet) truncation/padding to a maximum length will be deactivated.
            is_split_into_words (``bool``, *optional*, defaults to ``False``):
                Whether or not the input is already pre-tokenized (e.g., split into words). If set to ``True``, the
                tokenizer assumes the input is already split into words (for instance, by splitting it on whitespace)
                which it will tokenize. This is useful for NER or token classification.
            \**kwargs:
                Additional keyword arguments.
        """
        raise NotImplementedError

    def get_input_ids(self, text, is_split_into_words: bool = False, **kwargs):
        if isinstance(text, str):
            tokens = self.tokenize(text, **kwargs)
            return self.tokens2ids(tokens)
        elif (
            isinstance(text, (list, tuple))
            and len(text) > 0
            and isinstance(text[0], str)
        ):
            if is_split_into_words:
                tokens = list(
                    itertools.chain(
                        *(
                            self.tokenize(t, is_split_into_words=True, **kwargs)
                            for t in text
                        )
                    )
                )
                return self.tokens2ids(tokens)
            else:
                return self.tokens2ids(text)
        elif (
            isinstance(text, (list, tuple))
            and len(text) > 0
            and isinstance(text[0], int)
        ):
            return text
        else:
            raise ValueError(
                "Input is not valid. Should be a string, a list/tuple of strings or a list/tuple of integers."
            )

    def _get_padding_truncation_strategies(
        self,
        padding=False,
        truncation=None,
        max_length=None,
        pad_to_multiple_of=None,
        verbose=True,
        **kwargs,
    ):
        """
        Find the correct padding/truncation strategy.
        """

        if padding is not False:
            if padding is True:
                if verbose:
                    if max_length is not None and (
                        truncation is None
                        or truncation is False
                        or truncation == "do_not_truncate"
                    ):
                        warnings.warn(
                            "``max_length`` is ignored when ``padding=True`` and there is no truncation strategy. "
                            "To pad to max length, use ``padding='max_length'``."
                        )
                padding_strategy = (
                    PaddingStrategy.LONGEST
                )  # Default to pad to the longest sequence in the batch
            elif not isinstance(padding, PaddingStrategy):
                padding_strategy = PaddingStrategy(padding)
            elif isinstance(padding, PaddingStrategy):
                padding_strategy = padding
        else:
            padding_strategy = PaddingStrategy.DO_NOT_PAD

        if truncation is not False and truncation is not None:
            if truncation is True:
                truncation_strategy = (
                    TruncationStrategy.LONGEST_FIRST
                )  # Default to truncate the longest sequences in pairs of inputs
            elif not isinstance(truncation, TruncationStrategy):
                truncation_strategy = TruncationStrategy(truncation)
            elif isinstance(truncation, TruncationStrategy):
                truncation_strategy = truncation
        else:
            truncation_strategy = TruncationStrategy.DO_NOT_TRUNCATE

        # Set max length if needed
        if max_length is None:
            if padding_strategy == PaddingStrategy.MAX_LENGTH:
                if self.model_max_length > LARGE_INTEGER:
                    if verbose:
                        logger.warning_once(
                            "Asking to pad to max_length but no maximum length is provided and the model has no"
                            " predefined maximum length. Default to no padding."
                        )
                    padding_strategy = PaddingStrategy.DO_NOT_PAD
                else:
                    max_length = self.model_max_length

            if truncation_strategy != TruncationStrategy.DO_NOT_TRUNCATE:
                if self.model_max_length > LARGE_INTEGER:
                    if verbose:
                        logger.warning_once(
                            "Asking to truncate to max_length but no maximum length is provided and the model has"
                            " no predefined maximum length. Default to no truncation."
                        )
                    truncation_strategy = TruncationStrategy.DO_NOT_TRUNCATE
                else:
                    max_length = self.model_max_length

        # Test if we have a padding token
        if padding_strategy != PaddingStrategy.DO_NOT_PAD and (
            self.pad_token_id is None
        ):
            raise ValueError(
                "Asking to pad but the tokenizer does not have a padding token. "
                "Please select a token to use as ``pad_token_id``, ``(tokenizer.pad_token_id = tokenizer.eos e.g.)``"
            )

        # Check that we will truncate to a multiple of pad_to_multiple_of if both are provided
        if (
            truncation_strategy != TruncationStrategy.DO_NOT_TRUNCATE
            and padding_strategy != PaddingStrategy.DO_NOT_PAD
            and pad_to_multiple_of is not None
            and max_length is not None
            and (max_length % pad_to_multiple_of != 0)
        ):
            raise ValueError(
                "Truncation and padding are both activated but "
                f"truncation length ({max_length}) is not a multiple of pad_to_multiple_of ({pad_to_multiple_of})."
            )

        return padding_strategy, truncation_strategy, max_length, kwargs

    def num_special_tokens_to_add(self, pair: bool = False) -> int:
        """
        Returns the number of added tokens when encoding a sequence with special tokens.

        Args:
            pair (``bool``, *optional*, defaults to ``False``):
                Whether the number of added tokens should be computed in the case of a sequence pair or a single
                sequence.

        Returns:
            ``int``: Number of special tokens added to sequences.

        NOTE:
            This encodes a dummy input and checks the number of added tokens, and is therefore not efficient.
            Do not put this inside your training loop.
        """
        token_ids_0 = []
        token_ids_1 = []
        return len(
            self.build_inputs_with_special_tokens(
                token_ids_0, token_ids_1 if pair else None
            )
        )

    @add_end_docstrings(ENCODE_KWARGS_DOCSTRING, ENCODE_ADDITIONAL_KWARGS_DOCSTRING)
    def _batch_prepare_for_model(
        self,
        batch_ids_pairs: List[Union[PreTokenizedInputPair, Tuple[List[int], None]]],
        add_special_tokens: bool = True,
        padding_strategy: PaddingStrategy = PaddingStrategy.DO_NOT_PAD,
        truncation_strategy: TruncationStrategy = TruncationStrategy.DO_NOT_TRUNCATE,
        max_length: Optional[int] = None,
        stride: int = 0,
        pad_to_multiple_of: Optional[int] = None,
        return_tensors: Optional[str] = None,
        return_token_type_ids: Optional[bool] = None,
        return_attention_mask: Optional[bool] = None,
        return_overflowing_tokens: bool = False,
        return_special_tokens_mask: bool = False,
        return_length: bool = False,
        verbose: bool = True,
        split_special_tokens: bool = False,
    ) -> Mapping:
        """
        Prepares a sequence of input id, or a pair of sequences of inputs ids so that it can be used by the model. It
        adds special tokens, truncates sequences if overflowing while taking into account the special tokens and
        manages a moving window (with user defined stride) for overflowing tokens

        Args:
            batch_ids_pairs: list of tokenized input ids or input ids pairs
        """

        batch_outputs = {}
        for first_ids, second_ids in batch_ids_pairs:
            outputs = self.prepare_for_model(
                first_ids,
                second_ids,
                add_special_tokens=add_special_tokens,
                padding=PaddingStrategy.DO_NOT_PAD.value,  # we pad in batch afterward
                truncation=truncation_strategy.value,
                max_length=max_length,
                stride=stride,
                pad_to_multiple_of=None,  # we pad in batch afterward
                return_attention_mask=False,  # we pad in batch afterward
                return_token_type_ids=return_token_type_ids,
                return_overflowing_tokens=return_overflowing_tokens,
                return_special_tokens_mask=return_special_tokens_mask,
                return_length=return_length,
                return_tensors=None,  # We convert the whole batch to tensors at the end
                prepend_batch_axis=False,
                verbose=verbose,
                split_special_tokens=split_special_tokens,
            )

            for key, value in outputs.items():
                if key not in batch_outputs:
                    batch_outputs[key] = []
                batch_outputs[key].append(value)

        batch_outputs = self.pad(
            batch_outputs,
            padding=padding_strategy.value,
            max_length=max_length,
            pad_to_multiple_of=pad_to_multiple_of,
            return_attention_mask=return_attention_mask,
        )

        tensor_converter = get_converter()
        batch_outputs = tensor_converter(batch_outputs, tensor_type=return_tensors)

        return

    @add_end_docstrings(ENCODE_KWARGS_DOCSTRING, ENCODE_ADDITIONAL_KWARGS_DOCSTRING)
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
        return_length: bool = False,
        verbose: bool = True,
        prepend_batch_axis: bool = False,
        **kwargs,
    ) -> Mapping:
        """
        Prepares a sequence of input id, or a pair of sequences of inputs ids so that it can be used by the model. It
        adds special tokens, truncates sequences if overflowing while taking into account the special tokens and
        manages a moving window (with user defined stride) for overflowing tokens. Please Note, for *pair_ids*
        different than ``None`` and *truncation_strategy = longest_first* or ``True``, it is not possible to return
        overflowing tokens. Such a combination of arguments will raise an error.

        Args:
            ids (``List[int]``):
                Tokenized input ids of the first sequence. Can be obtained from a string by the :meth:`text2ids`.
            pair_ids (``List[int]``, *optional*):
                Tokenized input ids of the second sequence. Can be obtained from a string by the :meth:`text2ids`.
        """

        (
            padding_strategy,
            truncation_strategy,
            max_length,
            kwargs,
        ) = self._get_padding_truncation_strategies(
            padding=padding,
            truncation=truncation,
            max_length=max_length,
            pad_to_multiple_of=pad_to_multiple_of,
            verbose=verbose,
            **kwargs,
        )

        pair = bool(pair_ids is not None)
        len_ids = len(ids)
        len_pair_ids = len(pair_ids) if pair else 0

        if return_token_type_ids and not add_special_tokens:
            raise ValueError(
                "Asking to return token_type_ids while setting add_special_tokens to False "
                "results in an undefined behavior. Please set add_special_tokens to True or "
                "set return_token_type_ids to None."
            )

        if (
            return_overflowing_tokens
            and truncation_strategy == TruncationStrategy.LONGEST_FIRST
            and pair_ids is not None
        ):
            raise ValueError(
                "Not possible to return overflowing tokens for pair of sequences with the "
                "``longest_first``. Please select another truncation strategy than ``longest_first``, "
                "for instance ``only_second`` or ``only_first``."
            )

        # Load from model defaults
        if return_token_type_ids is None:
            return_token_type_ids = "token_type_ids" in self.model_input_names
        if return_attention_mask is None:
            return_attention_mask = "attention_mask" in self.model_input_names

        encoded_inputs = {}

        # Compute the total size of the returned encodings
        total_len = (
            len_ids
            + len_pair_ids
            + (self.num_special_tokens_to_add(pair=pair) if add_special_tokens else 0)
        )

        # Truncation: Handle max sequence length
        overflowing_tokens = []
        if (
            truncation_strategy != TruncationStrategy.DO_NOT_TRUNCATE
            and max_length
            and total_len > max_length
        ):
            ids, pair_ids, overflowing_tokens = self.truncate_sequences(
                ids,
                pair_ids=pair_ids,
                num_tokens_to_remove=total_len - max_length,
                truncation_strategy=truncation_strategy,
                stride=stride,
            )

        if return_overflowing_tokens:
            encoded_inputs["overflowing_tokens"] = overflowing_tokens
            encoded_inputs["num_truncated_tokens"] = total_len - max_length

        # Add special tokens
        if add_special_tokens:
            sequence = self.build_inputs_with_special_tokens(ids, pair_ids)
            token_type_ids = self.create_token_type_ids_from_sequences(ids, pair_ids)
        else:
            sequence = ids + pair_ids if pair else ids
            token_type_ids = [0] * len(ids) + ([0] * len(pair_ids) if pair else [])

        # Build output dictionary
        encoded_inputs["input_ids"] = sequence
        if return_token_type_ids:
            encoded_inputs["token_type_ids"] = token_type_ids
        if return_special_tokens_mask:
            if add_special_tokens:
                encoded_inputs["special_tokens_mask"] = self.get_special_tokens_mask(
                    ids, pair_ids
                )
            else:
                encoded_inputs["special_tokens_mask"] = [0] * len(sequence)

        # Check lengths
        self._eventual_warn_about_too_long_sequence(
            encoded_inputs["input_ids"], max_length, verbose
        )

        # Padding
        if padding_strategy != PaddingStrategy.DO_NOT_PAD or return_attention_mask:
            encoded_inputs = self.pad(
                encoded_inputs,
                max_length=max_length,
                padding=padding_strategy.value,
                pad_to_multiple_of=pad_to_multiple_of,
                return_attention_mask=return_attention_mask,
            )

        if return_length:
            encoded_inputs["length"] = len(encoded_inputs["input_ids"])

        tensor_converter = get_converter()
        batch_outputs = tensor_converter(
            encoded_inputs,
            tensor_type=return_tensors,
            prepend_batch_axis=prepend_batch_axis,
        )

        return batch_outputs

    def truncate_sequences(
        self,
        ids: List[int],
        pair_ids: Optional[List[int]] = None,
        num_tokens_to_remove: int = 0,
        truncation_strategy: Union[str, TruncationStrategy] = "longest_first",
        stride: int = 0,
    ) -> Tuple[List[int], List[int], List[int]]:
        """
        Truncates a sequence pair in-place following the strategy.

        Args:
            ids (``List[int]``):
                Tokenized input ids of the first sequence. Can be obtained from a string by the :meth:``text2ids``.
            pair_ids (``List[int]``, *optional*):
                Tokenized input ids of the second sequence. Can be obtained from a string by the :meth:``text2ids``.
            num_tokens_to_remove (``int``, *optional*, defaults to 0):
                Number of tokens to remove using the truncation strategy.
            truncation_strategy (``str`` or :class:`~egrecho.core.tokenizer.TruncationStrategy`, *optional*, defaults to ``False``):
                The strategy to follow for truncation. Can be:

            - ``'longest_first'``: Truncate to a maximum length specified with the argument ``max_length`` or to the
                maximum acceptable input length for the model if that argument is not provided. This will truncate
                token by token, removing a token from the longest sequence in the pair if a pair of sequences (or a
                batch of pairs) is provided.
            - ``'only_first'``: Truncate to a maximum length specified with the argument ``max_length`` or to the
                maximum acceptable input length for the model if that argument is not provided. This will only
                truncate the first sequence of a pair if a pair of sequences (or a batch of pairs) is provided.
            - ``'only_second'``: Truncate to a maximum length specified with the argument ``max_length`` or to the
                maximum acceptable input length for the model if that argument is not provided. This will only
                truncate the second sequence of a pair if a pair of sequences (or a batch of pairs) is provided.
            - ``'do_not_truncate'`` (default): No truncation (i.e., can output batch with sequence lengths greater
                than the model maximum admissible input size).

            stride (``int``, *optional*, defaults to 0):
                If set to a positive number, the overflowing tokens returned will contain some tokens from the main
                sequence returned. The value of this argument defines the number of additional tokens.

        Returns:
            ``Tuple[List[int], List[int], List[int]]``: The truncated ``ids``, the truncated ``pair_ids`` and the list of
            overflowing tokens. Note: The *longest_first* strategy returns empty list of overflowing tokens if a pair
            of sequences (or a batch of pairs) is provided.
        """
        if num_tokens_to_remove <= 0:
            return ids, pair_ids, []

        if not isinstance(truncation_strategy, TruncationStrategy):
            truncation_strategy = TruncationStrategy(truncation_strategy)

        overflowing_tokens = []
        if truncation_strategy == TruncationStrategy.ONLY_FIRST or (
            truncation_strategy == TruncationStrategy.LONGEST_FIRST and pair_ids is None
        ):
            if len(ids) > num_tokens_to_remove:
                window_len = min(len(ids), stride + num_tokens_to_remove)
                if self.truncation_side == "left":
                    overflowing_tokens = ids[:window_len]
                    ids = ids[num_tokens_to_remove:]
                elif self.truncation_side == "right":
                    overflowing_tokens = ids[-window_len:]
                    ids = ids[:-num_tokens_to_remove]
                else:
                    raise ValueError(
                        f"invalid truncation strategy: {self.truncation_side}, use 'left' or 'right'."
                    )

            else:
                error_msg = (
                    f"We need to remove {num_tokens_to_remove} to truncate the input "
                    f"but the first sequence has a length {len(ids)}. "
                )
                if truncation_strategy == TruncationStrategy.ONLY_FIRST:
                    error_msg = (
                        error_msg + "Please select another truncation strategy than "
                        f"{truncation_strategy}, for instance 'longest_first' or 'only_second'."
                    )
                logger.error(error_msg)
        elif truncation_strategy == TruncationStrategy.LONGEST_FIRST:
            logger.warning(
                "Be aware, overflowing tokens are not returned for the setting you have chosen,"
                f" i.e. sequence pairs with the '{TruncationStrategy.LONGEST_FIRST.value}' "
                "truncation strategy. So the returned list will always be empty even if some "
                "tokens have been removed."
            )
            len_pair_ids = len(pair_ids) if pair_ids is not None else 0
            len_ids = len(ids)
            first_remove = min(abs(len_pair_ids - len_ids), num_tokens_to_remove)
            second_remove = num_tokens_to_remove - first_remove
            if len_ids > len_pair_ids:
                ids_to_move = first_remove + second_remove // 2
                pair_ids_to_move = second_remove - second_remove // 2
            else:
                ids_to_move = second_remove // 2
                pair_ids_to_move = first_remove + second_remove - (second_remove // 2)

            if self.truncation_side == "right":
                ids = ids[:-ids_to_move] if ids_to_move > 0 else ids
                pair_ids = (
                    pair_ids[:-pair_ids_to_move]
                    if pair_ids is not None and pair_ids_to_move > 0
                    else pair_ids
                )
            elif self.truncation_side == "left":
                ids = ids[ids_to_move:]
                pair_ids = pair_ids[pair_ids_to_move:] if pair_ids is not None else None
            else:
                raise ValueError(f"invalid truncation strategy:{self.truncation_side}")

        elif (
            truncation_strategy == TruncationStrategy.ONLY_SECOND
            and pair_ids is not None
        ):
            if len(pair_ids) > num_tokens_to_remove:
                window_len = min(len(pair_ids), stride + num_tokens_to_remove)
                if self.truncation_side == "right":
                    overflowing_tokens = pair_ids[-window_len:]
                    pair_ids = pair_ids[:-num_tokens_to_remove]
                elif self.truncation_side == "left":
                    overflowing_tokens = pair_ids[:window_len]
                    pair_ids = pair_ids[num_tokens_to_remove:]
                else:
                    raise ValueError(
                        f"invalid truncation strategy:{self.truncation_side}"
                    )
            else:
                logger.error(
                    f"We need to remove {num_tokens_to_remove} to truncate the input "
                    f"but the second sequence has a length {len(pair_ids)}. "
                    f"Please select another truncation strategy than {truncation_strategy}, "
                    "for instance 'longest_first' or 'only_first'."
                )

        return (ids, pair_ids, overflowing_tokens)

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
            return token_ids_0
        return token_ids_0 + token_ids_1

    def create_token_type_ids_from_sequences(
        self, token_ids_0: List[int], token_ids_1: Optional[List[int]] = None
    ) -> List[int]:
        """
        Create the token type IDs corresponding to the sequences passed.
        Should be overridden in a subclass if the model has a special way of building those.

        Args:
            token_ids_0 (``List[int]``): The first tokenized sequence.
            token_ids_1 (``List[int]``, *optional*): The second tokenized sequence.

        Returns:
            ``List[int]``: The token type ids.
        """
        if token_ids_1 is None:
            return len(token_ids_0) * [0]
        return [0] * len(token_ids_0) + [1] * len(token_ids_1)

    def get_special_tokens_mask(
        self,
        token_ids_0: List[int],
        token_ids_1: Optional[List[int]] = None,
        already_has_special_tokens: bool = False,
    ) -> List[int]:
        """
        Retrieves sequence ids from a token list that has no special tokens added. This method is called when adding
        special tokens using the tokenizer `prepare_for_model`.

        Args:
            token_ids_0 (``List[int]``):
                List of ids of the first sequence.
            token_ids_1 (``List[int]``, *optional*):
                List of ids of the second sequence.
            already_has_special_tokens (``bool``, *optional*, defaults to ``False``):
                Whether or not the token list is already formatted with special tokens for the model.

        Returns:
            A list of integers in the range [0, 1]: 1 for a special token, 0 for a sequence token.
        """
        assert already_has_special_tokens and token_ids_1 is None, (
            "You cannot use ``already_has_special_tokens=False`` with this tokenizer. "
            "Please use a slow (full python) tokenizer to activate this argument. "
            "Or set ``return_special_tokens_mask=True`` when calling the encoding method "
            "to get the special tokens mask in any tokenizer. "
        )

        all_special_ids = self.all_special_ids  # cache the property

        special_tokens_mask = [
            1 if token in all_special_ids else 0 for token in token_ids_0
        ]

        return special_tokens_mask

    def pad(
        self,
        encoded_inputs: Union[
            UserDict,
            List[UserDict],
            Dict[str, EncodedInput],
            Dict[str, List[EncodedInput]],
            List[Dict[str, EncodedInput]],
        ],
        padding: Union[bool, str, PaddingStrategy] = True,
        max_length: Optional[int] = None,
        pad_to_multiple_of: Optional[int] = None,
        return_attention_mask: Optional[bool] = None,
        return_tensors: Optional[Union[str, TensorType]] = None,
        verbose: bool = True,
    ) -> UserDict:
        """
        Pad a single encoded input or a batch of encoded inputs up to predefined length or to the max sequence length
        in the batch.

        Padding side (left/right) padding token ids are defined at the tokenizer level (with ``self.padding_side``,
        ``self.pad_token_id`` and ``self.pad_token_type_id``).

        Please note that with a fast tokenizer, using the ``__call__`` method is faster than using a method to encode the
        text followed by a call to the ``pad`` method to get a padded encoding.

        NOTE:
            If the ``encoded_inputs`` passed are dictionary of numpy arrays or PyTorch tensors, the
            result will use the same type unless you provide a different tensor type with ``return_tensors``. In the case of
            PyTorch tensors, you will lose the specific device of your tensors however.

        Args:
            encoded_inputs ([``BatchEncoding``], list of [``BatchEncoding``], ``Dict[str, List[int]]``, ``Dict[str, List[List[int]]`` or ``List[Dict[str, List[int]]]``):
                Tokenized inputs. Can represent one input ([``BatchEncoding``] or ``Dict[str, List[int]]``) or a batch of
                tokenized inputs (list of [``BatchEncoding``], *Dict[str, List[List[int]]]* or *List[Dict[str,
                List[int]]]*) so you can use this method during preprocessing as well as in a PyTorch Dataloader
                collate function.

                Instead of ``List[int]`` you can have tensors (numpy arrays, PyTorch tensors), see
                the note above for the return type.
            padding (``bool``, ``str`` or :class:`~egrecho.utils.types.PaddingStrategy`, *optional*, defaults to ``True``):
                Select a strategy to pad the returned sequences (according to the model's padding side and padding
                index) among:

            - ``True`` or ``'longest'``: Pad to the longest sequence in the batch (or no padding if only a single
                sequence if provided).
            - ``'max_length'``: Pad to a maximum length specified with the argument ``max_length`` or to the maximum
                acceptable input length for the model if that argument is not provided.
            - ``False`` or ``'do_not_pad'`` (default): No padding (i.e., can output a batch with sequences of different
                lengths).

            max_length (``int``, *optional*):
                Maximum length of the returned list and optionally padding length (see above).
            pad_to_multiple_of (``int``, *optional*):
                If set will pad the sequence to a multiple of the provided value.

                This is especially useful to enable the use of Tensor Cores on NVIDIA hardware with compute capability
                ``>= 7.5`` (Volta).
            return_attention_mask (``bool``, *optional*):
                Whether to return the attention mask. If left to the default, will return the attention mask according
                to the specific tokenizer's default, defined by the ``return_outputs`` attribute.

            return_tensors (``str`` or :class:`TensorType`, *optional*):
                If set, will return tensors instead of list of python integers. Acceptable values are:

            - ``'pt'``: Return PyTorch ``torch.Tensor`` objects.
            - ``'np'``: Return Numpy ``np.ndarray`` objects.

            verbose (``bool``, *optional*, defaults to ``True``):
                Whether or not to print more information and warnings.
        """
        tensor_converter = get_converter()

        # If we have a list of dicts, let's convert it in a dict of lists
        # We do this to allow using this method as a collate_fn function in PyTorch Dataloader
        if isinstance(encoded_inputs, (list, tuple)) and isinstance(
            encoded_inputs[0], Mapping
        ):
            encoded_inputs = {
                key: [example[key] for example in encoded_inputs]
                for key in encoded_inputs[0].keys()
            }

        # The model's main input name, usually ``input_ids``, has be passed for padding
        if self.model_input_names[0] not in encoded_inputs:
            raise ValueError(
                "You should supply an encoding or a list of encodings to this method "
                f"that includes {self.model_input_names[0]}, but you provided {list(encoded_inputs.keys())}"
            )

        required_input = encoded_inputs[self.model_input_names[0]]

        if required_input is None or (
            isinstance(required_input, Sized) and len(required_input) == 0
        ):
            if return_attention_mask:
                encoded_inputs["attention_mask"] = []
            return encoded_inputs

        # If we have PyTorch/TF/NumPy tensors/arrays as inputs, we cast them as python objects
        # and rebuild them afterwards if no return_tensors is specified
        # Note that we lose the specific device the tensor may be on for PyTorch

        first_element = required_input[0]
        if isinstance(first_element, (list, tuple)):
            # first_element might be an empty list/tuple in some edge cases so we grab the first non empty element.
            for item in required_input:
                if len(item) != 0:
                    first_element = item[0]
                    break
        # At this state, if ``first_element`` is still a list/tuple, it's an empty one so there is nothing to do.
        if not isinstance(first_element, (int, list, tuple)):
            if is_torch_tensor(first_element):
                return_tensors = "pt" if return_tensors is None else return_tensors
            elif isinstance(first_element, np.ndarray):
                return_tensors = "np" if return_tensors is None else return_tensors
            else:
                raise ValueError(
                    f"type of {first_element} unknown: {type(first_element)}. "
                    "Should be one of a python, numpy, pytorch object."
                )

            for key, value in encoded_inputs.items():
                encoded_inputs[key] = to_py_obj(value)

        # Convert padding_strategy in PaddingStrategy
        padding_strategy, _, max_length, _ = self._get_padding_truncation_strategies(
            padding=padding, max_length=max_length, verbose=verbose
        )

        required_input = encoded_inputs[self.model_input_names[0]]
        if required_input and not isinstance(required_input[0], (list, tuple)):
            encoded_inputs = self._pad(
                encoded_inputs,
                max_length=max_length,
                padding_strategy=padding_strategy,
                pad_to_multiple_of=pad_to_multiple_of,
                return_attention_mask=return_attention_mask,
            )
            return tensor_converter(encoded_inputs, tensor_type=return_tensors)

        batch_size = len(required_input)
        assert all(
            len(v) == batch_size for v in encoded_inputs.values()
        ), "Some items in the output dictionary have a different batch size than others."

        if padding_strategy == PaddingStrategy.LONGEST:
            max_length = max(len(inputs) for inputs in required_input)
            padding_strategy = PaddingStrategy.MAX_LENGTH

        batch_outputs = {}
        for i in range(batch_size):
            inputs = {k: v[i] for k, v in encoded_inputs.items()}
            outputs = self._pad(
                inputs,
                max_length=max_length,
                padding_strategy=padding_strategy,
                pad_to_multiple_of=pad_to_multiple_of,
                return_attention_mask=return_attention_mask,
            )

            for key, value in outputs.items():
                if key not in batch_outputs:
                    batch_outputs[key] = []
                batch_outputs[key].append(value)

        return tensor_converter(batch_outputs, tensor_type=return_tensors)

    def _pad(
        self,
        encoded_inputs: Union[Dict[str, EncodedInput], UserDict],
        max_length: Optional[int] = None,
        padding_strategy: PaddingStrategy = PaddingStrategy.DO_NOT_PAD,
        pad_to_multiple_of: Optional[int] = None,
        return_attention_mask: Optional[bool] = None,
    ) -> dict:
        """
        Pad encoded inputs (on left/right and up to predefined length or max length in the batch)

        Args:
            encoded_inputs:
                Dictionary of tokenized inputs (``List[int]``) or batch of tokenized inputs (``List[List[int]]``).
            max_length: maximum length of the returned list and optionally padding length (see below).
                Will truncate by taking into account the special tokens.
            padding_strategy:
                PaddingStrategy to use for padding.

            - PaddingStrategy.LONGEST Pad to the longest sequence in the batch
            - PaddingStrategy.MAX_LENGTH: Pad to the max length (default)
            - PaddingStrategy.DO_NOT_PAD: Do not pad

                The tokenizer padding sides are defined in self.padding_side:

            - 'left': pads on the left of the sequences
            - 'right': pads on the right of the sequences

            pad_to_multiple_of: (optional) Integer if set will pad the sequence to a multiple of the provided value.
                This is especially useful to enable the use of Tensor Core on NVIDIA hardware with compute capability
                ``>= 7.5`` (Volta).
            return_attention_mask:
                (optional) Set to False to avoid returning attention mask (default: set to model specifics)
        """
        # Load from model defaults
        if return_attention_mask is None:
            return_attention_mask = "attention_mask" in self.model_input_names

        required_input = encoded_inputs[self.model_input_names[0]]

        if padding_strategy == PaddingStrategy.LONGEST:
            max_length = len(required_input)

        if (
            max_length is not None
            and pad_to_multiple_of is not None
            and (max_length % pad_to_multiple_of != 0)
        ):
            max_length = ((max_length // pad_to_multiple_of) + 1) * pad_to_multiple_of

        needs_to_be_padded = (
            padding_strategy != PaddingStrategy.DO_NOT_PAD
            and len(required_input) != max_length
        )

        # Initialize attention mask if not present.
        if return_attention_mask and "attention_mask" not in encoded_inputs:
            encoded_inputs["attention_mask"] = [1] * len(required_input)

        if needs_to_be_padded:
            difference = max_length - len(required_input)

            if self.padding_side == "right":
                if return_attention_mask:
                    encoded_inputs["attention_mask"] = (
                        encoded_inputs["attention_mask"] + [0] * difference
                    )
                if "token_type_ids" in encoded_inputs:
                    encoded_inputs["token_type_ids"] = (
                        encoded_inputs["token_type_ids"]
                        + [self.pad_token_type_id] * difference
                    )
                if "special_tokens_mask" in encoded_inputs:
                    encoded_inputs["special_tokens_mask"] = (
                        encoded_inputs["special_tokens_mask"] + [1] * difference
                    )
                encoded_inputs[self.model_input_names[0]] = (
                    required_input + [self.pad_token_id] * difference
                )
            elif self.padding_side == "left":
                if return_attention_mask:
                    encoded_inputs["attention_mask"] = [
                        0
                    ] * difference + encoded_inputs["attention_mask"]
                if "token_type_ids" in encoded_inputs:
                    encoded_inputs["token_type_ids"] = [
                        self.pad_token_type_id
                    ] * difference + encoded_inputs["token_type_ids"]
                if "special_tokens_mask" in encoded_inputs:
                    encoded_inputs["special_tokens_mask"] = [
                        1
                    ] * difference + encoded_inputs["special_tokens_mask"]
                encoded_inputs[self.model_input_names[0]] = [
                    self.pad_token_id
                ] * difference + required_input
            else:
                raise ValueError(f"Invalid padding strategy:{self.padding_side}")

        return encoded_inputs

    def _eventual_warn_about_too_long_sequence(
        self, ids: List[int], max_length: Optional[int], verbose: bool
    ):
        """
        Depending on the input and internal state we might trigger a warning about a sequence that is too long for its
        corresponding model

        Args:
            ids (``List[str]``): The ids produced by the tokenization
            max_length (``int``, *optional*): The max_length desired (does not trigger a warning if it is set)
            verbose (``bool``): Whether or not to print more information and warnings.

        """
        if max_length is None and len(ids) > self.model_max_length and verbose:
            logger.warning_once(
                "Token indices sequence length is longer than the specified maximum sequence length "
                f"for this model ({len(ids)} > {self.model_max_length}). Running this sequence through the model "
                "will result in indexing errors"
            )

    def batch_decode(
        self,
        sequences: Union[List[int], List[List[int]], "np.ndarray", "torch.Tensor"],
        skip_special_tokens: bool = False,
        **kwargs,
    ) -> List[str]:
        """
        Convert a list of lists of token ids into a list of strings by calling decode.

        Args:
            sequences (``Union[List[int], List[List[int]], np.ndarray, torch.Tensor]``):
                List of tokenized input ids. Can be obtained using the :meth:`__call__` method.
            skip_special_tokens (``bool``, *optional*, defaults to ``False``):
                Whether or not to remove special tokens in the decoding.
            \**kwargs (additional keyword arguments, *optional*):
                Will be passed to the underlying model specific decode method.

        Returns:
            ``List[str]``: The list of decoded sentences.
        """
        return [
            self.decode(
                seq,
                skip_special_tokens=skip_special_tokens,
                **kwargs,
            )
            for seq in sequences
        ]

    def decode(
        self,
        token_ids: Union[int, List[int], "np.ndarray", "torch.Tensor"],
        skip_special_tokens: bool = False,
        **kwargs,
    ) -> str:
        """
        Converts a sequence of ids in a string, using the tokenizer and vocabulary with options to remove special
        tokens and clean up tokenization spaces.

        Similar to doing ``self.ids2text(token_ids)``.

        Args:
            token_ids (``Union[int, List[int], np.ndarray, torch.Tensor]``):
                List of tokenized input ids. Can be obtained using the :meth:`__call__` method.
            skip_special_tokens (``bool``, *optional*, defaults to ``False``):
                Whether or not to remove special tokens in the decoding.
            \**kwargs (additional keyword arguments, *optional*):
                Will be passed to the underlying model specific decode method.

        Returns:
            ``str``: The decoded sentence.
        """
        # Convert inputs to python lists
        token_ids = to_py_obj(token_ids)

        return self._decode(
            token_ids=token_ids,
            skip_special_tokens=skip_special_tokens,
            **kwargs,
        )

    def _decode(
        self,
        token_ids: Union[int, List[int]],
        skip_special_tokens: bool = False,
        clean_up_tokenization_spaces: bool = None,
        **kwargs,
    ) -> str:
        raise NotImplementedError

    def __repr__(self) -> str:
        return (
            f"{self.__class__.__name__}(config={self.config} \n\tvocab_size={self.vocab_size}"
            f" padding_side={self.padding_side}, truncation_side={self.truncation_side})"
        )


@lru_cache
def get_converter():
    return (
        transformers.tokenization_utils_base.BatchEncoding
        if _TRANSFORMERS_AVAILABLE
        else convert_to_tensors
    )


def convert_to_tensors(
    encoded_inputs,
    tensor_type: Optional[Union[str, TensorType]] = None,
    prepend_batch_axis: bool = False,
):
    """
    Convert the inner content of a dict to tensors.

    Args:
        encoded_inputs (Union[Dict[str, EncodedInput], UserDict]):
            encoded inputs.
        tensor_type (``str`` or :class:`TensorType`, *optional*):
            The type of tensors to use. If ``str``, should be one of the values of the enum :class:`TensorType`. If
            ``None``, no modification is done.
        prepend_batch_axis (``int``, *optional*, defaults to ``False``):
            Whether or not to add the batch dimension during the conversion.
    """
    assert isinstance(encoded_inputs, Mapping), type(encoded_inputs)

    if tensor_type is None:
        return encoded_inputs

    # Convert to TensorType
    if not isinstance(tensor_type, TensorType):
        tensor_type = TensorType(tensor_type)

    # Get a function reference for the correct framework
    if tensor_type == TensorType.PYTORCH:

        import torch

        is_tensor = torch.is_tensor

        def as_tensor(value, dtype=None):
            if isinstance(value, list) and isinstance(value[0], np.ndarray):
                return torch.tensor(np.array(value))
            return torch.tensor(value)

    else:

        def as_tensor(value, dtype=None):
            if isinstance(value, (list, tuple)) and isinstance(
                value[0], (list, tuple, np.ndarray)
            ):
                value_lens = [len(val) for val in value]
                if len(set(value_lens)) > 1 and dtype is None:
                    # we have a ragged list so handle explicitly
                    value = as_tensor([np.asarray(val) for val in value], dtype=object)
            return np.asarray(value, dtype=dtype)

        is_tensor = is_numpy_array

    # Do the tensor conversion in batch
    for key, value in encoded_inputs.items():
        try:
            if prepend_batch_axis:
                value = [value]

            if not is_tensor(value):
                tensor = as_tensor(value)

                # Removing this for now in favor of controlling the shape with ``prepend_batch_axis``
                # # at-least2d
                # if tensor.ndim > 2:
                #     tensor = tensor.squeeze(0)
                # elif tensor.ndim < 2:
                #     tensor = tensor[None, :]

                encoded_inputs[key] = tensor
        except Exception as e:
            if key == "overflowing_tokens":
                raise ValueError(
                    "Unable to create tensor returning overflowing tokens of different lengths. "
                    "Please see if a fast version of this tokenizer is available to have this feature available."
                ) from e
            raise ValueError(
                "Unable to create tensor, you should probably activate truncation and/or padding with"
                " 'padding=True' 'truncation=True' to have batched tensors with the same length. Perhaps your"
                f" features (``{key}`` in this case) have excessive nesting (inputs type ``list`` where type ``int`` is"
                " expected)."
            ) from e

    return encoded_inputs
