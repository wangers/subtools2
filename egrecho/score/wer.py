# -*- coding:utf-8 -*-
# (Author: Leo 2024-06)

import re
import warnings
from abc import ABC, abstractmethod
from dataclasses import dataclass
from functools import partial
from pathlib import Path
from typing import Generator, List, Optional, Union

from egrecho.utils.apply import apply_to_collection
from egrecho.utils.common import asdict_filt
from egrecho.utils.imports import _JIWER_AVAILABLE
from egrecho.utils.torch_utils import to_py_obj
from egrecho.utils.types import ModelOutput, StrEnum

if not _JIWER_AVAILABLE:
    raise ModuleNotFoundError(
        "wer metric requires that jiwer is installed. [ HINT ] `pip install jiwer`"
    )
import jiwer
from jiwer import CharacterOutput, WordOutput

JIWER_OUT = Union[CharacterOutput, WordOutput]

_SENTENCE_RE = re.compile(r'^sentence \d+\nREF', re.MULTILINE)


@dataclass
class TXTMetricOutput(ModelOutput):
    """Container holds outputs of text (wer/cer related) metrics.

    Args:
        error_rate: CER/WER
        total: Number of words overall references
        sub_rate: Substitutions error rate
        ins_rate: Insertions error rate
        del_rate: deletions error rate
    """

    error_rate: Optional[float]
    ins_rate: Optional[int] = None
    del_rate: Optional[int] = None
    sub_rate: Optional[int] = None
    total: Optional[int] = None

    def to_dict(self):
        data = apply_to_collection(self, float, partial(round, ndigits=3))
        data = asdict_filt(data, filt_type='none')
        return data


@dataclass
class WERCountOutput(ModelOutput):
    """Container holds outputs of text (wer/cer related) metric statics.

    Args:
        errors: Number of edit operations to get from the reference to the prediction, summed over all samples
        total: Number of words overall references
        substitutions: The number of substitutions required to transform hypothesis
                       sentences to reference sentences
        insertions: The number of insertions required to transform hypothesis
                       sentences to reference sentences
        deletions: The number of deletions required to transform hypothesis
                       sentences to reference sentences
        jiwer_out: output of package ``jiwer``. see: ``jiwer.process_characters``,``jiwer.process_characters``:
            `jitwer <https://github.com/jitsi/jiwer/blob/master/jiwer/process.py>`_.
    """

    errors: Optional[int] = None
    total: Optional[int] = None
    insertions: Optional[int] = None
    deletions: Optional[int] = None
    substitutions: Optional[int] = None
    jiwer_out: Optional[JIWER_OUT] = None

    @property
    def wer(self):
        return self.errors / self.total

    def visualize_alignment(
        self,
        show_measures: bool = True,
        skip_correct: bool = True,
    ):
        """
        Visualize the output of [jiwer.process_words][process.process_words] and
        [jiwer.process_characters][process.process_characters]. The visualization
        shows the alignment between each processed reference and hypothesis pair.
        If `show_measures=True`, the output string will also contain all measures in the
        output. Refs `jitwer
        <https://github.com/jitsi/jiwer/blob/master/jiwer/alignment.py>`_.

        Args:
            output: The processed output of reference and hypothesis pair(s).
            show_measures: If enabled, the visualization will include measures like the WER
                        or CER
            skip_correct: If enabled, the visualization will exclude correct reference and hypothesis pairs

        Returns:
            (str): The visualization as a string

        Example::

            import jiwer

            out = jiwer.process_words(
                ["short one here", "quite a bit of longer sentence"],
                ["shoe order one", "quite bit of an even longest sentence here"],
            )
            print(jiwer.visualize_alignment(out))

        .. code-block:: text

            will produce this visualization:
            sentence 1
            REF:    # short one here
            HYP: shoe order one    *
                    I     S        D

            sentence 2
            REF: quite a bit of  #    #  longer sentence    #
            HYP: quite * bit of an even longest sentence here
                       D         I    I       S             I

            number of sentences: 2
            substitutions=2 deletions=2 insertions=4 hits=5

            mer=61.54%
            wil=74.75%
            wip=25.25%
            wer=88.89%

        .. code-block:: text

            When ``show_measures=False``, only the alignment will be printed:
            sentence 1
            REF:    # short one here
            HYP: shoe order one    *
                    I     S        D

            sentence 2
            REF: quite a bit of  #    #  longer sentence    #
            HYP: quite * bit of an even longest sentence here
                       D         I    I       S             I
        """
        if self.jiwer_out is None:
            warnings.warn(
                f"output of ``jitwer`` not exists. [HINT] Try to use ``wer_update(..., details='all')``"
            )
        else:
            return jiwer.visualize_alignment(
                self.jiwer_out, show_measures=show_measures, skip_correct=skip_correct
            )

    @staticmethod
    def _gen_visualzes(visualizes: str) -> Generator:

        for block in _SENTENCE_RE.split(visualizes):
            if block.strip():
                yield "REF" + block.strip()

    def gen_alins(self, skip_correct=False) -> Generator:
        visualizes = self.visualize_alignment(
            show_measures=False, skip_correct=skip_correct
        )
        yield from self._gen_visualzes(visualizes)


class WerDetail(StrEnum):
    SIMPLE = 'simple'
    SUBCOUNTS = 'subcounts'
    ALL = 'all'


def wer_update(
    preds: Union[str, List[str]],
    target: Union[str, List[str]],
    use_cer: bool = False,
    details: Union[str, WerDetail] = WerDetail.SUBCOUNTS,
) -> WERCountOutput:
    """Update the wer score with the current set of references and predictions.

    Args:
        preds: Transcription(s) to score as a string or list of strings
        target: Reference(s) for each speech input as a string or list of strings
        use_cer: set True to enable cer
        details: defautls to ``WerDetail.SUBCOUNTS``

        - 'simple': only counts of errors and total.
        - 'subcounts': plus counts of ins, del, sub
        - 'all': plus outputs of package ``jiwer``. see: ``jiwer.process_characters``,``jiwer.process_characters``:
            `jitwer <https://github.com/jitsi/jiwer/blob/master/jiwer/process.py>`_.

    NOTE: jiwer does not allow empty string as target, care that case. When ``use_cer=True``, the space (` `) is
        included as a character, strip it outside if neccesarry.

    Example:
        >>> predictions = ["this is the prediction", "there is an other sample"]
        >>> references = ["this is the reference", "there is another one"]
        >>> w = wer_update(predictions, references)
        >>> w, w.wer
        TxtMetricOutput(errors=4, total=8, insertions=1, deletions=0, substitutions=3, jiwer_out=None), 0.5
        >>> c = wer_update(predictions, references, use_cer=True)
        >>> c, c.wer
        TxtMetricOutput(errors=14, total=41, insertions=5, deletions=0, substitutions=9, jiwer_out=None), 0.3414
        >>> assert sum(len(l) for l in references) == c.total
        ... # Now i want to get aligns
        >>> w = wer_update(predictions,references,details='all')
        >>> print(w.visualize_alignment())

    .. code-block:: text

        sentence 1
        REF: this is the  reference
        HYP: this is the prediction
                                  S

        sentence 2
        REF: there is ** another    one
        HYP: there is an   other sample
                       I       S      S

        number of sentences: 2
        substitutions=3 deletions=0 insertions=1 hits=5

        mer=44.44%
        wil=65.28%
        wip=34.72%
        wer=50.00%

    """

    if use_cer:
        jiwer_out: CharacterOutput = jiwer.process_characters(target, preds)
    else:
        jiwer_out: WordOutput = jiwer.process_words(target, preds)
    S, D, I, H = (
        jiwer_out.substitutions,
        jiwer_out.deletions,
        jiwer_out.insertions,
        jiwer_out.hits,
    )
    total = H + S + D
    errors = S + D + I
    if details == WerDetail.SIMPLE:
        return WERCountOutput(errors=errors, total=total)
    elif details == WerDetail.SUBCOUNTS:
        return WERCountOutput(
            errors=errors, total=total, insertions=I, substitutions=S, deletions=D
        )
    elif details == WerDetail.ALL:
        return WERCountOutput(
            errors=errors,
            total=total,
            insertions=I,
            substitutions=S,
            deletions=D,
            jiwer_out=jiwer_out,
        )
    else:
        raise ValueError(
            f'Invalid detail mode {details}. [HINT] Select from {WerDetail._member_map_}.'
        )


def wer_compute(errors, total):
    """Compute the word error rate.

    Args:
        errors: Number of edit operations to get from the reference to the prediction, summed over all samples
        total: Number of words overall references

    Returns:
        Word error rate score

    """
    return errors / total


class WERMixin(ABC):
    r"""Base class for text cer/wer metric. Core methods:

    - :meth:`update` (**must be implemented**):
        Accumulates metric-related statics (errors, total words, etc.).
    - :meth:`compute` (**must be implemented**):
        Computes cer/wer here.

    Class attributes (overridden by derived classes):

        - **use_cer** (``bool``) -- True to enable cer.
        - **details** (``str``) -- :class:``WerDetail.SUBCOUNTS``.

    Example::

        from egrecho.score.wer import DetailWER

        # organize 2 batches
        preds_batch1 = ["this is the prediction", "there is an other sample"]
        refs_batch1 = ["this is the reference", "there is another one"]
        preds_batch2 = ["hello duck", "i like python"]
        refs_batch2 = ["hello world", "i like monthy python"]
        preds = [preds_batch1, preds_batch2]
        refs = [refs_batch1, refs_batch2]

        # set details='all' to get align ref/pred pair
        metric = DetailWER(details='all')

        # record preds if necessary
        with DetailWER.open_writer('recogs.txt') as writer:
            for pre, ref in zip(preds, refs):
                metric_outs = metric.update(pre, ref)
                for box in metric_outs.gen_alins():
                    writer.write(box)
        with open('recogs.txt') as fr:
            recogs = fr.read()
        metric_rs = metric.compute()
        print(metric_rs.to_dict())
        print(recogs)

    .. code-block:: text

        Here is the metrics:

        {'error_rate': 0.429,
        'ins_rate': 0.071,
        'del_rate': 0.071,
        'sub_rate': 0.286,
        'total': 14}

        Here is the writed contents:

        REF: this is the  reference
        HYP: this is the prediction
                                  S
        --------------------|boxend|--------------------
        REF: there is ** another    one
        HYP: there is an   other sample
                       I       S      S
        --------------------|boxend|--------------------
        REF: hello world
        HYP: hello  duck
                       S
        --------------------|boxend|--------------------
        REF: i like monthy python
        HYP: i like ****** python
                         D
        --------------------|boxend|--------------------

    """

    use_cer: bool
    details: Union[str, WerDetail]

    @abstractmethod
    def update(self):
        raise NotImplementedError

    @abstractmethod
    def compute(self):
        raise NotImplementedError

    def reset(self):
        raise NotImplementedError

    def _update(
        self,
        preds: Union[str, List[str]],
        target: Union[str, List[str]],
        use_cer: Optional[bool] = None,
        details: Optional[Union[str, WerDetail]] = None,
    ):
        """
        Updates metric state.
        Args:
            preds: Transcription(s) to score as a string or list of strings
            target: Reference(s) for each speech input as a string or list of strings
            use_cer: set True to enable cer
            details: defautls to ``WerDetail.SUBCOUNTS``
        """
        use_cer = use_cer or self.use_cer
        details = details or self.details
        outputs = wer_update(preds, target, use_cer=use_cer, details=details)
        return outputs

    @classmethod
    def open_writer(cls, path: Union[str, Path], overwrite: bool = True, **kwargs):
        """
        Get file handler of text file to write wer alignment. Use it in ``with`` context like ``open``.

        Example::

            from egrecho.score.wer import SimpleWER
            predictions = ["this is the prediction", "there is an other sample"]
            references = ["this is the reference", "there is another one"]
            metric = SimpleWER(details='all')
            metric_outs = metric.update(predictions, references)
            with SimpleWER.open_writer('recogs.txt') as writer:
                for box in metric_outs.gen_alins():
                    writer.write(box)
            with open('recogs.txt') as fr:
                rs = fr.read()
            print(rs)

        .. code-block:: text

            Here is the writed contents:

            REF: this is the  reference
            HYP: this is the prediction
                                      S
            --------------------|boxend|--------------------
            REF: there is ** another    one
            HYP: there is an   other sample
                           I       S      S
            --------------------|boxend|--------------------

        """
        from egrecho.utils.io.writer import TextBoxWriter

        return TextBoxWriter(path, overwrite=overwrite, **kwargs)


class SimpleWER(WERMixin):
    def __init__(
        self,
        use_cer: bool = False,
        details: Union[str, WerDetail] = WerDetail.SUBCOUNTS,
    ) -> None:
        self.use_cer = use_cer
        self.details = details

        self.errors = 0
        self.total = 0

    def update(
        self, preds: Union[str, List[str]], target: Union[str, List[str]], **kwargs
    ):
        outputs = self._update(preds, target, **kwargs)
        self.errors += outputs.errors
        self.total += outputs.total
        return outputs

    def compute(self) -> TXTMetricOutput:
        errors, total = to_py_obj(self.errors), to_py_obj(self.total)
        wer = 0.0 if errors == 0 else errors / total
        return TXTMetricOutput(error_rate=wer, total=total)


class DetailWER(WERMixin):
    def __init__(
        self,
        use_cer: bool = False,
        details: Union[str, WerDetail] = WerDetail.SUBCOUNTS,
    ) -> None:
        self.check_detail_lvl(details)
        self.use_cer = use_cer
        self.details = details

        self.errors = 0
        self.total = 0
        self.ins = 0
        self.dels = 0
        self.subs = 0

    def update(
        self, preds: Union[str, List[str]], target: Union[str, List[str]], **kwargs
    ):
        details = kwargs.pop('details', self.details)
        self.check_detail_lvl(details)
        outputs = self._update(preds, target, details=details, **kwargs)
        self.errors += outputs.errors
        self.total += outputs.total
        self.ins += outputs.insertions
        self.dels += outputs.deletions
        self.subs += outputs.substitutions
        return outputs

    def compute(
        self,
    ) -> TXTMetricOutput:

        detail_cnts = list(
            map(to_py_obj, [self.errors, self.ins, self.dels, self.subs, self.total])
        )
        errors, total = detail_cnts[0], detail_cnts[-1]
        if errors == 0:
            return TXTMetricOutput(
                error_rate=0.0, ins_rate=0.0, del_rate=0.0, sub_rate=0.0, total=total
            )

        wer, ins_rate, del_rate, sub_rate = (n / total for n in detail_cnts[:-1])
        return TXTMetricOutput(
            error_rate=wer,
            ins_rate=ins_rate,
            del_rate=del_rate,
            sub_rate=sub_rate,
            total=total,
        )

    @classmethod
    def check_detail_lvl(cls, details: Union[str, WerDetail] = WerDetail.SUBCOUNTS):
        if details == WerDetail.SIMPLE:
            raise ValueError(
                f"Unsupport details='simple' for {cls.__name__}, [HINT] choose from {set(WerDetail)-{'simple'}!r}"
            )
