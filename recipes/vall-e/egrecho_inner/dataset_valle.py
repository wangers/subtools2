# -*- coding:utf-8 -*-
# (Author: Leo 202406)

"""
modified from lhoste.dataset.speech_synthesis.py
"""

from typing import Callable, Dict, List

import torch
from lhotse.cut import CutSet
from lhotse.dataset.collation import read_features_from_cuts
from lhotse.dataset.speech_synthesis import validate_for_tts
from lhotse.utils import ifnone
from lhotse.workarounds import Hdf5MemoryIssueFix


class ValleDataset(torch.utils.data.Dataset):
    """
    The PyTorch Dataset for the speech synthesis(e.g. TTS) task.
    Each item in this dataset is a dict of:

    .. code-block::

        {
            'codes': List[Tensor] of len B
            'cut_ids': List[str] of len B
            'text': List[str] of len B  # when return_text=True
            'tokens': List[List[str]]  # when return_tokens=True
            'speakers': List[str] of len B  # when return_spk_ids=True
            'cut': List of Cuts  # when return_cuts=True
        }
    """

    def __init__(
        self,
        cut_transforms: List[Callable[[CutSet], CutSet]] = None,
        return_text: bool = True,
        return_tokens: bool = False,
        return_spk_ids: bool = False,
        return_cuts: bool = False,
    ) -> None:
        super().__init__()

        self.cut_transforms = ifnone(cut_transforms, [])

        self.return_text = return_text
        self.return_tokens = return_tokens
        self.return_spk_ids = return_spk_ids
        self.return_cuts = return_cuts

        # This attribute is a workaround to constantly growing HDF5 memory
        # throughout the epoch. It regularly closes open file handles to
        # reset the internal HDF5 caches.
        self.hdf5_fix = Hdf5MemoryIssueFix(reset_interval=100)

    def __getitem__(self, cuts: CutSet) -> Dict[str, torch.Tensor]:
        validate_for_tts(cuts)
        self.hdf5_fix.update()

        for transform in self.cut_transforms:
            cuts = transform(cuts)

        # Sort the cuts again after transforms
        cuts = cuts.sort_by_duration(ascending=False)

        features = read_features_from_cuts(cuts)
        batch = {}
        batch["codes"] = features
        batch["cut_ids"] = [cut.id for cut in cuts]
        if self.return_text:
            text = [cut.supervisions[0].text for cut in cuts]
            batch["text"] = text

        if self.return_tokens:
            tokens = [cut.tokens for cut in cuts]
            batch["tokens"] = tokens

        if self.return_spk_ids:
            batch["speakers"] = [cut.supervisions[0].speaker for cut in cuts]

        if self.return_cuts:
            batch["cut"] = [cut for cut in cuts]

        return batch


if __name__ == "__main__":
    from lhotse import load_manifest_lazy
    from tokenizer_utils import EncodecTokenExtractor

    extractor = EncodecTokenExtractor()
    path = "exp/egs/libritts/cuts_test.jsonl.gz"
    cuts: CutSet = load_manifest_lazy(path)
    ds = ValleDataset(return_tokens=True, return_cuts=True)
    cuts_s = cuts.subset(first=2)

    batch = ds[cuts_s]
    samples = batch["cut"][0].load_audio()
    codes = extractor.extract(samples, 24000)
    codes_ds = batch["codes"][0]
    codes = torch.from_numpy(codes)
    assert torch.allclose(codes, codes_ds)
