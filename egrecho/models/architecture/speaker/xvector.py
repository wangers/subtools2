# -*- coding:utf-8 -*-
# Copyright xmuspeech (Author: Leo 2023-08)

from dataclasses import dataclass
from typing import List, Optional, Tuple

import torch

from egrecho.core.module import TopVirtualModel
from egrecho.core.onnx_export import OnnxConfig, OnnxExportMixin
from egrecho.utils.types import ModelOutput


@dataclass
class XvectorOutput(ModelOutput):
    xvector: Optional[torch.FloatTensor] = None


class XvectorMixin(TopVirtualModel, OnnxExportMixin):
    """
    Base model for x-vector and provides interface for pipeline.

    you should implement two functions after inheriting this object.
        - :method::``forward(...)``: just like pytorch needed.
        - :method::``extract_embedding(...)``: needed if use pipeline.
    """

    def extract_embedding(
        self, *args, max_chunk: Optional[int] = None, **kwargs
    ) -> ModelOutput:
        """
        [Abstract] Extracts embeddings. This method is mainly to expose interface for pipeline forward format (e.g.,
        maybe you need a extra pre/process isolate to `forward` and a readiable dict-like result.)

        Note:
            - This method is used for inference to extract embeddings from the given input tensor.
            - Subclasses must implement this method to provide the actual embedding extraction logic.

        Example::

            model = YouModelClass()
            inputs = ...
            with torch.no_grad():
                results = model.extract_embedding(inputs)
        """
        raise NotImplementedError

    @staticmethod
    def split_chunks(
        x: torch.Tensor,
        max_chunk: Optional[int],
        even: bool = False,
    ) -> Tuple[List[torch.Tensor], List[int]]:
        """Split a tensor into a list of tensors with a specified `max_chunk` size.

        For `even=True` mode, chunk sizes will adapt to be equal. The chunk sizes might
        be slightly different from `max_chunk`.

        For `even=False`, the last chunk might be smaller. The last two chunk sizes will be combined and
        then split into two even chunks, meaning all chunk sizes are `max_chunk` except the last two.

        Args:
            x (torch.Tensor): The input tensor to be split.
            max_chunk (int): The maximum size of each chunk.
            even (bool, optional): Whether to split the tensor into even chunks.
                If False, the last chunk might be smaller. Default is False.

        Returns:
            Tuple[List[torch.Tensor], int]: A tuple containing the list of chunks and their chunk sizes.

        Example:
            >>> x = torch.rand((3, 99))  # A tensor with shape (3, 100)
            >>> chunks, chunk_sizes = XvectorMixin.split_chunks(x, max_chunk=30, even=True)
            >>> assert [chunk.shape[1] for chunk in chunks] == chunk_sizes == [25, 25, 25, 24]
            >>> chunks, chunk_sizes = XvectorMixin.split_chunks(x, max_chunk=30, even=False)
            >>> assert [chunk.shape[1] for chunk in chunks] == chunk_sizes == [30, 30, 20, 19]
        """

        assert x.ndim > 1
        total_length = x.shape[1]

        # Calculate chunk sizes
        if max_chunk is not None and max_chunk > 0:
            chunk_sizes = get_chunksize(
                total_len=total_length, chunk_size=max_chunk, even=even
            )

            # Adjust last two sizes if not even to avoid the last is too short.
            if not even and len(chunk_sizes) > 1:
                last_two_sizes = [chunk_sizes.pop() for _ in range(2)]
                last_two_len = sum(last_two_sizes)
                last_two_sizes = [last_two_len - (last_two_len // 2), last_two_len // 2]
                chunk_sizes.extend(last_two_sizes)
        else:
            chunk_sizes = [total_length]

        chunks = []
        offset = 0

        # Split the tensor into chunks
        for chunk_size in chunk_sizes:
            chunks.append(x[:, offset : offset + chunk_size])
            offset += chunk_size
        return chunks, chunk_sizes

    @classmethod
    def pipeline_out(cls, model_out: torch.Tensor) -> XvectorOutput:
        """Transform output to dict. overwrite it for your specify model."""
        return XvectorOutput(model_out)

    def onnx_sample(self):
        return self.example_input_array

    def get_onnx_config(self):
        return ComOnnxConfig()


class ComOnnxConfig(OnnxConfig):
    @property
    def inputs(self):
        """Return an ordered dict contains the model's input arguments name with their dynamic axis.

        Returns:
            For each output: its name associated to the axes symbolic name and the axis position within the tensor
        """
        return {'input_features': {0: 'B', 1: 'T'}}

    @property
    def outputs(self):
        """Return an ordered dict contains the model's output arguments name with their dynamic axis.

        Returns:
            For each output: its name associated to the axes symbolic name and the axis position within the tensor
        """
        return {'embs': {0: 'B'}}


def get_chunksize(
    total_len: int,
    chunk_size: int,
    even: bool = True,
) -> List[int]:
    """
    Generate a chunk sizes list.

    Args:
        total_len: int
            The lengths to be divided.
        chunk_size: int
        even: True
            If True, the max differ between chunksize is 1.

    Returns:
        List with chunk sizes.

    Example:
        >>> get_chunksize(15, chunk_size=10, even=True)
        [8, 7]
        >>> get_chunksize(15, chunk_size=4, even=False)
    """

    assert chunk_size > 0
    q, r = divmod(total_len, int(chunk_size))
    split_num = q + (1 if r > 0 else 0)
    if even:
        q, r = divmod(total_len, split_num)
        full_size = q + (1 if r > 0 else 0)
    else:
        full_size = int(chunk_size)

    if even:
        partial_size = full_size - 1
        num_full = total_len - partial_size * split_num
        num_partial = split_num - num_full
    else:
        partial_size = total_len - full_size * (split_num - 1)
        num_partial = 1 if partial_size > 0 else 0
        num_full = split_num - num_partial

    # yield full_size for expect time.
    chunk_sizes = [full_size for _ in range(num_full)]
    partial_size = [partial_size for _ in range(num_partial)]
    chunk_sizes.extend(partial_size)
    return chunk_sizes
