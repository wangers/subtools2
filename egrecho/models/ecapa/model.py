# -*- coding:utf-8 -*-
# Copyright xmuspeech (Author: Leo 2023-08)

from typing import Optional, Union

import torch

from egrecho.models.architecture.speaker import XvectorMixin, XvectorOutput
from egrecho.models.ecapa.ecapa_config import EcapaSVConfig
from egrecho.models.ecapa.ecapa_xvector import EcapaXvector
from egrecho.nn.head import MarginHead


class EcapaModel(XvectorMixin):
    """
    Ecapa model with teacher for training.

    Args:
        config (dict, EcapaConifg):
            default configuration to initiate model.
        inputs_dim (int):
            expose to link with datamodule.
        num_classes (int):
            expose to link with datamodule.
    """

    main_input_name = ["input_features"]

    def __init__(
        self,
        config: Union[EcapaSVConfig, dict] = None,
        inputs_dim: Optional[int] = None,
        num_classes: Optional[int] = None,
    ) -> None:
        self.config = EcapaSVConfig.from_config(config=config)
        if inputs_dim is not None:
            self.config.inputs_dim = inputs_dim
        if num_classes is not None:
            self.config.num_classes = num_classes
        super().__init__()
        # save_hyperparameters can't handle dataclass.
        config = self.config.to_dict()
        self.save_hyperparameters("config")

        self.ecapa = EcapaXvector(self.config)
        if self.config.head_name:
            self.head = MarginHead.from_name(
                self.config.head_name,
                self.config.embd_dim,
                self.config.num_classes,
                **self.config.head_params
            )

        self.example_input_array = {
            "input_features": torch.randn(2, 200, self.config.inputs_dim)
        }

    def forward(self, input_features):
        """
        Backbone forward.

        Args:
            input_features (Tensor):
                input tensor of shape (B, T, F)
            position (bool):
                some case with two embd layer, far can get the second-to-last embedding.

        """
        embd, embd_far = self.ecapa(input_features)

        return embd, embd_far

    def extract_embedding(
        self, input_features, max_chunk: int = 4000, position: str = "near"
    ) -> XvectorOutput:
        """
        Pipline interface.

        Args:
            input_features (Tensor):
                input tensor of shape (B, T, F)
            position (bool):
                some case with two embd layer, far can get the second-to-last embedding.
        """
        if position == "far" and self.ecapa.embd_layer_num == 1:
            raise RuntimeError(
                "Request embd in far positon, but got one embd layer related to near."
            )
        chunk_inputs, chunk_sizes = self.split_chunks(
            input_features, max_chunk=max_chunk
        )
        embedding_stats = torch.empty(0)
        for idx, chunk in enumerate(chunk_inputs):
            model_out = self(chunk)
            this_embedding = model_out[0] if position == "near" else model_out[1]
            if embedding_stats.size(0) == 0:
                embedding_stats = this_embedding * chunk_sizes[idx]
            else:
                embedding_stats += chunk_sizes[idx] * this_embedding
        embedding = embedding_stats / sum(chunk_sizes)

        return XvectorOutput(xvector=embedding)
