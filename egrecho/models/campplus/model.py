# -*- coding:utf-8 -*-
# Copyright xmuspeech (Author: Leo 2023-08)

from typing import Optional, Union

import torch

from egrecho.models.architecture.speaker import XvectorMixin, XvectorOutput
from egrecho.models.campplus.campplus import CamPP
from egrecho.models.campplus.campplus_config import CamPPSVConfig
from egrecho.nn.classifier import Classifier


class CamPPModel(XvectorMixin):
    """
    Cam++ model with teacher for training.

    Args:
        config (dict, CamPPSVConfig):
            default configuration to initiate model.
        inputs_dim (int):
            expose to link with datamodule.
        num_classes (int):
            expose to link with datamodule.
    """

    main_input_name = ["input_features"]

    def __init__(
        self,
        config: Union[CamPPSVConfig, dict] = None,
        inputs_dim: Optional[int] = None,
        num_classes: Optional[int] = None,
    ) -> None:
        self.config = CamPPSVConfig.from_config(config=config)
        if inputs_dim is not None:
            self.config.inputs_dim = inputs_dim
        if num_classes is not None:
            self.config.num_classes = num_classes
        super().__init__()
        # save_hyperparameters can't handle dataclass.
        config = self.config.to_dict()
        self.save_hyperparameters("config")

        self.cam = CamPP(self.config)
        if self.config.classifier_str:
            self.classifier = Classifier.from_str(
                self.config.classifier_str,
                self.config.embd_dim,
                self.config.num_classes,
                **self.config.classifier_params,
            )

        self.example_input_array = {
            "input_features": torch.randn(2, 300, self.config.inputs_dim)
        }

    def forward(self, input_features):
        """
        Backbone forward.

        Args:
            input_features (Tensor):
                input tensor of shape (B, T, F)
        """
        embd = self.cam(input_features)

        return embd

    def extract_embedding(self, input_features, max_chunk: int = 4000) -> XvectorOutput:
        """
        Pipline interface.

        Args:
            input_features (Tensor):
                input tensor of shape (B, T, F)
            max_chunk (int):
                longer input feature will be chunked.
        """

        chunk_inputs, chunk_sizes = self.split_chunks(
            input_features, max_chunk=max_chunk
        )
        embedding_stats = torch.empty(0)
        for idx, chunk in enumerate(chunk_inputs):
            this_embedding = self(chunk)
            if embedding_stats.size(0) == 0:
                embedding_stats = this_embedding * chunk_sizes[idx]
            else:
                embedding_stats += chunk_sizes[idx] * this_embedding
        embedding = embedding_stats / sum(chunk_sizes)

        return XvectorOutput(xvector=embedding)


# Test.
if __name__ == "__main__":
    from egrecho.utils.misc import if_continue, parse_bytes

    cfg = CamPPSVConfig()
    cam = CamPPModel(cfg)
    print(cam)
    num_params = parse_bytes(cam.get_num_params())
    num_param_msg = f"Num params: {num_params}"

    print(num_param_msg)

    print("Whether test jit.")
    if if_continue():
        export_f = "export.jit"
        method = "trace"  # script, trace
        if method == "onnx":
            cam.export_onnx(export_f)
            print(f"export onnx model to {export_f} success.")
        else:
            cam_m = cam.export_jit(export_f, method=method)
            print(cam_m)
            print(f"export to {export_f} success.")
            cam.eval()
            cam_m.eval()
            with torch.no_grad():
                inputs = torch.randn(1, 1000, 80)
                embd = cam(inputs)
                embd_m = cam_m(inputs)
                print(
                    f"========The diff of embd between orig and depoly is {((embd - embd_m) ** 2).sum()}========="
                )
