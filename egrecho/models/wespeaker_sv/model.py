# -*- coding:utf-8 -*-
#  (Author: Leo 2025-08)
"""
Wespeaker model.
"""
import os
import shutil
from pathlib import Path
from typing import Optional, Union

import torch
import yaml

from egrecho.models.architecture.speaker import XvectorMixin, XvectorOutput
from egrecho.models.wespeaker_sv.wespk_config import WeSpkSVConfig
from egrecho.nn.classifier import Classifier
from egrecho.utils.dist import is_global_rank_zero


def load_wespk_model(
    model_id: str = None,
    model_dir: str = None,
):
    """
    Assets = {
        "chinese": "cnceleb_resnet34.tar.gz",
        "english": "voxceleb_resnet221_LM.tar.gz",
        "campplus": "campplus_cn_common_200k.tar.gz",
        "eres2net": "eres2net_cn_commom_200k.tar.gz",
        "vblinkp": "voxblink2_samresnet34.zip",
        "vblinkf": "voxblink2_samresnet34_ft.zip",
    }
    """
    try:
        import wespeaker
    except ImportError as e:
        error_msg = f"Failed to import wespeaker: {str(e)}. Please ensure it's installed correctly. Refer to: https://github.com/wenet-e2e/wespeaker/blob/master/README.md."
        raise ImportError(error_msg) from e

    from wespeaker.cli.hub import Hub

    if model_dir is None:
        model_dir = Hub.get_model(model_id)

    model = wespeaker.load_model_pt(model_dir)
    model.train()
    # just get config info
    with open(os.path.join(model_dir, "config.yaml"), "r") as f:
        config = yaml.load(f, Loader=yaml.FullLoader)
    return model, config


class WeSpkModel(XvectorMixin):
    """
    Cam++ model with teacher for training.

    Args:
        config (dict, WeSpkSVConfig):
            default configuration to initiate model.
        num_classes (int):
            expose to link with datamodule.
    """

    CONFIG_CLS = WeSpkSVConfig
    main_input_name = ["input_features"]

    def __init__(
        self,
        config: Union[WeSpkSVConfig, dict] = None,
        num_classes: Optional[int] = None,
    ) -> None:
        config = WeSpkSVConfig.from_config(config=config)

        if num_classes is not None:
            config.num_classes = num_classes
        super().__init__(config)
        # save_hyperparameters can't handle dataclass.
        config = self.config.to_dict()
        self.save_hyperparameters("config")

        self.config: WeSpkSVConfig  # ide hint

        self.wespk_backbone, wespk_config = load_wespk_model(
            self.config.init_model_id, self.config.init_model_dir
        )

        embd_dim = wespk_config["model_args"]["embed_dim"]
        inputs_dim = wespk_config["model_args"].get("feat_dim", 80)
        if self.config.classifier_str:
            self.classifier = Classifier.from_str(
                self.config.classifier_str,
                embd_dim,
                self.config.num_classes,
                **self.config.classifier_params,
            )

        self.example_input_array = {"input_features": torch.randn(2, 300, inputs_dim)}

    def forward(self, input_features):
        """
        Backbone forward.

        Args:
            input_features (Tensor):
                input tensor of shape (B, T, F)
        """
        embd = self.wespk_backbone(input_features)

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
            outputs = self(chunk)
            this_embedding = outputs[-1] if isinstance(outputs, tuple) else outputs
            if embedding_stats.size(0) == 0:
                embedding_stats = this_embedding * chunk_sizes[idx]
            else:
                embedding_stats += chunk_sizes[idx] * this_embedding
        embedding = embedding_stats / sum(chunk_sizes)

        return XvectorOutput(xvector=embedding)

    def save_to(
        self,
        savedir,
        **kwargs,
    ):
        """Saves to a directory.

        Args:
            savedir: path
            \**kwargs: args passing to :meth:`~egrecho.core.loads.SaveLoadHelper.save_to` of object
                of :class:`~egrecho.core.loads.SaveLoadHelper`.

        """

        # save a base wespeaker model dir for init
        if self.config.init_model_dir is not None:
            self.config.init_model_id = None

            if is_global_rank_zero():
                # shutil.copytree(self.config.init_model_dir, Path(savedir) / 'wespeaker_init', dirs_exist_ok=True)

                src_dir = Path(self.config.init_model_dir)
                dst_dir = Path(savedir) / "wespeaker_init"
                dst_dir.mkdir(parents=True, exist_ok=True)

                for file_name in ["avg_model.pt", "config.yaml"]:
                    src = src_dir / file_name
                    dst = dst_dir / file_name
                    if src.exists():
                        shutil.copy2(src, dst)
                    else:
                        raise FileNotFoundError(f"Required file not found: {src}")

        super().save_to(savedir, **kwargs)

    @classmethod
    def fetch_from(
        cls,
        dirpath,
        config=None,
        init_weight="pretrained",
        map_location: Optional[torch.device] = "cpu",
        strict: bool = True,
        init_model_dir="auto",
        **kwargs,
    ):
        """Fetch pretrained from a directory.

        Args:
            dirpath: srcdir
            config: config path/dict which could override the underlying model init cfg (model_config.yaml).
            init_weight: Init weight from ('pretrained'|'random'), or string ckpt
                name (model_weight.ckpt) or full path to ckpt /path/to/model_weight.ckpt.
                Default: ``'pretrained'``.

            map_location: MAP_LOCATION_TYPE as in torch.load().
                Defaults to 'cpu'.
            strict: Whether to strictly enforce that the keys in checkpoint match
                the keys returned by this module's state dict.
                Defaults: ``True``
            init_model_dir: wespk model dir for initiation.

            \**kwargs(Dict[str,Any]): additional parameters of model cfg.

        """
        if (
            init_model_dir == "auto"
            and (wespk_init := (Path(dirpath) / "wespeaker_init")).exists()
        ):
            print(f"Auto init wespeaker model from dir  => {wespk_init}")
            init_model_dir = wespk_init
        return super().fetch_from(
            dirpath,
            config=config,
            init_weight=init_weight,
            map_location=map_location,
            strict=strict,
            init_model_dir=init_model_dir,
            **kwargs,
        )


# Test.
if __name__ == "__main__":
    import warnings

    from egrecho.utils.misc import if_continue, parse_bytes

    cfg = WeSpkSVConfig(init_model_id="vblinkp")
    model = WeSpkModel(cfg)

    try:
        from lightning.pytorch.utilities.model_summary import ModelSummary

        model_summary = ModelSummary(model, max_depth=2)
        print(model_summary)

        del model_summary
    except:  # noqa
        warnings.warn("Failed calling lightning model summary.")
        print(model)
        num_params = parse_bytes(model.get_num_params())
        num_param_msg = f"Num params: {num_params}"
        print(num_param_msg)

    print("Whether test jit.")
    if if_continue():

        method = "trace"  # script, trace, onnx

        if method == "onnx":
            export_f = "export.onnx"
            model.export_onnx(export_f)
            print(f"export onnx model to {export_f} success.")
        else:
            export_f = "export.jit"
            cam_m = model.export_jit(export_f, method=method)
            print(cam_m)
            print(f"export to {export_f} success.")
            model.eval()

            with torch.no_grad():
                inputs = torch.randn(1, 1000, 80)
                embd = model(inputs)
                embd_m = model(inputs)
                print(embd.shape, embd)
                print(
                    f"========The diff of embd between orig and depoly is {((embd - embd_m) ** 2).sum()}========="
                )
