# -*- coding:utf-8 -*-
# (Author: Leo 2025-08)

from dataclasses import dataclass, field

from typing_extensions import Literal, Optional

from egrecho.core.config import DataclassConfig


@dataclass
class WeSpkConfig(DataclassConfig):
    """
    Configuration class for the wespk model.

    Args:
        init_model_id (str, optional):
            hub model id:

            .. code-block:: python

                Assets = {
                    "chinese": "cnceleb_resnet34.tar.gz",
                    "english": "voxceleb_resnet221_LM.tar.gz",
                    "campplus": "campplus_cn_common_200k.tar.gz",
                    "eres2net": "eres2net_cn_commom_200k.tar.gz",
                    "vblinkp": "voxblink2_samresnet34.zip",
                    "vblinkf": "voxblink2_samresnet34_ft.zip",
                }

        init_model_dir (str, optional):
            If provided, get wespeaker model from it.
    """

    init_model_id: str = "campplus"
    init_model_dir: str = None


@dataclass
class WeSpkSVConfig(WeSpkConfig):
    """
    Configuration class for the wespeaker backbone with Classification task.
    hub model id:

    .. code-block:: python

        Assets = {
            "chinese": "cnceleb_resnet34.tar.gz",
            "english": "voxceleb_resnet221_LM.tar.gz",
            "campplus": "campplus_cn_common_200k.tar.gz",
            "eres2net": "eres2net_cn_commom_200k.tar.gz",
            "vblinkp": "voxblink2_samresnet34.zip",
            "vblinkf": "voxblink2_samresnet34_ft.zip",
        }

    Args:
        num_classes:
            Need to be set to label number. Defaults to 2.
        classifier_str (str, optional):
            Margin classifier. Default is aam.
        classifier_params (dict, optinal):
            other kwargs passing to margin classifier.
            sub_k: sub center. Default is 1.
            do_topk: wheter do hard sample margin.
    """

    num_classes: int = 2
    classifier_str: Optional[Literal["linear", "am", "aam"]] = "aam"
    classifier_params: dict = field(
        default_factory=lambda: {"sub_k": 1, "do_topk": False}
    )
