from dataclasses import dataclass, field, fields

from typing_extensions import Dict, Literal, Optional, Type, Union

from egrecho.core.config import DataclassConfig

default_pooling_params = {
    "num_q": 1,
    "num_head": 1,
    "time_attention": True,
    "hidden_size": 128,
    "stddev": True,
}

default_margin_warm_kwargs = {"start_epoch": 5, "end_epoch": 15, "offset_margin": -0.2}


@dataclass
class EcapaConfig(DataclassConfig):
    """
    Configuration class for the Ecapa.

    Args:
        inputs_dim (int, optional):
            The dimension of the input feature. Default is 80.
        num_targets (int, optional):
            The number of target classes. Default is 2.
        channels (int, optional):
            The number of channels in the model. Default is 512.
        embd_dim (int, optional):
            The embedding dimension. Default is 192.
        mfa_dim (int, optional):
            The dimension of the MFA layer. Default is 1536.
        pooling_method (str, optional):
            The pooling method used in the model. Default is 'mqmhasp'.
        pooling_params (Dict, optional): Pooling parameters.
            Default is a predefined dictionary of `asp`.
        embd_layer_num (Literal[1, 2], optional):
            The number of embedding layers. Default is 1.
        post_norm (bool, optional):
            Whether to apply post-normalization. Default is False.
    """

    inputs_dim: int = 80
    channels: int = 512
    embd_dim: int = 192
    mfa_dim: int = 1536
    pooling_params: dict = field(default_factory=lambda: default_pooling_params)
    embd_layer_num: Literal[1, 2] = 1
    post_norm: bool = False

    def __post_init__(self):
        self.pooling_params = default_pooling_params.update(self.pooling_params)

    @classmethod
    def get_model_config(
        self,
        teacher_name: Optional[str] = None,
        config: Union[dict, "DataclassConfig"] = None,
    ):
        """
        Get specify model config.

        Args:
            teacher:
                which task.
            config:
                dict or dataclass config.
        """
        if teacher_name == "SVTeacher":
            cfg_cls = EcapaSVConfig
        else:
            cfg_cls = EcapaConfig
        return cfg_cls.from_config(config, strict=True)

    @property
    def teacher_kwargs(self) -> Dict:
        d = self.to_dict(filt_type="orig")
        to_filter = [f.name for f in fields(EcapaConfig)]
        return {k: d[k] for k in d if k not in to_filter}

    @property
    def teacher_class(self) -> Type:
        raise NotImplementedError


@dataclass
class EcapaSVConfig(EcapaConfig):
    """
    Configuration class for the Ecapa with SV task.

    Args:
        num_classes:
            Need to be set to label number. Defaults to 2.
        head_name (str, optional):
            Margin head. Default is aam.
        head_kwargs (dict, optinal):
            other kwargs passing to margin head.
            sub_k: sub center. Default is 1.
            do_topk: wheter do hard sample margin.
    """

    num_classes: int = 2
    head_name: Literal["linear", "am", "aam"] = "aam"
    head_kwargs: dict = field(default_factory=lambda: {"sub_k": 1, "do_topk": False})
    margin_warm: bool = True
    margin_warm_kwargs: dict = field(default_factory=lambda: default_margin_warm_kwargs)

    def __post_init__(self):
        super().__post_init__()
        self.margin_warm_kwargs = default_margin_warm_kwargs.update(
            self.margin_warm_kwargs
        )

    @property
    def teacher_class(self) -> Type:
        from egrecho.models.architecture.speaker import SVTeacher

        return SVTeacher
