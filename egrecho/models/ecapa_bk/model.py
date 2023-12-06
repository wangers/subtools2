from typing import Literal, Optional, Union

from egrecho.core.module import TopVirtualModel
from egrecho.models.ecapa.ecapa_config import EcapaConfig
from egrecho.models.ecapa.ecapa_xvector import EcapaXvector


class EcapaModel(TopVirtualModel):
    main_input_name = ["input_features"]

    def __init__(
        self,
        inputs_dim: Optional[int] = None,
        teacher_name: Optional[Literal["SVTeacher"]] = None,
        ecapa_config: Union[EcapaConfig, dict] = None,
    ) -> None:
        """
        Ecapa model with teacher for training.

        Args:
            teacher_name (str):
                choose a teacher to train, relax to ignore it if not neccesary.
            ecapa_config (dict, EcapaConifg):
                default configuration to initiate model.
        """
        super().__init__()
        self.ecapa_config = EcapaConfig.get_model_config(
            teacher_name=teacher_name, config=ecapa_config
        )
        if inputs_dim is not None:
            self.ecapa_config.inputs_dim = inputs_dim
        # save_hyperparameters can't handle dataclass.
        ecapa_config = self.ecapa_config.to_dict()
        self.save_hyperparameters()

        self.ecapa = EcapaXvector(self.ecapa_config)
        self.example_input_array = self.ecapa.dummy_inputs
        if teacher_name is not None:
            teacher_cls = self.ecapa_config.teacher_class
            teacher_kwds = self.ecapa_config.teacher_kwargs

            self.attach_teacher(teacher_cls(self.ecapa_config.embd_dim, **teacher_kwds))
            self.teacher.setup_model()

    def forward(self, input_features):
        """
        Backbone forward.

        Args:
            input_features (Tensor):
                input tensor of shape (B, T, F)
        """
        return self.ecapa(input_features)
