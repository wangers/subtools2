# -*- coding:utf-8 -*-
# Copyright xmuspeech (Author: Leo 2024-08)

from dataclasses import dataclass
from typing import List, Optional, Union

import torch
import torch.nn.functional as F

from egrecho.core.module import TopVirtualModel
from egrecho.models.e2_tts.e2tts import E2TTSFlow
from egrecho.models.e2_tts.e2tts_config import E2TTSConfig
from egrecho.utils.common import alt_none
from egrecho.utils.mask import make_non_pad_mask
from egrecho.utils.types import ModelOutput


@dataclass
class E2TTSTrainOutput(ModelOutput):
    loss: torch.FloatTensor
    x_ctx: Optional[torch.FloatTensor] = None
    pred_vector_field: Optional[torch.FloatTensor] = None


@dataclass
class E2TTSInferOutput(ModelOutput):
    mel: torch.FloatTensor
    cond_mask: Optional[torch.Tensor] = None
    attention_mask: Optional[torch.Tensor] = None

    def to_mel_list(self, with_cond: bool = False) -> List[torch.Tensor]:
        attn_mask = self.gen_attn_mask(with_cond)
        return [mel[one_mask] for mel, one_mask in zip(self.mel, attn_mask)]

    def gen_attn_mask(self, with_cond: bool = False) -> torch.BoolTensor:
        """Attention mask for audio generation"""
        bsz, seq_len, device = *self.mel.shape[:2], self.mel.device
        attn_mask = alt_none(
            self.attention_mask, torch.ones((bsz, seq_len), device=device)
        )
        attn_mask = attn_mask.bool()
        if not with_cond:
            cond_mask = alt_none(
                self.cond_mask, torch.zeros((bsz, seq_len), device=device)
            )
            cond_mask = cond_mask.bool()
            attn_mask = (~cond_mask) & attn_mask
        return attn_mask

    @property
    def cond_lens(self) -> List[int]:
        if self.cond_mask is not None:
            return self.cond_mask.sum(-1).long().tolist()
        return [0] * self.mel.shape[0]

    @property
    def val_lens(self) -> List[int]:
        if self.attention_mask is not None:
            return self.attention_mask.sum(-1).long().tolist()
        return [self.mel.shape[1]] * self.mel.shape[0]

    @property
    def gen_lens(self) -> List[int]:
        return (torch.tensor(self.val_lens) - torch.tensor(self.cond_lens)).tolist()


class E2TTS(TopVirtualModel):
    """Implementation of E2TTS.

    "Embarrassingly Easy Fully Non-Autoregressive Zero-Shot TTS", https://arxiv.org/abs/2406.18009
    """

    CONFIG_CLS = E2TTSConfig

    def __init__(
        self,
        config: Union[E2TTSConfig, dict] = None,
    ):
        config = E2TTSConfig.from_config(config=config)
        super().__init__(config)
        # save_hyperparameters can't handle dataclass.
        config = self.config.to_dict(filt_type="default")
        self.save_hyperparameters("config")

        self.e2tts = E2TTSFlow(self.config)

        self.example_input_array = {
            "input_features": torch.randn(2, 1000, self.config.inputs_dim),
            "text_input_ids": torch.randint(0, self.config.vocab_size, size=[2, 300]),
        }

    def forward(
        self,
        input_features: torch.FloatTensor,
        text_input_ids: Optional[torch.Tensor],
        attention_mask: Optional[torch.Tensor] = None,
        text_attention_mask: Optional[torch.Tensor] = None,
    ):
        """Forward training

        Args:
            input_features (`torch.FloatTensor` of shape `(batch_size, sequence_length, feature_size)`):
                Input feature.

            attention_mask (`torch.Tensor` of shape `(batch_size, sequence_length)`, *optional*):
                Mask to avoid performing attention on padding features. Mask values selected in `[0, 1]`:

                - 1 for **not masked**,
                - 0 for **masked**.

            text_input_ids (`torch.LongTensor` of shape `(batch_size, text_sequence_length)`):
                Indices of input sequence text tokens in the vocabulary.  should
                you provide it.
        """
        loss, cond, pred_v = self.e2tts(
            input_features, text_input_ids, attention_mask=attention_mask
        )

        return E2TTSTrainOutput(loss, cond, pred_v)

    @torch.inference_mode()
    def generate(
        self,
        input_features: torch.FloatTensor,
        gen_duration: Union[int, torch.Tensor],
        *,
        text_input_ids: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        steps=32,
        cfg_strength=1.0,  # they used a classifier free guidance strength of 1.
        max_duration=4096,  # in case the duration predictor goes haywire
        odeint_kwargs=None,
        text_attention_mask: Optional[torch.Tensor] = None,
    ):
        """ODE based sampling"""

        out, cond_mask, mask = self.e2tts.generate(
            input_features,
            gen_duration,
            text_input_ids=text_input_ids,
            attention_mask=attention_mask,
            steps=steps,
            cfg_strength=cfg_strength,
            max_duration=max_duration,
            odeint_kwargs=odeint_kwargs,
        )

        return E2TTSInferOutput(out, cond_mask, mask)

    @torch.inference_mode()
    def generate_loss_step(
        self,
        input_features: torch.FloatTensor,
        attention_mask: Optional[torch.Tensor] = None,
        mask_frac: float = 0.7,
        **kwargs,
    ):
        assert 0 < mask_frac <= 1.0
        if attention_mask is None:
            attention_mask = torch.ones(
                input_features.shape[:2], device=input_features.device
            )
        lens = attention_mask.sum(-1).long()
        gen_duration = (mask_frac * lens).long()
        ref_feats = input_features
        attention_mask = make_non_pad_mask(lens - gen_duration)
        melout: E2TTSInferOutput = self.generate(
            input_features, gen_duration, attention_mask=attention_mask, **kwargs
        )
        mel_loss = F.l1_loss(melout.mel, ref_feats, reduction="none")
        gen_mask = melout.attention_mask.bool() & ~(melout.cond_mask.bool())
        mel_loss = mel_loss[gen_mask].mean()
        return mel_loss, melout, ref_feats


def example_usage():

    from egrecho.utils.cuda_utils import release_memory, to_device

    device = "cpu"  # "cpu"
    config = E2TTSConfig()

    x = torch.randn(4, 1000, config.inputs_dim)
    x_lens = torch.randint(600, 1000, size=[4])
    x_lens[-1] = 1000
    x_mask = make_non_pad_mask(x_lens)
    x = x.masked_fill(~x_mask[..., None], 0.0)
    y = torch.randint(0, config.vocab_size, size=[4, 300])
    y_lens = torch.randint(150, 300, size=[4])
    y_lens[-1] = 300
    y_mask = make_non_pad_mask(y_lens)
    y = y.masked_fill(~y_mask, 0)
    main_model_inputs = dict(
        input_features=x,
        attention_mask=x_mask,
        text_input_ids=y,
    )

    model_inputs = main_model_inputs.copy()

    model = E2TTS(config)
    print(model)
    print(f"Params num: {model.get_num_params()} ....")
    model_inputs = to_device(model_inputs, device)
    model.to(device=device)

    # Traning
    model_outs: E2TTSTrainOutput = model(**model_inputs)
    print(
        f"Training forward: loss={model_outs.loss}, x_ctx shape={model_outs.x_ctx.shape}, vf_shape={model_outs.pred_vector_field.shape}"
    )

    # Generate
    model.eval()
    x = torch.randn(4, 300, config.inputs_dim)
    x_lens = torch.randint(150, 300, size=[4])
    x_lens[-1] = 300
    x_mask = make_non_pad_mask(x_lens)
    x = x.masked_fill(~x_mask[..., None], 0.0)

    model_inputs = dict(
        input_features=x,
        gen_duration=700,
        attention_mask=x_mask,
        text_input_ids=y,
    )

    model_inputs = to_device(model_inputs, device)
    gens: E2TTSInferOutput = model.generate(**model_inputs)
    print("Generates mel:")
    print(
        f"gen batch mel shape: {gens.mel.shape}, cond_lens: {gens.cond_lens}, gen_lens: {gens.gen_lens}"
    )
    release_memory(model)


if __name__ == "__main__":
    from egrecho.utils.seeder.seed import set_all_seed

    set_all_seed()
    example_usage()
