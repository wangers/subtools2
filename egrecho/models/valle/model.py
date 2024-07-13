# -*- coding:utf-8 -*-
# Copyright xmuspeech (Author: Leo 2024-04)

from dataclasses import dataclass
from typing import Optional, Union

import numpy as np
import torch

from egrecho.core.module import TopVirtualModel
from egrecho.models.valle.valle import ArDecoder, NarDecoder, padding_codes
from egrecho.models.valle.valle_config import ValleModelConfig
from egrecho.utils.mask import make_non_pad_mask
from egrecho.utils.types import ModelOutput


@dataclass
class ValleOutput(ModelOutput):
    ar_logits: Optional[torch.FloatTensor] = None
    ar_loss: Optional[torch.FloatTensor] = None
    nar_logits: Optional[torch.FloatTensor] = None
    nar_loss: Optional[torch.FloatTensor] = None
    ar_labels: Optional[torch.Tensor] = None
    nar_labels: Optional[torch.Tensor] = None


class Valle(TopVirtualModel):
    """Implementation of vall-e.

    "Neural Codec Language Models are Zero-Shot Text to Speech Synthesizers", https://arxiv.org/abs/2301.02111
    """

    CONFIG_CLS = ValleModelConfig

    def __init__(
        self,
        config: Union[ValleModelConfig, dict] = None,
    ):
        config = ValleModelConfig.from_config(config=config)
        super().__init__(config)
        # save_hyperparameters can't handle dataclass.
        config = self.config.to_dict(filt_type="default")
        self.save_hyperparameters("config")

        self.ar_model = ArDecoder(self.config) if self.config.has_ar else None
        self.nar_model = NarDecoder(self.config) if self.config.has_nar else None

        # self.example_input_array = {"input_features": torch.randn(2, 200, self.config.inputs_dim)}

    def sample_nar_qnt_idx(self):
        if (train_nar_rng := getattr(self, "train_nar_rng", None)) is None:
            train_nar_rng = np.random.default_rng(42)
            self.train_nar_rng = train_nar_rng

        return train_nar_rng.integers(1, self.config.num_codebooks)

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        text_input_ids: Optional[torch.Tensor] = None,
        text_attention_mask: Optional[torch.Tensor] = None,
        prefix_codes: Optional[torch.Tensor] = None,
        prefix_attention_mask: Optional[torch.Tensor] = None,
    ):
        """
        Training forward.

        Args:
            input_ids (`torch.LongTensor` of shape `(batch_size, sequence_length, number_of_codebooks)`, *optional*):
                It will lately choose the first of discret code embeddings `(batch_size, sequence_length, 1)`.
            attention_mask (`torch.LongTensor` of shape `(batch_size, kv_len)`, *optional*):
                Default behavior: generate a tensor that ignores pad tokens in `input_ids`. Causal mask will also
                be used by default.
            text_input_ids (`torch.LongTensor` of shape `(batch_size, text_sequence_length)`):
                Indices of input sequence tokens in the vocabulary.
            text_attention_mask (`torch.Tensor` of shape `(batch_size, text_sequence_length)`, *optional*):
                Mask to avoid performing attention on padding token indices. Mask values selected in `[0, 1]`:

                - 1 for tokens that are **not masked**,
                - 0 for tokens that are **masked**.

            prefix_codes (`torch.LongTensor` of shape `(batch_size, prefix_codes_sequence_length)` *optional*):
                Indices of externel audio prompts tokens.
            prefix_attention_mask (`torch.Tensor` of shape `(batch_size, prefix_codes_sequence_length)`, *optional*):
                Mask to avoid performing attention on padding token indices. Mask values selected in `[0, 1]`.
        """
        has_ar, has_nar = self.config.has_ar, self.config.has_nar
        assert has_ar or has_nar, "No AR/NAR submodel for trainig."
        assert input_ids.ndim == 3, input_ids.shape
        input_ids, ar_tgt_ids = padding_codes(
            input_ids,
            attention_mask,
            pad_value=self.config.codebook_size,
            shift_tgt_eos=self.config.codebook_size if has_ar else None,
        )
        outputs = ValleOutput()
        if has_ar:
            # ignore -100
            if attention_mask is not None:
                ar_tgt_ids = ar_tgt_ids[..., 0].masked_fill(
                    ~(attention_mask.bool()), -100
                )
            ar_out = self.ar_model.forward(
                input_ids,
                attention_mask,
                text_input_ids=text_input_ids,
                text_attention_mask=text_attention_mask,
            )
            ar_logits = ar_out[0]

            ar_loss = loss_fn(
                ar_logits,
                ar_tgt_ids,
                ignore_index=-100,
            )
            outputs.ar_logits = ar_logits
            outputs.ar_loss = ar_loss
            outputs.ar_labels = ar_tgt_ids
        if has_nar:
            codebook_idx = self.sample_nar_qnt_idx()
            nar_logits = self.nar_model.forward(
                codebook_idx=codebook_idx,
                input_ids=input_ids,
                text_input_ids=text_input_ids,
                attention_mask=attention_mask,
                text_attention_mask=text_attention_mask,
                prefix_codes=prefix_codes,
                prefix_attention_mask=prefix_attention_mask,
            )
            codebook_size = self.config.codebook_size
            nar_loss = loss_fn(
                nar_logits[..., :codebook_size],
                input_ids[..., codebook_idx],
                ignore_index=codebook_size,
            )
            outputs.nar_logits = nar_logits
            outputs.nar_loss = nar_loss
            outputs.ar_labels = input_ids[:, -nar_logits.shape[1], codebook_idx]
        return outputs

    @torch.inference_mode()
    def generate(
        self,
        input_ids: torch.Tensor,
        text_input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        text_attention_mask: Optional[torch.Tensor] = None,
        prompt_text_attention_mask: Optional[torch.Tensor] = None,
        phn_dur: float = 0.22,
        top_k: int = -100,
        temperature: float = 1.0,
        top_p: float = 1.0,
    ):
        """
        Args:
            input_ids (`torch.LongTensor` of shape `(batch_size, sequence_length, number_of_codebooks)`, *optional*):
                It will lately choose the first of discret code embeddings `(batch_size, sequence_length, 1)`.
            text_input_ids (`torch.LongTensor` of shape `(batch_size, text_sequence_length)`):
                Indices of input sequence tokens in the vocabulary.
            attention_mask (`torch.LongTensor` of shape `(batch_size, sequence_length)`, *optional*):
                Mask to avoid performing attention, Mask values selected in `[0, 1]`:

                - 1 for tokens that are **not masked**,
                - 0 for tokens that are **masked**.

            text_attention_mask (`torch.Tensor` of shape `(batch_size, text_sequence_length)`, *optional*):
                Mask to avoid performing attention on padding token indices. Mask values selected in `[0, 1]`.
            prompt_text_attention_mask (`torch.Tensor` of shape `(batch_size, prompt_text_sequence_length)`, *optional*):
                Mask which can be used to infer prompt text parts in `text_input_ids`.
            phn_dur (`float`, *optional*, default to 0.22):
                Duration per phn, relevant to approximate the maximum length of the generated audio code sequence.
            topk (`int`, *optional*):
                The number of highest probability vocabulary tokens to keep for top-k-filtering.
                if negative, ignore topk filter. Default to -100.
            temperature (`float`, *optional*):
                Temperature value for controlling randomness in sampling. Defaults to 1.0.
            top_p (`float`, *optional*):
                Top-p probability threshold for nucleus sampling. Defaults to 1.

        Returns:
          Return the predicted audio code matrix.
        """
        if not all(
            completed := (self.nar_model is not None, self.ar_model is not None)
        ):
            raise RuntimeError(
                f"Generates tokens need both ar and nar submodels, but got [ar, nar] {completed}."
            )
        if self.nar_model.prefix_mode == "exter":
            if prompt_text_attention_mask is None:
                raise ValueError(
                    "For prefix_mode=exter, prompt_text_attention_mask is required since the "
                    "nar model need it to exclude the prompt text part in input text. "
                    "[HINT] prompt_text_attention_mask should be of shape `(batch_size, prompt_text_sequence_length)`"
                    ". Mask values selected in `[0, 1]`. While 1 means not pad and 0 means pad."
                )
        bsz = text_input_ids.shape[0]
        device = input_ids.device

        first_code, gen_att_mask = self.ar_model.generate(
            input_ids,
            text_input_ids,
            attention_mask=attention_mask,
            text_attention_mask=text_attention_mask,
            phn_dur=phn_dur,
            top_k=top_k,
            top_p=top_p,
            temperature=temperature,
        )

        # Exclude prompts part in text.
        if self.nar_model.prefix_mode == "exter":
            tot_txt_lens = (
                torch.full(
                    (bsz,), text_input_ids.shape[1], dtype=torch.long, device=device
                )
                if text_attention_mask is None
                else attention_mask.sum(-1)
            )
            prompt_txt_lens = prompt_text_attention_mask.sum(-1)
            gen_txt_lens = (tot_txt_lens - prompt_txt_lens).long()
            text_attention_mask = make_non_pad_mask(gen_txt_lens)
            new_text_input_ids = torch.full(
                (bsz, text_attention_mask.shape[1]),
                fill_value=self.config.pad_text_token_id,
                dtype=text_input_ids.dtype,
                device=device,
            )

            # roll gen texts to begining.
            for i in range(bsz):
                new_text_input_ids[i, : gen_txt_lens[i]] = text_input_ids[
                    i, prompt_txt_lens[i] : prompt_txt_lens[i] + gen_txt_lens[i]
                ]
            text_input_ids = new_text_input_ids
        preds = self.nar_model.generate(
            input_ids=first_code,
            text_input_ids=text_input_ids,
            prefix_codes=input_ids,
            attention_mask=gen_att_mask,
            text_attention_mask=text_attention_mask,
            prefix_attention_mask=attention_mask,
        )
        return preds, gen_att_mask


def loss_fn(logits, targets, ignore_index: int = -100):
    logits = logits.contiguous()
    targets = targets[:, -logits.shape[1] :].contiguous()
    loss = torch.nn.functional.cross_entropy(
        logits.view(-1, logits.size(-1)), targets.view(-1), ignore_index=ignore_index
    )
    return loss


def example_usage():

    from egrecho.utils.cuda_utils import release_memory, to_device

    device = "cpu"  # "cpu"
    config = ValleModelConfig()

    x = torch.randint(0, 100, size=[4, 8])
    x_lens = torch.randint(6, 8, size=[4])
    x_lens[-1] = 8
    x_mask = make_non_pad_mask(x_lens)
    y = torch.randint(0, 1000, size=[4, 16, 8])
    y_lens = torch.randint(8, 16, size=[4])
    y_lens[-1] = 16
    y_mask = make_non_pad_mask(y_lens)
    main_model_inputs = dict(
        input_ids=y,
        attention_mask=y_mask,
        text_input_ids=x,
        text_attention_mask=x_mask,
    )

    for prefix_mod in ['starter', 'exter']:
        config.prefix_mode = prefix_mod
        model_inputs = main_model_inputs.copy()
        if prefix_mod == 'exter':
            prefix_codes = torch.randint(0, 1000, size=[4, 12, 8])
            model_inputs['prefix_codes'] = prefix_codes
        model = Valle(config)
        print(model)
        print(
            f'Params num: {model.get_num_params()}, Test prefix_mod={prefix_mod} ....'
        )
        model_inputs = to_device(model_inputs, device)
        model.to(device=device)

        # Traning
        valle_outs: ValleOutput = model(**model_inputs)
        stats = dict(
            ar_logits_shape=valle_outs.ar_logits.shape,
            nar_logits_shape=valle_outs.nar_logits.shape,
            ar_loss=valle_outs.ar_loss.detach().item(),
            nar_loss=valle_outs.nar_loss.detach().item(),
        )
        print(f'Training forward: {stats}')

        # Generate
        model.eval()
        model_inputs = main_model_inputs.copy()
        infer_y = y[:, :8]
        infer_y_lens = torch.randint(4, 8, size=[4])
        infer_y_lens[-1] = 8
        infer_y_mask = make_non_pad_mask(infer_y_lens)
        model_inputs['input_ids'] = infer_y
        model_inputs['attention_mask'] = infer_y_mask
        if prefix_mod == 'exter':
            # we assume half of tot text is prompts part and we need exlude it in later nar for this mode.
            prompt_text_lens = x_lens // 2
            prompt_text_attention_mask = make_non_pad_mask(prompt_text_lens)
            model_inputs['prompt_text_attention_mask'] = prompt_text_attention_mask
        model_inputs = to_device(model_inputs, device)
        preds, att_mask = model.generate(**model_inputs)
        lens = att_mask.sum(-1)
        print('Generates codebook:')
        print(f'gen batch auidos shape: {preds.shape}, lens: {lens}')
        release_memory(model)


if __name__ == "__main__":
    from egrecho.utils.seeder.seed import set_all_seed

    set_all_seed()
    example_usage()
