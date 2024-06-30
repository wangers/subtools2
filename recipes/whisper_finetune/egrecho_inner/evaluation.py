# (Author: Leo 2024-06)

import functools
import gc
from dataclasses import dataclass
from pathlib import Path
from typing import List, Literal, Optional

import torch
from olr_datamodule import BatchProcessor, OlrAsrDataModule
from tqdm.auto import tqdm
from transformers import WhisperForConditionalGeneration, WhisperProcessor
from utils.data_utils import XNormlizer
from utils.utils import add_arguments
from utils.whisper_encoder_forward_monkey_patch import replace_whisper_encoder_forward

from egrecho.core.parser import BaseCommand, CommonParser
from egrecho.score.wer import DetailWER
from egrecho.utils.common import Timer, alt_none
from egrecho.utils.cuda_utils import release_memory
from egrecho.utils.io import SerializationFn, repr_dict
from egrecho.utils.logging import get_logger
from egrecho.utils.seeder import set_all_seed
from egrecho_cli import register_command

logger = get_logger()


def parse_gpu(gpu_id: str):
    from egrecho.utils.cuda_utils import GPUManager

    gpu_id = alt_none(gpu_id, '')
    gpu_id = gpu_id.lower().strip()
    if gpu_id == "auto":
        return GPUManager.detect()
    elif gpu_id == "" or gpu_id is None:
        gpu_id = "cpu"
    else:
        gpu_id = int(gpu_id)
    return gpu_id


def repr_results(*texts: List[str], prefix: Optional[str] = None):
    if not texts:
        return
    num_compare = len(texts)
    bsz = len(texts[0])
    if prefix is None:
        prefix = [f"[ {i} ]" for i in range(num_compare)]
    prefix = [pre + ": " for pre in prefix]
    assert len(prefix) == num_compare, f"{len(prefix)} != {len(texts)}"
    assert all(
        len(batch) == bsz for batch in texts
    ), f"batch size should be same as {bsz}, but got {[len(b) for b in texts]}"
    repr_str = "------------------------------------------------\n"
    for cmps in zip(*texts):
        repr_str += "\n".join(prefix[i] + cmp for i, cmp in enumerate(cmps))
        repr_str += "\n------------------------------------------------\n"
    return repr_str


@dataclass
class GenArgs:
    """Generate params.

    Args:
        return_timestamps:
            带时间戳解码
        language:
            设置默认语言, 可全称也可简写, 如果为None则评估的是多语言
        task:
            任务选择
        num_beams:
            解码beam, 1为贪心
        prompt_text:
            是否提供prompt
        no_speech_threshold:
            长转录模式下与logprob_threshold共同判定静音片段
        compression_ratio_threshold:
            长转录模式下控制温度重启
        temperature_fallback:
            是否temperature fallback
        logprob_threshold:
            temperature fallback阈值
        long_gen_kwargs_plus:
            长转录的额外参数
    """

    return_timestamps: bool = False
    language: str = "zh"
    task: Literal["transcribe", "translate"] = "transcribe"
    num_beams: int = 1
    no_speech_threshold: float = 0.6
    compression_ratio_threshold: float = 1.35
    temperature_fallback: bool = True
    logprob_threshold: float = 1.0
    prompt_text: Optional[str] = None
    long_gen_kwargs_plus: Optional[dict] = None

    @property
    def common_kwargs(self):
        return dict(
            return_timestamps=self.return_timestamps,
            language=self.language,
            task=self.task,
            num_beams=self.num_beams,
        )

    @property
    def long_kwargs(self):

        return dict(
            compression_ratio_threshold=self.compression_ratio_threshold,
            temperature=(0.0, 0.2, 0.4, 0.6, 0.8, 1.0)
            if self.temperature_fallback
            else 0,
            logprob_threshold=self.logprob_threshold,
            no_speech_threshold=self.no_speech_threshold,
            **(self.long_gen_kwargs_plus or {}),
        )


DESCRIPTION = "Eval whisper."


@register_command(name="eval", aliases=[])
class EvalWhisper(BaseCommand):
    @classmethod
    def get_dummy_parser(cls) -> CommonParser:
        return CommonParser(description=DESCRIPTION)

    @classmethod
    def setup_parser(cls, parser: CommonParser):
        parser.add_cfg_flag()
        OlrAsrDataModule.add_arguments(parser)
        add_arg = functools.partial(add_arguments, argparser=parser)
        add_arg(
            "model_path",
            type=str,
            default="exp/whisper-medium/lora/whisper-medium-finetune",
            help="合并模型的路径, 或者是huggingface上模型的名称",
        )
        add_arg(
            "outdir",
            type=str,
            default="exp/whisper-medium/lora/whisper-medium-finetune",
            help="结果文件存储目录",
        )
        add_arg(
            "gpu_id",
            type=str,
            default="auto",
            help="空字符串``''``为cpu, 'auto'自动选择空闲, 或固定 (``1``)",
        )
        add_arg(
            "use_fast_tokenizer",
            type=bool,
            default=True,
            help="是否用tokenizer包的fast tokenizer",
        )
        add_arg("local_files_only", type=bool, default=True, help="是否只在本地加载模型, 不尝试下载")
        add_arg(
            "remove_whisper_encoder_input_length_restriction",
            type=bool,
            default=False,
            help="支持变长输入",
        )
        add_arg("log_internal", type=int, default=10, help="大于零时的解码日志间隔")
        parser.add_class_args(GenArgs, "gen_args")
        return parser

    @staticmethod
    def run_from_args(args, parser: Optional[CommonParser] = None):
        args_init = parser.instantiate_classes(args)
        EvalWhisper(args_init)

    def __init__(
        self,
        args,
    ):

        gen_args: GenArgs = args.gen_args

        # 获取Whisper的数据处理器, 这个包含了特征提取器、tokenizer
        processor = WhisperProcessor.from_pretrained(
            args.model_path,
            language=gen_args.language,
            task=gen_args.task,
            no_timestamps=True,
            local_files_only=args.local_files_only,
            use_fast=args.use_fast_tokenizer,
        )
        # 处理 <30s 语音
        if not (
            feat_padding_to_max := not args.remove_whisper_encoder_input_length_restriction
        ):
            replace_whisper_encoder_forward()
        datamodule = OlrAsrDataModule(args)
        eval_dataloader = datamodule.test_dataloaders(datamodule.test_cuts())
        data_collator = BatchProcessor(
            processor, padding_to_max=feat_padding_to_max, process_for_train=False
        )
        # 获取模型
        device = parse_gpu(args.gpu_id)
        model = WhisperForConditionalGeneration.from_pretrained(
            args.model_path, device_map=device, local_files_only=args.local_files_only
        )
        model.eval()

        common_gen_kwargs, long_gen_kwargs = (
            gen_args.common_kwargs,
            gen_args.long_kwargs,
        )
        if prompt_text := gen_args.prompt_text:
            common_gen_kwargs["prompt_ids"] = processor.get_prompt_ids(
                prompt_text, return_tensors="pt"
            ).to(device)

        # 获取评估方法
        metric = DetailWER(details='all')

        x_normalizer = XNormlizer(processor.tokenizer)
        timer = Timer()
        times_audio_total = 0
        times_transcription_total = 0
        n_samples = 0
        set_all_seed()
        outdir = Path(args.outdir)
        outdir.mkdir(exist_ok=True, parents=True)
        recog_file = outdir / 'recogs-test.txt'
        metric_file = outdir / 'errs-test.json'
        with torch.cuda.amp.autocast(), torch.no_grad(), DetailWER.open_writer(
            recog_file
        ) as w_f:
            # 开始评估
            for step, batch in enumerate(
                tqdm(
                    eval_dataloader,
                    desc="Evaluating",
                    unit=" steps",
                    position=0,
                )
            ):

                batch = data_collator(batch)
                batch_langs = batch["langs"]
                language = common_gen_kwargs.pop("language")
                langs = [lang if lang else language for lang in batch_langs]
                common_gen_kwargs["language"] = langs
                input_features = batch["input_features"].to(device)
                # automatically use long-form args if required
                inner_batch_size, num_mels, seq_len = input_features.shape
                n_samples += inner_batch_size
                if seq_len <= 3000:
                    batch_gen_kwargs = common_gen_kwargs
                else:
                    batch_gen_kwargs = {**common_gen_kwargs, **long_gen_kwargs}
                timer.reset()
                generated_tokens = (
                    model.generate(
                        input_features=batch["input_features"].to(device),
                        attention_mask=batch["attention_mask"].to(device),
                        **batch_gen_kwargs,
                    )
                    .cpu()
                    .numpy()
                )

                this_trans_time = timer.elapse()
                this_audio_time = float(sum(batch["length_in_s"]))

                times_audio_total += this_audio_time
                times_transcription_total += this_trans_time

                # 将预测和实际的token转换为文本
                decoded_preds = processor.tokenizer.batch_decode(
                    generated_tokens, skip_special_tokens=True
                )
                batch_refs = batch["texts"]
                # 去除结果中可能存在的prompt
                if gen_args.prompt_text is not None:
                    decoded_preds = [
                        decoded_pred.replace(gen_args.prompt_text, "")
                        for decoded_pred in decoded_preds
                    ]

                normed_preds = x_normalizer(decoded_preds, langs)
                normed_refs = x_normalizer(batch_refs, langs)
                if step == 1 or (
                    args.log_internal > 0 and step % args.log_internal == 0
                ):
                    prefix = ["[ hyps_raw ]", "[ hyps ]", "[ refs ]"]
                    logger.info(
                        f"Decode info:\n"
                        f"{repr_results(decoded_preds,normed_preds,normed_refs,prefix=prefix)}"
                        f"[ Step: {step}, acc samples: {n_samples}, this batch_size: {len(normed_refs)}, "
                        f"this rtf: {round(this_trans_time/this_audio_time, 5)} ]: \n"
                    )
                m_out = metric.update(preds=normed_preds, target=normed_refs)
                for box in m_out.gen_alins():
                    w_f.write(box + '\n', flush=step % 50 == 0)
                # 删除计算的记录
                del generated_tokens, batch
                gc.collect()
            release_memory()
            # 计算评估结果
            m = metric.compute()
            stats = m.to_dict()
            stats["RTF"] = round(times_transcription_total / times_audio_total, 5)
            SerializationFn.save_file(stats, metric_file)
            print(repr_dict(stats))


if __name__ == "__main__":
    parser = EvalWhisper.get_dummy_parser()
    parser = EvalWhisper.setup_parser(parser)
    args = parser.parse_args()
    logger.info(
        f"Got parsed args: \n{parser.dump(args.clone(),skip_default=True)}", ranks=[0]
    )
    EvalWhisper.run_from_args(args, parser)
