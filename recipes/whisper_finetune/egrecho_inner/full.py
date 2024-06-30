# (Author: Leo 2024-06-04)

import functools
import os
from pathlib import Path
from typing import Optional

import torch
from olr_datamodule import BatchProcessor, OlrAsrDataModule
from trainer_patch import CustomTrainer
from transformers import (
    Seq2SeqTrainingArguments,
    WhisperForConditionalGeneration,
    WhisperProcessor,
)
from utils.callback import LhotseCallback, patch_skip_batches
from utils.model_utils import load_from_checkpoint
from utils.utils import add_arguments, make_inputs_require_grad
from utils.whisper_encoder_forward_monkey_patch import replace_whisper_encoder_forward

from egrecho.core.parser import BaseCommand, CommonParser
from egrecho.utils.constants import DEFAULT_TRAIN_FILENAME
from egrecho.utils.cuda_utils import release_memory
from egrecho.utils.logging import _infer_rank, get_logger

# torch._dynamo.config.suppress_errors = True

logger = get_logger()

DESCRIPTION = "Full finetune whisper"


class FullTune(BaseCommand):
    @classmethod
    def get_dummy_parser(cls) -> CommonParser:
        return CommonParser(description=DESCRIPTION)

    @classmethod
    def setup_parser(cls, parser: CommonParser):
        parser.add_cfg_flag()
        OlrAsrDataModule.add_arguments(parser)
        add_arg = functools.partial(add_arguments, argparser=parser)
        add_arg(
            "base_model", type=str, default="openai/whisper-tiny", help="Whisper的基础模型"
        )
        add_arg("output_dir", type=str, default="output/", help="训练保存模型的路径")
        add_arg("optim", type=str, default="adamw_torch", help="优化器")
        add_arg("lion_beta1", type=float, default=0.016, help="wd lion")
        add_arg("lr_scheduler_type", type=str, default="linear", help="学习率衰减")
        add_arg("num_train_epochs", type=int, default=3, help="训练的轮数")
        add_arg("max_steps", type=int, default=-1, help="设置特定训练步数，否则由轮数推出")
        add_arg("freeze_encoder", type=bool, default=True, help="是否freeze encoder")
        add_arg("warmup_steps", type=int, default=500, help="训练预热步数")
        add_arg("logging_steps", type=int, default=100, help="打印日志步数")
        add_arg("eval_steps", type=int, default=1000, help="多少步数评估一次")
        add_arg("save_steps", type=int, default=1000, help="多少步数保存模型一次")
        add_arg("learning_rate", type=float, default=1e-5, help="学习率大小")
        add_arg("fp16", type=bool, default=True, help="是否使用fp16训练模型")
        add_arg("use_compile", type=bool, default=False, help="是否使用Pytorch2.0的编译器")
        add_arg("local_files_only", type=bool, default=True, help="是否只在本地加载模型, 不尝试下载")
        add_arg(
            "language",
            type=str,
            default="Chinese",
            help="设置语言, 可全称也可简写, 如果为None则训练的是多语言",
        )
        add_arg(
            "task",
            type=str,
            default="transcribe",
            choices=['transcribe', 'translate'],
            help="模型的任务",
        )
        add_arg(
            "use_fast_tokenizer",
            type=bool,
            default=True,
            help="是否用tokenizer包的fast tokenizer",
        )
        add_arg("resume_from_checkpoint", type=str, default=None, help="恢复训练的检查点路径")
        add_arg("gradient_accumulation_steps", type=int, default=1, help="梯度累积步数")
        add_arg("gradient_checkpointing", type=bool, default=False, help="梯度检查点")
        add_arg("max_grad_norm", type=float, default=5, help="梯度裁剪")
        add_arg(
            "remove_whisper_encoder_input_length_restriction",
            type=bool,
            default=False,
            help="支持变长输入",
        )
        add_arg(
            "empty_init",
            type=bool,
            default=True,
            help="快速初始化模型, 注意模型并行(zero3)必须关闭, GPU不够放下整个模型必须关闭",
        )
        add_arg("deepspeed", type=Optional[str], default=None, help="deepspeed配置文件")
        return parser

    @staticmethod
    def run_from_args(args, parser: Optional[CommonParser] = None):
        FullTune(args, parser)

    def __init__(
        self,
        args,
        parser,
    ):
        # 获取Whisper的数据处理器, 这个包含了特征提取器、tokenizer
        processor = WhisperProcessor.from_pretrained(
            args.base_model,
            language=args.language,
            task=args.task,
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
        data_collator = BatchProcessor(processor, feat_padding_to_max)

        # 后续处理lhotse数据到模型输入格式
        datamodule.attach_trainer_collator(data_collator)

        rank = _infer_rank() or 0
        # lhotse是迭代式dataset，遍历训练采样器得到更新步数，注意分布式要保证每个rank得到一样的值
        if (max_steps := args.max_steps) < 0:
            infer_step = datamodule.infer_sampler_len()
            steps_per_epoch = infer_step // args.gradient_accumulation_steps
            max_steps = steps_per_epoch * args.num_train_epochs

            logger.info(
                f"Steps per epoch {steps_per_epoch}\t max steps {max_steps}", ranks=rank
            )
        # 通过empty init快速初始化模型 (https://lernapparat.de/faster-model-init)
        device_map = (
            {"": int(os.environ.get("LOCAL_RANK") or 0)} if args.empty_init else None
        )

        # 获取模型
        model = WhisperForConditionalGeneration.from_pretrained(
            args.base_model,
            torch_dtype=torch.float16 if args.fp16 else "auto",
            device_map=device_map,
            local_files_only=args.local_files_only,
        )
        model.config.forced_decoder_ids = None
        model.config.suppress_tokens = []
        # 量化模型
        # model = prepare_model_for_kbit_training(model)
        # 注册forward, 否则多卡训练会失败
        model.model.encoder.conv1.register_forward_hook(make_inputs_require_grad)

        output_dir = args.output_dir
        # 定义训练参数
        training_args = Seq2SeqTrainingArguments(
            output_dir=output_dir,  # 保存检查点和意志的目录
            per_device_train_batch_size=args.max_cuts,  # 训练batch_size大小
            per_device_eval_batch_size=args.max_cuts,  # 评估batch_size大小
            gradient_accumulation_steps=args.gradient_accumulation_steps,  # 训练梯度累计步数
            gradient_checkpointing=args.gradient_checkpointing,
            max_grad_norm=args.max_grad_norm,
            learning_rate=args.learning_rate,  # 学习率大小
            warmup_steps=args.warmup_steps,  # 预热步数
            num_train_epochs=args.num_train_epochs,  # 微调训练轮数
            max_steps=max_steps,  # 步数，参数设定或者遍历得出
            save_strategy="steps",  # 指定按照步数保存检查点
            eval_strategy="steps",  # 指定按照步数评估模型
            load_best_model_at_end=True,  # 指定是否在结束时加载最优模型
            save_total_limit=5,  # 只保存最新检查点的数量
            fp16=args.fp16,  # 是否使用半精度训练
            report_to=["tensorboard"],  # 指定使用tensorboard保存log
            save_steps=args.save_steps,  # 指定保存检查点的步数
            eval_steps=args.eval_steps,  # 指定评估模型的步数
            torch_compile=args.use_compile,  # 使用Pytorch2.0的编译器
            optim=args.optim,  # 指定优化方法
            lr_scheduler_type=args.lr_scheduler_type,
            logging_steps=args.logging_steps,  # 指定打印log的步数
            deepspeed=args.deepspeed,
            label_names=["labels"],
        )

        if args.freeze_encoder:
            print('Model freeze encoder!')
            model.freeze_encoder()

        if training_args.local_rank == 0 or training_args.local_rank == -1:
            tmp_configdir = Path(args.output_dir) / "config"
            tmp_configdir.mkdir(exist_ok=True, parents=True)
            parser.save(
                args,
                tmp_configdir / DEFAULT_TRAIN_FILENAME,
                skip_none=True,
                overwrite=True,
            )
            print("=" * 90)
            print(f'trainable params: {model.num_parameters(only_trainable=True)}')
            print("=" * 90)
        # 定义训练器

        # 定义训练器
        trainer = CustomTrainer(
            args=training_args,
            model=model,
            tokenizer=processor.feature_extractor,
            callbacks=[LhotseCallback],
        )

        model.config.use_cache = False

        # 用自定义的lhotse数据处理
        trainer.datamodule = datamodule
        model.config.use_cache = False
        trainer._load_from_checkpoint = load_from_checkpoint
        if args.resume_from_checkpoint:
            patch_skip_batches()

        # 开始训练
        trainer.train(resume_from_checkpoint=args.resume_from_checkpoint)

        # 保存最后的模型
        trainer.save_state()
        save_directory = os.path.join(output_dir, "checkpoint-final")

        if training_args.local_rank == 0 or training_args.local_rank == -1:
            model.save_pretrained(save_directory)
            processor.save_pretrained(save_directory)
            parser.save(
                args,
                Path(output_dir) / "checkpoint-final" / DEFAULT_TRAIN_FILENAME,
                skip_none=True,
                overwrite=True,
            )
        release_memory()


if __name__ == "__main__":
    parser = FullTune.get_dummy_parser()
    parser = FullTune.setup_parser(parser)
    args = parser.parse_args()
    logger.info(
        f"Got parsed args: \n{parser.dump(args.clone(),skip_default=True)}", ranks=[0]
    )
    FullTune.run_from_args(args, parser)
