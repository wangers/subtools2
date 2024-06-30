import functools
import os
from pathlib import Path
from typing import Optional

from peft import PeftConfig, PeftModel
from transformers import (
    WhisperFeatureExtractor,
    WhisperForConditionalGeneration,
    WhisperProcessor,
    WhisperTokenizerFast,
)
from utils.utils import add_arguments

from egrecho.core.parser import BaseCommand, CommonParser
from egrecho.utils.logging import get_logger
from egrecho_cli import register_command

logger = get_logger()

DESCRIPTION = "Merge lora whisper."


@register_command(name="merge-lora-whisper", aliases=["mlw"])
class MergeLoraWhisper(BaseCommand):
    @classmethod
    def get_dummy_parser(cls) -> CommonParser:
        return CommonParser(description=DESCRIPTION)

    @classmethod
    def setup_parser(cls, parser: CommonParser):
        parser.add_cfg_flag()
        add_arg = functools.partial(add_arguments, argparser=parser)
        add_arg(
            "lora_model",
            type=str,
            default="exp/whisper-medium/lora/checkpoint-best/",
            help="微调保存的模型路径",
        )
        add_arg("output_dir", type=Optional[str], default=None, help="合并模型的保存目录")
        add_arg("local_files_only", type=bool, default=True, help="是否只在本地加载模型，不尝试下载")
        return parser

    @staticmethod
    def run_from_args(args, parser=None):
        MergeLoraWhisper(args)

    def __init__(
        self,
        args,
    ):
        # 检查模型文件是否存在
        assert os.path.exists(args.lora_model), f"模型文件{args.lora_model}不存在"
        output_dir = args.output_dir or Path(args.lora_model).parent

        # 获取Lora配置参数
        peft_config = PeftConfig.from_pretrained(args.lora_model)
        # 获取Whisper的基本模型
        base_model = WhisperForConditionalGeneration.from_pretrained(
            peft_config.base_model_name_or_path,
            device_map={"": "cpu"},
            local_files_only=args.local_files_only,
        )
        # 与Lora模型合并
        model = PeftModel.from_pretrained(
            base_model, args.lora_model, local_files_only=args.local_files_only
        )
        feature_extractor = WhisperFeatureExtractor.from_pretrained(
            peft_config.base_model_name_or_path, local_files_only=args.local_files_only
        )
        tokenizer = WhisperTokenizerFast.from_pretrained(
            peft_config.base_model_name_or_path, local_files_only=args.local_files_only
        )
        processor = WhisperProcessor.from_pretrained(
            peft_config.base_model_name_or_path, local_files_only=args.local_files_only
        )

        # 合并参数
        model = model.merge_and_unload()
        model.train(False)

        # 保存的文件夹路径
        ckpt = str(Path(args.lora_model).name) + "-lora-merge"
        save_directory = os.path.join(output_dir, ckpt)

        os.makedirs(save_directory, exist_ok=True)

        # 保存模型到指定目录中
        model.save_pretrained(save_directory)
        feature_extractor.save_pretrained(save_directory)
        tokenizer.save_pretrained(save_directory)
        processor.save_pretrained(save_directory)
        print(f"合并模型保存在：{save_directory}")


if __name__ == "__main__":
    parser = MergeLoraWhisper.get_dummy_parser()
    parser = MergeLoraWhisper.setup_parser(parser)
    args = parser.parse_args()
    logger.info(
        f"Got parsed args: \n{parser.dump(args.clone(),skip_default=True)}", ranks=[0]
    )
    MergeLoraWhisper.run_from_args(args, parser)
