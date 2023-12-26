# -*- coding:utf-8 -*-
# Copyright xmuspeech (Author: Leo 2023-12)

from typing import Optional

from egrecho.core.parser import BaseCommand, CommonParser
from egrecho.training.average_models import average_best_models
from egrecho_cli.register import register_command

DESCRIPTION = """Average best models"""


@register_command(name="avg-best", aliases=[], help=DESCRIPTION)
class AverageBest(BaseCommand):
    @classmethod
    def get_dummy_parser(cls) -> CommonParser:
        return CommonParser(description=DESCRIPTION)

    @classmethod
    def setup_parser(cls, parser: CommonParser):
        parser.add_cfg_flag()

        parser.add_function_arguments(average_best_models, as_positional=True)

        return parser

    @staticmethod
    def run_from_args(args, parser: Optional[CommonParser] = None):
        args.pop("cfg", None)
        average_best_models(**args)


if __name__ == "__main__":
    parser = AverageBest.get_dummy_parser()
    parser = AverageBest.setup_parser(parser)
    args = parser.parse_args()
    AverageBest.run_from_args(args, parser)
