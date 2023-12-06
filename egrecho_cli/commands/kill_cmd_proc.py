# -*- coding:utf-8 -*-
# Copyright xmuspeech (Author: Leo 2023-9)

from typing import Optional

from egrecho.core.parser import BaseCommand, CommonParser
from egrecho.utils.misc import kill_name_proc
from egrecho_cli.register import register_command

DESCRIPTION = """Find processes using `ps aux | grep` and terminate them.
This can be helpful in case of hanging multiprocesses."""


@register_command(name="kcp", aliases=["kill_cmd_process"], help=DESCRIPTION)
class KillCmdProcess(BaseCommand):
    @classmethod
    def get_dummy_parser(cls) -> CommonParser:
        return CommonParser(description=DESCRIPTION)

    @classmethod
    def setup_parser(cls, parser: CommonParser):
        parser.add_argument(
            "grep_str",
            type=str,
            help="The string after 'grep' to search for processes.",
        )
        parser.add_argument(
            "--force-kill",
            "-k",
            action="store_true",
            help="If provided, uses SIGKILL (kill -9) to forcefully terminate processes.",
        )

        return parser

    @staticmethod
    def run_from_args(args, parser: Optional[CommonParser] = None):
        kill_name_proc(args.grep_str, force_kill=args.force_kill)


if __name__ == "__main__":
    parser = KillCmdProcess.get_dummy_parser()
    parser = KillCmdProcess.setup_parser(parser)
    args = parser.parse_args()
    KillCmdProcess.run_from_args(args, parser)
