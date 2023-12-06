# -*- coding:utf-8 -*-
# Copyright xmuspeech (Author: Leo 2023-9)

from collections import namedtuple
from typing import Dict, Optional

command_regeistry = namedtuple("command_regeistry", ("cls", "aliases", "help"))
COMMAND_REGISTRY: Dict[str, command_regeistry] = {}


def register_command(
    name: str,
    aliases: Optional[str] = None,
    help: Optional[str] = None,
):
    """
    Register cli commands.

    ::

        @register_command('cli_name')
        class CommandClass(BaseCommand):
        ...
    """
    if aliases is None:
        aliases = []

    def wrapper(cls_):
        global COMMAND_REGISTRY
        assert (
            name not in COMMAND_REGISTRY
        ), f"cli name:{name} is already known, change another name."
        COMMAND_REGISTRY[name] = command_regeistry(cls_, aliases, help)
        return cls_

    return wrapper
