# -*- coding:utf-8 -*-
# Copyright xmuspeech (Author: Leo 2023-9)

import importlib
import os
import sys
import traceback
import warnings
from typing import Dict, Type

from egrecho.core.parser import BaseCommand, CommonParser
from egrecho.utils.imports import _PL_AVAILABLE
from egrecho.utils.io import DataFilesList
from egrecho.utils.logging import get_logger
from egrecho.utils.misc import imports_local
from egrecho_cli.register import COMMAND_REGISTRY

logger = get_logger()
CONMMANDS_NAMESPACE = "egrecho_cli"
SCRIPT_PATTERNS = "commands" + os.path.sep + "**.py"


def setup_registry():
    """Register available commands."""
    # imports 'dynamic_egrecho'
    imports_local()

    cli_module = importlib.import_module(CONMMANDS_NAMESPACE)
    cli_base_path = cli_module.__file__.replace("__init__.py", "")
    scripts = DataFilesList.from_local_or_remote(
        SCRIPT_PATTERNS, cli_base_path, skip_metadata=True
    )
    script_modules = [
        os.path.join("egrecho_cli", os.path.relpath(s, cli_base_path))
        .replace(os.path.sep, ".")
        .replace(".py", "")
        for s in scripts
    ]
    for script_module in script_modules:
        if "pl_" in script_module and not _PL_AVAILABLE:
            warnings.warn(
                f'Skip ({script_module!r}), scripts with prefix of "pl_" means you need install pytorch lightning.'
            )
            continue  # need pl
        # elif "pl_" in script_module:
        #     try:
        #         importlib.import_module(script_module)
        #     except Exception as ex:
        #         warnings.warn(
        #             f"{str(ex)}\nSome errors occurs, skip lightning based script: ({script_module!r})."
        #         )
        #         continue
        else:
            importlib.import_module(script_module)


def main():
    # TODO: Instead of register all clis, define a supparser to register clis via groups.
    # e.g.:
    # supparser = ArgumentParser()
    # supparser.add_argument(...)
    # ...
    # args, remains = supparser.parse_args()
    # setup_registry(args.group)

    # Register cli classes.
    setup_registry()

    parser = CommonParser(
        description="egrecho CLI",
        usage="egrecho <commands> [args]",
        conflict_handler="error",
    )
    parser.add_cfg_flag()
    command_parsers = parser.add_subcommands(
        dest="subcommand",
        required=False,
        title="Can export default config file if subcommand support --print_config "
        "(egrecho conmmand --print_config(=skil_null)), then modify your train.yaml, "
        "e.g., egrecho train-asv --print_config run --model=EcapaModel > train.yaml, "
        "egrecho train-asv -c train.yaml run -h, egrecho train-asv -c=train.yaml -h",
    )
    named_cli_cls: Dict[str, Type[BaseCommand]] = {}
    named_parsers: Dict[str, CommonParser] = {}

    # Register subcommands.
    for command_name, command_registry in COMMAND_REGISTRY.items():
        # try setup parser to find errors.
        try:
            trial_parser = command_registry.cls.get_dummy_parser()
            command_registry.cls.setup_parser(trial_parser)
        except Exception as ex:  # noqa
            msg = f"{ex}\n[extra info] Faield to setup the discovered command {command_name}.\nSkip it."
            warnings.warn(f"{type(ex)}{msg}", RuntimeWarning)
            continue

        # if no error, add subcommand
        sub_parser = command_registry.cls.get_dummy_parser()
        sub_parser = sub_parser if sub_parser else CommonParser()
        command_parsers.add_subcommand(
            command_name,
            sub_parser,
            aliases=command_registry.aliases,
            help=command_registry.help or sub_parser.description,
        )

        sub_parser.prog = sub_parser.prog.replace("<commands> [args] [options] ", "")
        command_registry.cls.setup_parser(sub_parser)

        # Record subcommands to execute later.
        named_cli_cls[command_name] = command_registry.cls
        named_parsers[command_name] = sub_parser
        for alias in command_registry.aliases:
            named_cli_cls[alias] = command_registry.cls
            named_parsers[alias] = sub_parser

    args = parser.parse_args()

    command_name: str = getattr(args, "subcommand", None)

    if not command_name:
        parser.print_help()
        sys.exit(1)
    logger.info(
        f"Got parsed args: \n{parser.dump(args.clone(),skip_default=True)}", ranks=[0]
    )
    sub_args = getattr(args, command_name, args)

    # run!
    try:
        named_cli_cls[command_name].run_from_args(sub_args, named_parsers[command_name])
    except KeyboardInterrupt as exc:  # noqa
        msg = f"#### Keyboard exit ({command_name}): {' '.join(sys.argv)}"
        print(msg)
        sys.exit(1)
    except Exception as exc:
        traceback.print_exc()
        msg = f"#### Run command ({command_name}) error: {' '.join(sys.argv)}"
        print(msg)
        sys.exit(1)


# entry
if __name__ == "__main__":
    main()
