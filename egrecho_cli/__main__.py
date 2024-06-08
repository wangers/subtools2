# -*- coding:utf-8 -*-
# Copyright xmuspeech (Author: Leo 2023-9)

import importlib
import os
import sys
import traceback
import warnings
from pathlib import Path
from typing import Dict, Optional, Type, Union

from egrecho.core.parser import BaseCommand, CommonParser
from egrecho.utils.imports import _PL_AVAILABLE
from egrecho.utils.io import DataFilesList
from egrecho.utils.logging import get_logger
from egrecho_cli.register import COMMAND_REGISTRY

logger = get_logger()
CONMMANDS_NAMESPACE = "egrecho_cli"
SCRIPT_PATTERNS = "commands" + os.path.sep + "**.py"


def source_import(file_path):
    """This function imports python module directly from source code using importlib"""
    spec = importlib.util.spec_from_file_location('', file_path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def imports_local(localdir: Optional[Union[str, Path]] = None):
    from egrecho.utils.constants import LOCAL_EGRECHO

    localdir = LOCAL_EGRECHO if localdir is None else localdir

    local_module_path = Path(localdir)

    msg = ''
    if local_module_path.exists() and local_module_path.is_dir():
        local_module_path = local_module_path.absolute()

        init_path = local_module_path / "__init__.py"

        if not init_path.exists():
            init_path.touch()

        try:
            sys.path.append(str(local_module_path))
            impted = source_import(str(init_path))
            logger.info(
                f"Detected local package {localdir} and added it to env, success exec {impted}.\n",
                ranks=0,
            )
        except ImportError as e:
            msg = f"{e}\nFailed imports local source {localdir}."
            warnings.warn(msg)
        except Exception as e:
            msg = f"{e}\nSome Errors occurred in {localdir}.__init__.py ."
            warnings.warn(msg)
    return msg


def setup_registry():
    """Register available commands."""

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
    cli_registry_warns = []
    for script_module in script_modules:
        if "pl_" in script_module and not _PL_AVAILABLE:
            msg = f'Skip ({script_module!r}), scripts with prefix of "pl_" means you need install pytorch lightning.'
            warnings.warn(msg)
            cli_registry_warns.append(msg)
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
            try:
                importlib.import_module(script_module)
            except Exception as ex:
                msg = (
                    f"{str(ex)}\nSome errors occurs, skip script: ({script_module!r})."
                )
                warnings.warn(msg)
                cli_registry_warns.append(msg)
                continue
    # imports 'egrecho_inner'
    egrecho_inner_msg = imports_local()
    if egrecho_inner_msg:
        cli_registry_warns.append(egrecho_inner_msg)

    return cli_registry_warns


def main():
    # TODO: Instead of register all clis, define a supparser to register clis via groups.
    # e.g.:
    # supparser = ArgumentParser()
    # supparser.add_argument(...)
    # ...
    # args, remains = supparser.parse_args()
    # setup_registry(args.group)

    # Register cli classes.
    cli_registry_warns = setup_registry()

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
            cli_registry_warns.append(msg)
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

    if cli_registry_warns:
        warns = ''
        for warn in cli_registry_warns:
            warns += '#### ' + warn + '\n'
        logger.warning(f"Got cli registry warn msgs: \n{warns}")

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
