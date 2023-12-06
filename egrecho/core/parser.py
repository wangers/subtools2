# -*- coding:utf-8 -*-

# Copyright xmuspeech (Author: Leo 2023-04-11)


from abc import ABCMeta
from typing import Optional, Type

from jsonargparse import (  # noqa
    ActionConfigFile,
    ArgumentParser,
    DefaultHelpFormatter,
    Namespace,
)


class CommonParser(ArgumentParser):
    """
    Almost inherit from `jsonargparse.ArgumentParser`.
    """

    def __init__(
        self, *args, description="Egrecho CLI", conflict_handler="resolve", **kwargs
    ):
        kwargs["formatter_class"] = kwargs.get(
            "formatter_class", JsnDefaultHelpFormatter
        )
        super().__init__(
            *args,
            description=description,
            conflict_handler=conflict_handler,
            allow_abbrev=False,
            **kwargs,
        )

        self.exec_fac = None

    def add_cfg_flag(self):
        self.add_argument(
            "-c",
            "--cfg",
            action=ActionConfigFile,
            help="Path to a configuration file yaml format.",
        )

    def add_class_args(
        self,
        theclass: Type,
        nested_key: Optional[str] = None,
        subclass_mode: bool = False,
        instantiate: bool = True,
        default=None,
        **kwargs,
    ):
        """A convenient access of add class/subclass arguments.

        Args:
            theclass: Class from which to add arguments.
            nested_key: Key for nested namespace.
            subclass_mode: Whether allow any subclass of the given class.
            instantiate: Whether the class group should be instantiated by :code:`instantiate_classes`.
            **kwargs: other args will pass to `add_subclass_arguments/add_class_arguments` of jsonargparser.
        """
        if subclass_mode:
            return self.add_subclass_arguments(
                theclass,
                nested_key,
                required=True,
                instantiate=instantiate,
                default=default,
                **kwargs,
            )
        return self.add_class_arguments(
            theclass,
            nested_key,
            fail_untyped=False,
            instantiate=instantiate,
            default=default,
            **kwargs,
        )

    # def parse_orig_known_args(  # type: ignore[override]
    #     self,
    #     args: Optional[Sequence[str]] = None,
    #     namespace: Optional[Namespace] = None,
    #     env: Optional[bool] = None,
    #     defaults: bool = True,
    #     with_meta: Optional[bool] = None,
    #     **kwargs,
    # ) -> Namespace:
    #     """Parses command line argument strings.

    #     All the arguments from `argparse.ArgumentParser.parse_args
    #     <https://docs.python.org/3/library/argparse.html#argparse.ArgumentParser.parse_args>`_
    #     are supported. Additionally it accepts:

    #     Args:
    #         args: List of arguments to parse or None to use sys.argv.
    #         env: Whether to merge with the parsed environment, None to use parser's default.
    #         defaults: Whether to merge with the parser's defaults.
    #         with_meta: Whether to include metadata in config object, None to use parser's default.

    #     Returns:
    #         A config object with all parsed values.

    #     Raises:
    #         ArgumentError: If the parsing fails error and exit_on_error=True.
    #     """
    #     skip_check = get_private_kwargs(kwargs, _skip_check=False)
    #     return_parser_if_captured(self)
    #     argcomplete_autocomplete(self)

    #     if args is None:
    #         args = sys.argv[1:]
    #     else:
    #         args = list(args)
    #         if not all(isinstance(a, str) for a in args):
    #             self.error(f"All arguments are expected to be strings: {args}")
    #     self.args = args

    #     try:
    #         cfg = self._parse_defaults_and_environ(defaults, env)
    #         if namespace:
    #             cfg = self.merge_config(namespace, cfg)

    #         with _ActionSubCommands.parse_kwargs_context(
    #             {"env": env, "defaults": defaults}
    #         ):
    #             cfg, unk = super().parse_known_args(args=args, namespace=cfg)
    #         # if unk:
    #         #     self.error(f'Unrecognized arguments: {" ".join(unk)}')

    #         # parsed_cfg = self._parse_common(
    #         #     cfg=cfg,
    #         #     env=env,
    #         #     defaults=defaults,
    #         #     with_meta=with_meta,
    #         #     skip_check=skip_check,
    #         # )

    #     except (TypeError, KeyError) as ex:
    #         self.error(str(ex), ex)

    #     self._logger.debug("Parsed command line arguments: %s", args)
    #     return cfg, unk


class JsnDefaultHelpFormatter(DefaultHelpFormatter):
    def __init__(self, *args, **kwargs) -> None:
        kwargs["max_help_position"] = kwargs.get("max_help_position", 8)
        super().__init__(*args, **kwargs)

    def _fill_text(self, text, width, indent):
        return "".join(indent + line for line in text.splitlines(keepends=True))


class BaseCommand(metaclass=ABCMeta):
    @classmethod
    def get_dummy_parser(cls) -> ArgumentParser:
        """
        Get a dummy parser to be set in :method:`setup_parser`
        """
        ...

    @classmethod
    def setup_parser(cls, parser: CommonParser) -> ArgumentParser:
        """
        Setup a parser.
        """
        raise NotImplementedError

    @staticmethod
    def run_from_args(args: Namespace, parser: Optional[CommonParser] = None, **kwargs):
        """
        Run this command with args.
        """
        raise NotImplementedError
