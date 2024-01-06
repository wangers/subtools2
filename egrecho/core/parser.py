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
    Almost inherit from ``jsonargparse.ArgumentParser``.
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
            instantiate: Whether the class group should be instantiated by :meth:`instantiate_classes`.
            \**kwargs: other args will pass to
                :meth:`add_subclass_arguments`/:meth:`add_class_arguments` of jsonargparser.
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


class JsnDefaultHelpFormatter(DefaultHelpFormatter):
    def __init__(self, *args, **kwargs) -> None:
        kwargs["max_help_position"] = kwargs.get("max_help_position", 8)
        super().__init__(*args, **kwargs)

    def _fill_text(self, text, width, indent):
        return "".join(indent + line for line in text.splitlines(keepends=True))


class BaseCommand(metaclass=ABCMeta):
    """Provides a standardized form of command line class.

    This class is intend to build a cli system. Core methods to be implemented:

    - :meth:`get_dummy_parser`: setup a dummy parser without arguments.
    - :meth:`setup_parser`: add args to setup the parser. (**must be implemented**)
    - :meth:`run_from_args`: Run the program. (**must be implemented**)

    Workflow follows:

        :meth:`get_dummy_parser` -> :meth:`setup_parser` -> parse args -> :meth:`run_from_args`

    Example::

        from egrecho.core.parser import BaseCommand, CommonParser

        def hello(name: str):
            print(f"Hello {name}!")


        class Hello(BaseCommand):
            @classmethod
            def get_dummy_parser(cls):
                return CommonParser(description='Greet')

            @classmethod
            def setup_parser(cls, parser: CommonParser):
                parser.add_function_arguments(hello)
                return parser

            @staticmethod
            def run_from_args(args, parser=None):
                hello(**args)


        if __name__ == "__main__":
            parser = Hello.get_dummy_parser()
            parser = Hello.setup_parser(parser)
            args = parser.parse_args()
            Hello.run_from_args(args, parser)


        or


        # in ipynb
        parser = Hello.get_dummy_parser()
        parser = Hello.setup_parser(parser)
        args = parser.parse_args(['--name=Word'])
        Hello.run_from_args(args, parser)  # Hello Word!
    """

    @classmethod
    def get_dummy_parser(cls) -> ArgumentParser:
        """
        Get a dummy parser to be set in :meth:`setup_parser`
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
