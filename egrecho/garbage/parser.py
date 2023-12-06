import argparse
import dataclasses
from argparse import ArgumentParser, ArgumentTypeError
from copy import copy
from inspect import isclass
from typing import (
    Any,
    Callable,
    Dict,
    List,
    Literal,
    NewType,
    Optional,
    Union,
    get_type_hints,
)

from egrecho.utils import asdict_filt

DataClass = NewType("DataClass", Any)
DataClassType = NewType("DataClassType", Any)


class CommonParser(ArgumentParser):
    """
    Almost inherit from `argparse.ArgumentParser`, specify some params.
    Support parse args defined as dataclasses.
    """

    def __init__(self, description=None, conflict_handler="resolve", **kwargs):
        kwargs["formatter_class"] = kwargs.get("formatter_class", DefaultHelpFormmer)
        super().__init__(
            description=description,
            conflict_handler=conflict_handler,
            allow_abbrev=False,
            **kwargs,
        )
        self.register("type", "bool", string_to_bool)

    def add_dataclass_arguments(self, dtype: DataClassType):
        """
        This implementation is based on `huggingface`.
        dataclasses parser:
        https://github.com/huggingface/transformers/blob/main/src/transformers/hf_argparser.py
        """
        if hasattr(dtype, "_argument_group_name"):
            parser = self.add_argument_group(dtype._argument_group_name)
        else:
            parser = self

        try:
            type_hints: Dict[str, type] = get_type_hints(dtype)
        except NameError:
            raise RuntimeError(
                f"Type resolution failed for f{dtype}. Try declaring the class in global scope or "
                "removing line of `from __future__ import annotations` which opts in Postponed "
                "Evaluation of Annotations (PEP 563)"
            )

        for field in dataclasses.fields(dtype):
            if not field.init:
                continue
            field.type = type_hints[field.name]
            self._parse_dataclass_field(parser, field)

    @staticmethod
    def _parse_dataclass_field(parser: ArgumentParser, field: dataclasses.Field):
        field_name = f"--{field.name}"
        kwargs = field.metadata.copy()
        # field.metadata is not used at all by Data Classes,
        # it is provided as a third-party extension mechanism.
        if isinstance(field.type, str):
            raise RuntimeError(
                "Unresolved type detected, which should have been done with the help of "
                "`typing.get_type_hints` method by default"
            )

        aliases = kwargs.pop("aliases", [])
        if isinstance(aliases, str):
            aliases = [aliases]

        origin_type = getattr(field.type, "__origin__", field.type)
        if origin_type is Union:
            if str not in field.type.__args__ and (
                len(field.type.__args__) != 2 or type(None) not in field.type.__args__
            ):
                raise ValueError(
                    "Only `Union[X, NoneType]` (i.e., `Optional[X]`) is allowed for `Union` because"
                    " the argument parser only supports one type per argument."
                    f" Problem encountered in field '{field.name}'."
                )
            if type(None) not in field.type.__args__:
                # filter `str` in Union
                field.type = (
                    field.type.__args__[0]
                    if field.type.__args__[1] == str
                    else field.type.__args__[1]
                )
                origin_type = getattr(field.type, "__origin__", field.type)
            elif bool not in field.type.__args__:
                # filter `NoneType` in Union (except for `Union[bool, NoneType]`)
                field.type = (
                    field.type.__args__[0]
                    if isinstance(None, field.type.__args__[1])
                    else field.type.__args__[1]
                )
                origin_type = getattr(field.type, "__origin__", field.type)

        # A variable to store kwargs for a boolean field, if needed
        # so that we can init a `no_*` complement argument (see below)
        bool_kwargs = {}
        if origin_type is Literal or (
            isinstance(field.type, type) and issubclass(field.type, Enum)
        ):
            if origin_type is Literal:
                kwargs["choices"] = field.type.__args__
            else:
                kwargs["choices"] = [x.value for x in field.type]

            kwargs["type"] = make_choice_type_function(kwargs["choices"])

            if field.default is not dataclasses.MISSING:
                kwargs["default"] = field.default
            else:
                kwargs["required"] = True
        elif field.type is bool or field.type == Optional[bool]:
            # Copy the currect kwargs to use to instantiate a `no_*` complement argument below.
            # We do not initialize it here because the `no_*` alternative must be instantiated after the real argument
            bool_kwargs = copy(kwargs)

            # Hack because type=bool in argparse does not behave as we want.
            kwargs["type"] = string_to_bool
            if field.type is bool or (
                field.default is not None and field.default is not dataclasses.MISSING
            ):
                # Default value is False if we have no default when of type bool.
                default = (
                    False if field.default is dataclasses.MISSING else field.default
                )
                # This is the value that will get picked if we don't include --field_name in any way
                kwargs["default"] = default
                # This tells argparse we accept 0 or 1 value after --field_name
                kwargs["nargs"] = "?"
                # This is the value that will get picked if we do --field_name (without value)
                kwargs["const"] = True
        elif isclass(origin_type) and issubclass(origin_type, list):
            kwargs["type"] = field.type.__args__[0]
            kwargs["nargs"] = "+"
            if field.default_factory is not dataclasses.MISSING:
                kwargs["default"] = field.default_factory()
            elif field.default is dataclasses.MISSING:
                kwargs["required"] = True
        else:
            kwargs["type"] = field.type
            if field.default is not dataclasses.MISSING:
                kwargs["default"] = field.default
            elif field.default_factory is not dataclasses.MISSING:
                kwargs["default"] = field.default_factory()
            else:
                kwargs["required"] = True
        parser.add_argument(field_name, *aliases, **kwargs)

        # Add a complement `no_*` argument for a boolean field AFTER the initial field has already been added.
        # Order is important for arguments with the same destination!
        # We use a copy of earlier kwargs because the original kwargs have changed a lot before reaching down
        # here and we do not need those changes/additional keys.
        if field.default is True and (
            field.type is bool or field.type == Optional[bool]
        ):
            bool_kwargs["default"] = False
            parser.add_argument(
                f"--no_{field.name}",
                action="store_false",
                dest=field.name,
                **bool_kwargs,
            )

    @staticmethod
    def parse_dataclass(args: argparse.Namespace, dtype: DataClassType) -> DataClass:
        keys = {f.name for f in dataclasses.fields(dtype) if f.init}
        inputs = {
            k: v
            for k, v in asdict_filt(vars(args), flit_type="none").items()
            if k in keys
        }
        obj = dtype(**inputs)
        return obj


class DefaultHelpFormmer(argparse.HelpFormatter):
    """
    A `HelpFormatter` which combines `argparse.ArgumentDefaultsHelpFormatter` and `argparse.RawTextHelpFormatter`
    """

    def __init__(self, *args, **kwargs) -> None:
        kwargs["max_help_position"] = kwargs.get("max_help_position", 8)
        super().__init__(*args, **kwargs)

    def _get_help_string(self, action):
        help = action.help
        if "%(default)" not in action.help:
            if action.default is not argparse.SUPPRESS:
                defaulting_nargs = [argparse.OPTIONAL, argparse.ZERO_OR_MORE]
                if action.option_strings or action.nargs in defaulting_nargs:
                    help += " (default: %(default)s)"
        return help

    def _split_lines(self, text, width):
        return text.splitlines()

    def _fill_text(self, text, width, indent):
        return "".join(indent + line for line in text.splitlines(keepends=True))

    def _format_action_invocation(self, action):
        if not action.option_strings or action.nargs == 0:
            return super()._format_action_invocation(action)
        else:
            default = self._get_default_metavar_for_optional(action)
            args_string = self._format_args(action, default)

            return ", ".join(action.option_strings) + " " + args_string


def string_to_bool(v):
    if isinstance(v, bool):
        return v
    if v.lower() in ("yes", "true", "t", "y", "1"):
        return True
    elif v.lower() in ("no", "false", "f", "n", "0"):
        return False
    else:
        raise ArgumentTypeError(
            f"Truthy value expected: got {v} but expected one of yes/no, true/false, t/f, y/n, 1/0 (case insensitive)."
        )


def make_choice_type_function(choices: list) -> Callable[[str], Any]:
    """
    Creates a mapping function from each choices string representation to the actual value. Used to support multiple
    value types for a single argument.

    Args:
        choices (list): List of choices.

    Returns:
        Callable[[str], Any]: Mapping function from string representation to actual value for each choice.
    """
    str_to_choice = {str(choice): choice for choice in choices}
    return lambda arg: str_to_choice.get(arg, arg)


def DataClassArg(
    *,
    aliases: Union[str, List[str]] = None,
    help: str = None,
    default: Any = dataclasses.MISSING,
    default_factory: Callable[[], Any] = dataclasses.MISSING,
    metadata: dict = None,
    **kwargs,
) -> dataclasses.Field:
    """Argument helper enabling a concise syntax to create dataclass fields for parsing with `HfArgumentParser`.

    Example comparing the use of `HfArg` and `dataclasses.field`:
    ```
    @dataclass
    class Args:
        regular_arg: str = dataclasses.field(default="Huggingface", metadata={"aliases": ["--example", "-e"], "help": "This syntax could be better!"})
        hf_arg: str = HfArg(default="Huggingface", aliases=["--example", "-e"], help="What a nice syntax!")
    ```

    Args:
        aliases (Union[str, List[str]], optional):
            Single string or list of strings of aliases to pass on to argparse, e.g. `aliases=["--example", "-e"]`.
            Defaults to None.
        help (str, optional): Help string to pass on to argparse that can be displayed with --help. Defaults to None.
        default (Any, optional):
            Default value for the argument. If not default or default_factory is specified, the argument is required.
            Defaults to dataclasses.MISSING.
        default_factory (Callable[[], Any], optional):
            The default_factory is a 0-argument function called to initialize a field's value. It is useful to provide
            default values for mutable types, e.g. lists: `default_factory=list`. Mutually exclusive with `default=`.
            Defaults to dataclasses.MISSING.
        metadata (dict, optional): Further metadata to pass on to `dataclasses.field`. Defaults to None.

    Returns:
        Field: A `dataclasses.Field` with the desired properties.
    """
    if metadata is None:
        # Important, don't use as default param in function signature because dict is mutable and shared across function calls
        metadata = {}
    if aliases is not None:
        metadata["aliases"] = aliases
    if help is not None:
        metadata["help"] = help

    return dataclasses.field(
        metadata=metadata, default=default, default_factory=default_factory, **kwargs
    )
