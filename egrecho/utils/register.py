# -*- encoding: utf-8 -*-
# Copyright xmuspeech (Author: Leo 2023-08)

from itertools import chain
from typing import Any, Callable, Dict, List, Optional, Tuple, Union

FUNCTION_ITEM = Dict[str, Any]


class Register:
    """This class is used to register functions or classes.

    Args:
        name (str): The name of the registry.

    Example::

        LITTLES = Register("littles")

        @LITTLES.register(name='litclass')
        class LitClass:
            def __init__(self, a, b=123):
                self.a = a
                self.b = b

        or

        LITTLES.register(LitClass, name='litclass')
    """

    def __init__(self, name: str):
        self._name = name
        self._registed: List[FUNCTION_ITEM] = []

    @property
    def name(self):
        return self._name

    def register(
        self,
        fn_or_cls: Optional[Callable] = None,
        name: Optional[str] = None,
        override: bool = False,
        **metadata,
    ) -> Callable:
        """Adds a function or class.

        Args:
            fn_or_cls (Callable): The function to be registered.
            name (str): Name string.
            override (bool): Whether override if exists.
            **metadata (dict): Additional dict to be saved.
        """
        if fn_or_cls is not None:
            return self._do_register(
                fn_or_cls=fn_or_cls, name=name, override=override, metadata=metadata
            )

        if name is not None and not isinstance(name, str):
            raise TypeError(f"`name` must be a str, but got {name}")

        # case for `@` wrapper.
        def register_wrapper(fn_or_cls):
            return self._do_register(
                fn_or_cls, name=name, override=override, metadata=metadata
            )

        return register_wrapper

    def _do_register(
        self,
        fn_or_cls: Callable,
        name: Optional[str] = None,
        override: bool = False,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> Callable:
        """Adds a function or class.

        Args:
            fn_or_cls (Callable): The function to be registered.
            name (str): Name string.
            override (bool): Whether override if exists.
            metadata (dict): Additional dict to be saved.
        """
        if not callable(fn_or_cls):
            raise TypeError(f"You can only register a callable, but got: {fn_or_cls}")
        if name is None:
            name = (
                fn_or_cls.func.__name__
                if hasattr(fn_or_cls, "func")
                else fn_or_cls.__name__
            )
        item = {"fn": fn_or_cls, "name": name, "metadata": metadata or {}}
        match_idx = self._match_idx(item)
        if override and match_idx is not None:
            self._registed[match_idx] = item
        else:
            if match_idx is not None:
                raise ValueError(
                    f"{name} with metadata: {metadata} is already in {self}, use: `override=True`."
                )
            self._registed.append(item)
        # return so as to use it normally if via importing
        return fn_or_cls

    def _match_idx(self, item: FUNCTION_ITEM):
        for idx, register_item in enumerate(self._registed):
            if all(register_item[k] == item[k] for k in ("name", "metadata")):
                return idx
        return None

    def get(
        self,
        key: str,
        with_metadata: bool = False,
        strict: bool = True,
        **metadata,
    ):
        """Retrieves functions or classes with key name which has already been registed before.

        Args:
            key: Name of the registered function.
            with_metadata: Whether to include the associated metadata in the return value.
            strict: Whether to return all matches or just one.
            metadata: Metadata used to filter against existing registry item's metadata.
        """
        matches = [e for e in self._registed if key == e["name"]]
        if not matches:
            raise KeyError(
                f"Key: {key} is not in {self}. Available keys: {self.keys()}"
            )

        if metadata:
            matches = [m for m in matches if metadata.items() <= m["metadata"].items()]
            if not matches:
                raise KeyError(
                    "Found no matches that fit your metadata criteria. Try removing some metadata"
                )

        matches = [e if with_metadata else e["fn"] for e in matches]
        return matches[0] if strict else matches

    def keys(self):
        """Get all name registred."""
        return sorted(item["name"] for item in self._registed)

    def remove(self, key: str) -> None:
        self._registed = [f for f in self._registed if f["name"] != key]

    def __add__(self, other) -> "ConcatRegister":
        registers = []
        if isinstance(self, ConcatRegister):
            registers += self.registers
        else:
            registers += [self]

        registers = (
            other.registers + tuple(registers)
            if isinstance(other, ConcatRegister)
            else [other] + registers
        )

        return ConcatRegister(*registers)

    def __len__(self) -> int:
        return len(self._registed)

    def __contains__(self, key) -> bool:
        return any(key == e["name"] for e in self._registed)

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(name={self.name}, registed={self._registed})"


class ExternalRegister(Register):
    """TODO"""

    remove = None
    _do_register = None


class ConcatRegister(Register):
    """This class is used to concatenate multiple registers."""

    def __init__(self, *registers: Register):
        super().__init__(
            ",".join(
                {
                    register.name
                    for register in sorted(
                        registers,
                        key=lambda r: 1 if isinstance(r, ExternalRegister) else 0,
                    )
                }
            )
        )

        self.registers = registers

    def _do_register(
        self,
        fn: Callable,
        name: Optional[str] = None,
        override: bool = False,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> Callable:
        """Register in the first available registry."""
        for register in self.registers:
            if (
                not isinstance(register, ExternalRegister)
                and getattr(register, "_do_register", None) is not None
            ):
                return register._do_register(
                    fn, name=name, override=override, metadata=metadata
                )
        raise RuntimeError(
            f"Faield register (name={name}, fn={fn}, metadata={metadata}) "
            f"to {self}. Probably because all registers are {ExternalRegister.__name__!r}, check it."
        )

    def get(
        self,
        key: str,
        with_metadata: bool = False,
        strict: bool = True,
        **metadata,
    ) -> Union[Callable, FUNCTION_ITEM, List[FUNCTION_ITEM], List[Callable]]:
        matches = []
        external_matches = []

        for register in self.registers:
            if key in register:
                result = register.get(
                    key, with_metadata=with_metadata, strict=strict, **metadata
                )
                if not isinstance(result, list):
                    result = [result]

                if isinstance(register, ExternalRegister):
                    external_matches += result
                else:
                    matches += result

        if not strict:
            return matches + external_matches

        if len(matches) > 0:
            return matches[0]

        if len(external_matches) == 1:
            return external_matches[0]

        if len(matches) == 0 and len(external_matches) == 0:
            raise KeyError("No matches found in registry.")
        raise KeyError(
            "Multiple matches from external registers, a strict lookup is not possible."
        )

    def keys(self) -> List[str]:
        return list(chain.from_iterable(register.keys() for register in self.registers))

    def remove(self, key: str) -> None:
        for register in self.registers:
            if (
                key in register
                and not isinstance(register, ExternalRegister)
                and getattr(register, "remove", None) is not None
            ):
                register.remove(key)

    def __len__(self) -> int:
        return sum(len(register) for register in self.registers)

    def __contains__(self, key) -> bool:
        return any(key in register for register in self.registers)

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(registers={self.registers})"


class StrRegister:
    """Registers multiple strings, which can be used to create alias names.

    Args:
        name (str): The name of the registry.

    Example::

        TOTAL_STEPS = StrRegister("total_steps")

        TOTAL_STEPS.register("num_training_steps")

        or

        TOTAL_STEPS.register(["num_training_steps",])

        assert TOTAL_STEPS.keys() == ["total_steps","num_training_steps"]
        assert "num_training_steps" in TOTAL_STEPS
    """

    def __init__(self, name: str):
        self._name = name
        self._registed: List[str] = [
            name,
        ]

    @property
    def name(self):
        return self._name

    def register(
        self,
        str_or_strs: Union[str, List, Tuple],
    ) -> Union[str, List[str]]:
        """Adds a string or sequence of strings.

        Args:
            str_or_strs: String(s) to be registed.
        """
        msg = f"`str_or_strs` must be a str or list/tuple of strings, but got {str_or_strs}"
        if not isinstance(str_or_strs, str) and not isinstance(
            str_or_strs, (list, tuple)
        ):
            raise ValueError(msg)
        if isinstance(str_or_strs, (list, tuple)) and any(
            not isinstance(s, str) for s in str_or_strs
        ):
            raise ValueError(msg)
        return self._do_register(str_or_strs)

    def _do_register(
        self,
        str_or_strs: Union[str, List[str]],
    ) -> Union[str, List[str]]:
        if isinstance(str_or_strs, str):
            str_or_strs = (str_or_strs,)
        for s in str_or_strs:
            if s not in self._registed:
                self._registed.append(s)

        return str_or_strs

    def get(
        self,
        key: str,
        default: Optional[str] = None,
    ) -> Optional[str]:
        if key in self._registed:
            return key
        return default

    def keys(self):
        """Get all name registred."""
        return self._registed

    def remove(self, key: str) -> None:
        self._registed = [f for f in self._registed if f != key]

    def __add__(self, other) -> "ConcatStrRegister":
        registers = []
        if isinstance(self, ConcatStrRegister):
            registers += self.registers
        else:
            registers += [self]

        registers = (
            other.registers + tuple(registers)
            if isinstance(other, ConcatStrRegister)
            else [other] + registers
        )

        return ConcatStrRegister(*registers)

    def __len__(self) -> int:
        return len(self._registed)

    def __contains__(self, key) -> bool:
        return key in self._registed

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(name={self.name}, registed={self._registed})"


class ConcatStrRegister(StrRegister):
    """This class is used to concatenate multiple registers."""

    def __init__(self, *registers: StrRegister):
        super().__init__(",".join({register.name for register in registers}))

        self.registers = registers

    def _do_register(
        self,
        str_or_strs: Union[str, List[str]],
    ) -> Union[str, List[str]]:
        """Register in the first available registry."""
        for register in self.registers:
            return register._do_register(str_or_strs)

    def get(
        self,
        key: str,
        default: Optional[str] = None,
    ):
        for register in self.registers:
            if key in register:
                return key
        return default

    def keys(self) -> List[str]:
        return list(chain.from_iterable(register.keys() for register in self.registers))

    def remove(self, key: str) -> None:
        for register in self.registers:
            if key in register:
                register.remove(key)

    def __len__(self) -> int:
        return sum(len(register) for register in self.registers)

    def __contains__(self, key) -> bool:
        return any(key in register for register in self.registers)

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(registers={self.registers})"
