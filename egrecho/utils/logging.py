# -*- coding:utf-8 -*-
# Copyright xmuspeech (Author: Leo 2023-03)

import inspect
import logging
import os
from typing import List, Optional

from rich.logging import RichHandler

_seen_logs = set()

DATE_FORMAT = "%m/%d %H:%M:%S"


def get_logger(name: str = __name__) -> "Logger":
    """Get logger singleton instance based on package name."""
    name = name.split(".")[0]
    return Logger.get_instance(name=name)


class Logger:
    """Sigleton pattern for logger.

    Args:
        name (str): The name of the logger.
    """

    __loggers = {}
    # __format = "[ %(name)s - %(pathname)s:%(lineno)s - %(funcName)s - %(levelname)s ]\n#### %(message)s"

    __format = "%(asctime)s >> [ subtools - %(name)s - %(levelname)s ]: %(message)s"

    @staticmethod
    def get_instance(name: str):
        if name in Logger.__loggers:
            return Logger.__loggers[name]
        else:
            return Logger(name)

    def __init__(self, name):
        assert (
            name not in Logger.__loggers
        ), f"logger {name} is already in dict, use method: get_logger."
        self._name = name

        handler = RichHandler(
            show_time=False,
            show_level=False,
            show_path=False,
            markup=True,
            rich_tracebacks=True,
        )
        handler.setLevel(logging.INFO)
        formatter = logging.Formatter(Logger.__format, datefmt=DATE_FORMAT)
        handler.setFormatter(formatter)

        self._logger = logging.getLogger(name)
        self._logger.setLevel(logging.INFO)
        self._logger.addHandler(handler)
        self._logger.propagate = False
        self._force_verbose = None

        Logger.__loggers[name] = self

    @staticmethod
    def __get_call_info(stack_pos=2):
        """This function aims to get caller of logger through ``inspect.stack()``.

        [frameinfo[frame, filename, lineno, function, code_context, index], ... ]
        stack[0]: this function, stack[1]: Logger.info(), stack[2]: the caller of logger outside.
        """
        stack = inspect.stack()

        fn = stack[stack_pos][1]
        ln = stack[stack_pos][2]
        func = stack[stack_pos][3]

        return fn, ln, func

    @staticmethod
    def get_msg_prefix(fn, ln, func, *, verbose=False):
        if not verbose:
            try:
                fn = os.path.basename(fn)
            except (TypeError, ValueError, AttributeError):
                fn = fn
            return f'{fn}:{ln}'
        return f'{fn}:{ln} {func}'

    @staticmethod
    def _check_valid_logging_level(level: str):
        assert level in [
            "INFO",
            "DEBUG",
            "WARNING",
            "ERROR",
        ], "found invalid logging level"

    def set_level(self, level: str) -> None:
        """Set the logging level

        Args:
            level (str): Can only be INFO, DEBUG, WARNING and ERROR.
        """
        self._check_valid_logging_level(level)
        self._logger.setLevel(getattr(logging, level))

    def _log(self, level, message, ranks: List[int] = None):
        if ranks is None:
            getattr(self._logger, level)(message)
        else:
            rank = _infer_rank()
            if rank is not None:
                rank_flag = rank
            else:
                rank = 0
                rank_flag = None

            if isinstance(ranks, str):
                ranks = [int(rank) for rank in ranks.split(",")]
            ranks = ranks if isinstance(ranks, (list, tuple)) else [ranks]

            if rank in ranks:
                message = (
                    f"rank {rank_flag}: {message}" if rank_flag is not None else message
                )
                getattr(self._logger, level)(message)

    def info(
        self,
        message: str,
        ranks: List[int] = None,
        stack_pos: int = 2,
        verbose: Optional[bool] = None,
    ) -> None:
        """Log an info message.

        Args:
            message (str): The message to be logged.
            ranks (List[int]): List of parallel ranks.
        """
        if not bool(verbose := self._force_verbose):
            verbose = False if verbose is None else verbose

        message_prefix = self.get_msg_prefix(
            *self.__get_call_info(stack_pos=stack_pos), verbose=verbose
        )
        message = "{}\n#### {}".format(message_prefix, message)
        self._log("info", message, ranks)

    def info_once(self, message: str, ranks: List[int] = None) -> None:
        """
        Log a warning, but only once.

        Args:
             message: Message to display
             ranks (List[int]): List of parallel ranks.
        """
        global _seen_logs
        if message not in _seen_logs:
            _seen_logs.add(message)
            self.info(message, ranks, stack_pos=3)

    def warning(
        self,
        message: str,
        ranks: List[int] = None,
        stack_pos: int = 2,
        verbose: Optional[bool] = None,
    ) -> None:
        """Log a warning message.

        Args:
            message (str): The message to be logged.
            ranks (List[int]): List of parallel ranks.
        """
        if not bool(verbose := self._force_verbose):
            verbose = False if verbose is None else verbose

        message_prefix = self.get_msg_prefix(
            *self.__get_call_info(stack_pos=stack_pos), verbose=verbose
        )
        message = "{}\n#### {}".format(message_prefix, message)
        self._log("warning", message, ranks)

    def warning_once(self, message: str, ranks: List[int] = None) -> None:
        """
        Log a warning, but only once.

        Args:
            message (str): The message to be logged.
            ranks (List[int]): List of parallel ranks.
        """
        global _seen_logs
        if message not in _seen_logs:
            _seen_logs.add(message)
            self.warning(message, ranks=ranks, stack_pos=3)

    def debug(
        self,
        message: str,
        ranks: List[int] = None,
        stack_pos: int = 2,
        verbose: Optional[bool] = None,
    ) -> None:
        """Log a debug message.

        Args:
            message (str): The message to be logged.
            ranks (List[int]): List of parallel ranks.
        """
        if not bool(verbose := self._force_verbose):
            verbose = True if verbose is None else verbose

        message_prefix = self.get_msg_prefix(
            *self.__get_call_info(stack_pos=stack_pos), verbose=verbose
        )
        message = "{}\n#### {}".format(message_prefix, message)
        self._log("debug", message, ranks)

    def error(
        self,
        message: str,
        ranks: List[int] = None,
        stack_pos: int = 2,
        verbose: Optional[bool] = None,
    ) -> None:
        """Log an error message.

        Args:
            message (str): The message to be logged.
            ranks (List[int]): List of parallel ranks.
        """
        if not bool(verbose := self._force_verbose):
            verbose = True if verbose is None else verbose

        message_prefix = self.get_msg_prefix(
            *self.__get_call_info(stack_pos=stack_pos), verbose=verbose
        )
        message = "{}\n#### {}".format(message_prefix, message)
        self._log("error", message, ranks)

    def error_once(
        self,
        message: str,
        ranks: List[int] = None,
        verbose=True,
    ) -> None:
        """
        Log a error, but only once.

        Args:
            message (str): The message to be logged.
            ranks (List[int]): List of parallel ranks.
        """
        global _seen_logs
        if message not in _seen_logs:
            _seen_logs.add(message)
            self.error(
                message,
                ranks=ranks,
                stack_pos=3,
                verbose=verbose,
            )

    def force_verbose(self, verbose):
        self._force_verbose = verbose


def _infer_rank() -> Optional[int]:
    cand_rank_env = ("RANK", "LOCAL_RANK")
    for key in cand_rank_env:
        rank = os.environ.get(key)
        if rank is not None:
            return int(rank)
    return None
