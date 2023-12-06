# -*- coding:utf-8 -*-
# Copyright xmuspeech (Author: Leo 2023-05)

from typing import Dict

__all__ = ["rename_column", "rename_columns"]


def rename_column(data, orig_col: str, new_col: str, easy_check: bool = True):
    r"""
    rename sample (dict) column name. If `easy_check=True`, suppose all inputs has same sketch.
    """
    for index, sample in enumerate(data):
        if index == 0:
            if not isinstance(sample, dict):
                raise ValueError(
                    f"Only support func:{rename_column.__qualname__} rename column for dict, "
                    f"but got type:{type(sample)}: {sample}"
                )
            if easy_check:
                _check_column(sample, orig_col, new_col)
        yield _rename_column(sample, orig_col, new_col, check=not easy_check)


def rename_columns(data, col_mapping: Dict[str, str], easy_check: bool = True):
    r"""
    rename sample (dict) column names. If `easy_check=True`, suppose all inputs has same sketch.
    """
    for index, sample in enumerate(data):
        if index == 0:
            if not isinstance(sample, dict):
                raise ValueError(
                    f"Only support func:{rename_columns.__qualname__} rename columns for dict, "
                    f"but got type:{type(sample)}: {sample}"
                )
            if easy_check:
                _check_column(sample, col_mapping)
        yield _rename_columns(sample, col_mapping, check=not easy_check)


def _rename_column(sample: Dict, orig_col: str, new_col: str, check: bool = True):
    if check:
        _check_column(sample, orig_col, new_col)
    sample[new_col] = sample[orig_col]
    del sample[orig_col]
    return sample


def _rename_columns(sample: Dict, col_mapping: Dict[str, str], check: bool = True):
    if check:
        _check_columns(sample, col_mapping)
    sample = {new_col: sample[orig_col] for orig_col, new_col in col_mapping.items()}
    for orig_col in col_mapping:
        del sample[orig_col]
    return sample


def _check_columns(sample: Dict, col_mapping: Dict[str, str]):
    if any(col not in sample for col in col_mapping):
        raise ValueError(
            f"Error when renaming {list(col_mapping)} to {list(col_mapping.values())}: "
            f"columns {set(col_mapping) - set(sample)} are not in the sample."
        )
    if any(col in sample for col in col_mapping.values()):
        raise ValueError(
            f"Error when renaming {list(col_mapping)} to {list(col_mapping.values())}: "
            f"columns {set(sample) - set(col_mapping.values())} are already in the sample."
        )


def _check_column(sample: Dict, orig_col: str, new_col: str):
    if orig_col not in sample:
        raise ValueError(
            f"Error when renaming {orig_col} to {new_col}: column {orig_col} is not in the sample."
        )
    if new_col in sample:
        raise ValueError(
            f"Error when renaming {orig_col} to {new_col}: column {new_col} is already in the sample."
        )
