# -*- coding:utf-8 -*-
# Copyright xmuspeech (Author: Leo 2023-05)

from typing import Dict, List, Tuple, Union

__all__ = ["concat_columns_id"]


def concat_columns_id(
    data,
    concat_col_id: Union[str, List[str], Tuple[str, ...]],
    join_str: str = "_",
    easy_check: bool = True,
):
    r"""
    concat columns as `id` column. If `easy_check=True`, suppose all inputs has same sketch.
    """
    if isinstance(concat_col_id, str):
        concat_col_id = (concat_col_id,)
    elif not isinstance(concat_col_id, (list, tuple)):
        raise ValueError(
            f"Input colname(s) should be list/tuple/str, but got type:{type(concat_col_id)}:{concat_col_id}"
        )
    concat_col_id = (col for col in concat_col_id if col in set(concat_col_id))
    for index, sample in enumerate(data):
        if index == 0:
            if not isinstance(sample, dict):
                raise ValueError(
                    f"Only support func:{concat_columns_id.__qualname__} concat columns id for dict, but got type:{type(sample)}: {sample}"
                )
            if easy_check:
                _check_columns_id(sample, concat_col_id)
        yield _concat_columns_id(sample, concat_col_id, join_str, check=not easy_check)


def _concat_columns_id(
    sample: Dict,
    concat_col_id: Union[List[str], Tuple[str, ...]],
    join_str: str = "_",
    check: bool = True,
):
    if check:
        _check_columns_id(sample, concat_col_id)
    concat_id = join_str.join([str(sample[col]) for col in concat_col_id])
    sample["id"] = concat_id
    return sample


def _check_columns_id(sample: Dict, concat_col_id: Union[List[str], Tuple[str, ...]]):
    if "id" in sample and "id" not in concat_col_id:
        raise ValueError(
            f"Error when concatting {concat_col_id} to {'id'}: try to generate new 'id' from columns:{concat_col_id}, "
            f"but column {'id'} is already in the sample."
        )
    if any(col not in sample for col in concat_col_id):
        raise ValueError(
            f"Error when concatting {concat_col_id} to {'id'}: "
            f"columns {set(concat_col_id) - set(sample)} are not in the sample."
        )
