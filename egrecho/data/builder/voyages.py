from collections.abc import Mapping
from dataclasses import dataclass, field, fields
from typing import Sequence, Type, TypeVar, Union

from egrecho.data.dew import Dew
from egrecho.utils.common import asdict_filt
from egrecho.utils.misc import rich_exception_info

VT = TypeVar("VT", bound="VoyageTemplate")
EXTRA_COLUMN_NAME = "others"


def _sketch_to_tuple(sketch):
    assert isinstance(sketch, Sequence)
    if isinstance(sketch, str):
        return tuple(sketch.strip().split())
    return tuple(list(sketch))


@dataclass
class VoyageTemplate:
    task: str
    start_sketch: Sequence[str] = field(default_factory=tuple)
    end_sketch: Sequence[str] = field(default_factory=tuple)

    def __post_init__(self):
        self.start_sketch, self.end_sketch = _sketch_to_tuple(
            self.start_sketch
        ), _sketch_to_tuple(self.end_sketch)

        for sketch in self.start_sketch, self.end_sketch:
            if len(set(sketch)) != len(sketch):
                raise ValueError(f"There are repeat field name in sketch:({sketch}).")
            if EXTRA_COLUMN_NAME in sketch:
                raise ValueError(
                    f"name:({EXTRA_COLUMN_NAME}) is kept for extra columns for receive unknown input fields, "
                    f"but it apears in sketch({sketch}), change it to another name."
                )

    @classmethod
    def from_dict(cls: Type[VT], template_dict: dict) -> VT:
        field_names = {f.name for f in fields(cls)}
        return cls(**{k: v for k, v in template_dict.items() if k in field_names})

    def to_dict(self):
        return asdict_filt(self, filt_type="none")


@rich_exception_info
def build_table(
    entry: Union[Sequence, Mapping, Dew],
    sketch: Sequence[str],
    drop_extras: bool = True,
) -> dict:
    r"""
    Builds a dict table from entry.

    NOTE: If sketch column is more than entry, output contains the keys in entry,
    When sketch column is less than entry, it depends on `drop_extras`, if `drop_extras=False`,
    the additional attrs will gather to the `EXTRA_COLUMN_NAME` ("others") column  of output.

    Args:
        entry:
            inputs, can be tuple or dict.
        sketch:
            sketch column of target table. e.g., (audio, len, label)
        drop_extras:
            if True, the extra column in entry (not included in sketch) will be dropped,
            otherwise will gather to `EXTRA_COLUMN_NAME` ("others") column.

    Example:
        # tuple entry
        >>> entry = ('tom', 'speaker1')
        >>> sketch = ("id", "label")
        >>> build_table(entry, sketch)
        {'id': 'tom', 'label': 'speaker1'}
        >>> sketch = ("id") # sketch is less.
        >>> build_table(entry, sketch, drop_extras=True)
        {'id': 'tom'}
        >>> build_table(entry, sketch, drop_extras=False)
        {'id': 'tom', 'others': ('speaker1',)}
        >>> sketch = ("id", "label", "question") # sketch is more.
        {'id': 'tom', 'label': 'speaker1'}
        # dict entry
        >>> entry = {"id":'tom',"label":'speaker1'}
        >>> sketch = ("id", "label", "question")
        >>> build_table(entry, sketch, drop_extras=False)
        {'id': 'tom', 'others': {'label': 'speaker1'}}
    """
    if isinstance(sketch, str):
        sketch = (sketch,)
    assert len(set(sketch)) == len(
        sketch
    ), f"Target columns:({sketch}) has dumplicate name."

    if isinstance(entry, (Dew, Mapping)):
        if isinstance(entry, Dew):
            entry = entry.to_dict()
        valid_keys = entry.keys() & set(sketch)
        table = {key: entry[key] for key in valid_keys}

        if not drop_extras and len(entry) > len(table):
            extra_keys = entry.keys() - table.keys()
            extra = {EXTRA_COLUMN_NAME: {key: entry[key] for key in extra_keys}}
            table.update(extra)

        return table
    elif isinstance(entry, Sequence):
        valid_field_len = min(len(entry), len(sketch))
        table = {sketch[i]: value for i, value in enumerate(entry[:valid_field_len])}
        if not drop_extras and len(entry) > valid_field_len:
            table[EXTRA_COLUMN_NAME] = entry[valid_field_len:]
        return table

    else:
        raise TypeError(
            f"inputs to be build should be dict or sequence, Got {type(entry)}"
        )
