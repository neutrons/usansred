"""Data model definitions for USANS reduction."""

from dataclasses import field
from typing import Annotated

from pydantic import BaseModel, ConfigDict, Field, field_validator
from pydantic.dataclasses import dataclass

from usansred.utils import cast_to_bool


@dataclass
class XYData:
    x: list[float] = field(default_factory=list)
    y: list[float] = field(default_factory=list)
    e: list[float] = field(default_factory=list)
    t: list[float] = field(default_factory=list)

    def as_dict(self) -> dict:
        return {"X": self.x, "Y": self.y, "E": self.e, "T": self.t}


@dataclass
class IQData:
    q: list[float] = field(default_factory=list)
    i: list[float] = field(default_factory=list)
    e: list[float] = field(default_factory=list)
    t: list[float] = field(default_factory=list)

    def as_dict(self) -> dict:
        return {"Q": self.q, "I": self.i, "E": self.e, "T": self.t}


@dataclass
class MonitorData:
    xy_data: XYData = field(default_factory=XYData)
    iq_data: IQData = field(default_factory=IQData)
    filepath: str = ""


######################################
### Reduction input configurations ###
######################################


def _to_int(v: object) -> int:
    return int(v) if isinstance(v, str) else v


def _to_float(v: object) -> float:
    return float(v) if isinstance(v, str) else v


class _ScanBase(BaseModel):
    """Base class for scan configurations"""

    model_config = ConfigDict(extra="forbid")

    name: str
    start_scan_num: int
    num_of_scans: int
    exclude: list[int] = Field(default_factory=list)

    @field_validator("start_scan_num", "num_of_scans", mode="before")
    @classmethod
    def _coerce_scan_ints(cls, v):
        return _to_int(v)

    @field_validator("exclude", mode="before")
    @classmethod
    def _coerce_exclude(cls, v):
        return [_to_int(x) for x in v]

    @field_validator("exclude")
    @classmethod
    def _unique_exclude(cls, v):
        if len(v) != len(set(v)):
            raise ValueError("exclude must contain unique scan numbers")
        return v


class SampleConfig(_ScanBase):
    thickness: Annotated[float, Field(gt=0)]
    is_background: bool = False

    @field_validator("thickness", mode="before")
    @classmethod
    def _coerce_thickness(cls, v):
        return _to_float(v)


class BackgroundConfig(_ScanBase):
    thickness: Annotated[float, Field(gt=0)]
    is_background: bool = True  # fixed default, not settable by users

    @field_validator("thickness", mode="before")
    @classmethod
    def _coerce_thickness(cls, v):
        return _to_float(v)


class BinningConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    log_binning: bool = False
    steps_per_decade: int = 33
    q_min: Annotated[float, Field(gt=0)] = 1e-6

    @field_validator("log_binning", mode="before")
    @classmethod
    def _coerce_log_binning(cls, v):
        return cast_to_bool(v)


class ReductionInput(BaseModel):
    model_config = ConfigDict(extra="forbid")

    samples: Annotated[list[SampleConfig], Field(min_length=1)]
    background: BackgroundConfig | None = None
    save_all_harmonics: bool = False
    binning: BinningConfig = Field(default_factory=BinningConfig)
