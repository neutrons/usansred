"""Data model definitions for USANS reduction."""

from dataclasses import field
from typing import Annotated, ClassVar

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


def _to_int(v: object) -> object:
    return int(v) if isinstance(v, str) else v


def _to_float(v: object) -> object:
    return float(v) if isinstance(v, str) else v


class _ScanBase(BaseModel):
    """Base class for scan configurations"""

    model_config = ConfigDict(extra="forbid")

    name: Annotated[str, Field(min_length=1, description="Sample/background name.")]
    start_scan_num: Annotated[int, Field(description="Starting scan number (numeric string or integer is valid).")]
    num_of_scans: Annotated[
        int, Field(description="Number of consecutive scans in the sequence (numeric string or integer is valid).")
    ]
    exclude: list[int] = Field(default_factory=list, description="Optional list of scans to skip during reduction.")

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
    """Configuration for a single sample to reduce."""

    is_background: ClassVar[bool] = False
    thickness: Annotated[float, Field(gt=0, description="Sample thickness in cm (numeric string or float is valid).")]

    @field_validator("thickness", mode="before")
    @classmethod
    def _coerce_thickness(cls, v):
        return _to_float(v)


class BackgroundConfig(_ScanBase):
    """Optional background sample configuration."""

    is_background: ClassVar[bool] = True
    thickness: Annotated[float, Field(gt=0, description="Background sample thickness in cm.")]

    @field_validator("thickness", mode="before")
    @classmethod
    def _coerce_thickness(cls, v):
        return _to_float(v)


class BinningConfig(BaseModel):
    """Q binning settings applied during reduction."""

    model_config = ConfigDict(extra="forbid")

    log_binning: bool = Field(default=False, description="Enable logarithmic Q binning.")
    steps_per_decade: int = Field(
        default=33, ge=1, description="Number of Q bins per decade when log binning is enabled."
    )
    q_min: Annotated[float, Field(gt=0, description="Minimum Q value in 1/Å for log binning.")] = 1e-6

    @field_validator("log_binning", mode="before")
    @classmethod
    def _coerce_log_binning(cls, v):
        return cast_to_bool(v)


class ReductionConfig(BaseModel):
    """Top-level configuration for a USANS reduction run."""

    model_config = ConfigDict(extra="forbid")

    samples: Annotated[list[SampleConfig], Field(min_length=1, description="List of sample configurations to reduce.")]
    background: BackgroundConfig | None = Field(
        default=None, description="Background (empty cell) configuration. Omit to skip background subtraction."
    )
    save_all_harmonics: bool = Field(
        default=False, description="Save individual harmonic output files in addition to the combined result."
    )
    binning: BinningConfig = Field(default_factory=BinningConfig, description="Q binning configuration.")
