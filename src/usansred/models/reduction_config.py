"""Data model definitions for USANS reduction."""

from typing import Annotated, ClassVar

from pydantic import BaseModel, ConfigDict, Field, field_validator

from usansred.enums import MeasurementType
from usansred.utils import cast_to_bool

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
        int,
        Field(ge=1, description="Number of consecutive scans in the sequence (numeric string or integer is valid)."),
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

    measurement_type: ClassVar[MeasurementType] = MeasurementType.SAMPLE
    thickness: Annotated[float, Field(gt=0, description="Sample thickness in cm (numeric string or float is valid).")]

    @field_validator("thickness", mode="before")
    @classmethod
    def _coerce_thickness(cls, v):
        return _to_float(v)


class BackgroundConfig(_ScanBase):
    """Background sample configuration.

    A background is a measurement without the sample container,
    used to correct for instrument background and ambient scattering.
    """

    measurement_type: ClassVar[MeasurementType] = MeasurementType.BACKGROUND
    thickness: Annotated[float, Field(gt=0, description="Background sample thickness in cm.")]

    @field_validator("thickness", mode="before")
    @classmethod
    def _coerce_thickness(cls, v):
        return _to_float(v)


class EmptyCellConfig(_ScanBase):
    """Configuration for an empty cell scan.

    An empty cell is a measurement with a sample container present, but without any sample material.
    Used to help in the calculation of transmission coefficients and to correct for container scattering.
    """

    measurement_type: ClassVar[MeasurementType] = MeasurementType.EMPTY_CELL
    thickness: ClassVar[float] = 1.0  # Empty cell thickness is fixed at 1 cm for correction purposes


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
        default=None, description="Background sample configuration. Omit to skip background subtraction."
    )
    empty_cell: EmptyCellConfig | None = Field(
        default=None,
        description="Empty cell configuration. "
        "If omitted, transmission correction and empty-cell subtraction will be skipped.",
    )
    save_all_harmonics: bool = Field(
        default=False, description="Save individual harmonic output files in addition to the combined result."
    )
    binning: BinningConfig = Field(default_factory=BinningConfig, description="Q binning configuration.")
