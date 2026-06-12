from enum import StrEnum, auto


class MeasurementType(StrEnum):
    """Enum for measurement types."""

    SAMPLE = auto()
    BACKGROUND = auto()
    EMPTY_CELL = auto()
