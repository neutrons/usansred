from enum import StrEnum


class MeasurementType(StrEnum):
    """Enum for measurement types."""

    SAMPLE = "sample"
    BACKGROUND = "background"
    EMPTY_CELL = "empty_cell"
