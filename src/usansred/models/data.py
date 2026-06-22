from pydantic import Field
from pydantic.dataclasses import dataclass


@dataclass
class EventCounts:
    """Event counts associated with a single scan."""

    detector: int = Field(default=0)
    monitor: int = Field(default=0)
    transmission: int = Field(default=0)


@dataclass
class XYData:
    """Dataclass for X-Y data associated with a single scan"""

    x: list[float] = Field(default_factory=list)
    y: list[float] = Field(default_factory=list)
    e: list[float] = Field(default_factory=list)
    t: list[float] = Field(default_factory=list)

    def as_dict(self) -> dict:
        return {"X": self.x, "Y": self.y, "E": self.e, "T": self.t}


@dataclass
class IQData:
    """Dataclass for I-Q data associated with a single scan

    I vs. Q (intensity vs. momentum transfer) is determined from the X-Y data.
    """

    q: list[float] = Field(default_factory=list)
    i: list[float] = Field(default_factory=list)
    e: list[float] = Field(default_factory=list)
    t: list[float] = Field(default_factory=list)

    def as_dict(self) -> dict:
        return {"Q": self.q, "I": self.i, "E": self.e, "T": self.t}


@dataclass
class MonitorData:
    """Container for monitor data associated with a single scan."""

    xy_data: XYData = Field(default_factory=XYData)
    iq_data: IQData = Field(default_factory=IQData)
    filepath: str = ""
