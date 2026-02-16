"""Data model definitions for USANS reduction."""

from dataclasses import field

from pydantic.dataclasses import dataclass


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
