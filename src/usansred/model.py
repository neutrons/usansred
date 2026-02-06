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


# TODO: finish implementing this (Glass)
class RunNumberValidator:
    def validate_run_numbers(run_numbers: str) -> list[str]:
        """Validate and parse run numbers from a string.

        Valid formats are:
        - Single run number: "45306"
        - Comma-separated run numbers: "45306,45307,45308"
        - Range of run numbers: "45306-45310"
        - Combination of the above: "45306,45308-45310,45312"

        Parameters
        ----------
        run_numbers
            Comma-separated string of run numbers

        Returns
        -------
            List of run numbers as strings
        """
        runs = set()
        parts = run_numbers.split(",")
        for part in parts:
            part = part.strip()
            if "-" in part:
                start, end = part.split("-")
                runs.update(str(r) for r in range(int(start), int(end) + 1))
            else:
                runs.add(part)
        return sorted(runs)
