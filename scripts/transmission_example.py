"""Compute the scan-by-scan transmission example of the user documentation.

Reads the event counts of sample ``S115_dry`` and of the empty cell ``EmptyPCell`` from the
reference dataset of the ``usansred-data`` repository, as listed in ``setup-empty-cell.json``, and
prints, for each pair of scans, the quantities of the table in section "Example" of
``docs/source/user/normalizations.rst``:

- the share of the transmitted neutrons recorded by the main detector, ``N_D / (N_D + N_Tr)``,
  for the empty cell and for the sample;
- the transmitted fraction of the empty cell, ``tau = (N_D + N_Tr) / N_M``;
- the transmission coefficient of the scan from both detectors, ``tau / tau_empty_cell``;
- the transmission coefficient of the scan from the transmission detector alone,
  ``(N_Tr / N_M) / (N_Tr / N_M)_empty_cell``.

Scan ``k`` of the sample is paired with scan ``k`` of the empty cell. The table is printed as an
RST ``list-table``, followed by the coefficient summed over all scans, as computed by ``usansred``.
The script stops with an error if any count of a scan is zero, e.g. because an event-count file is
missing, since the table would then show wrong values.

Usage::

    git submodule update --init
    pixi run python scripts/transmission_example.py [setup file]
"""

import logging
import sys
from pathlib import Path

from usansred.reduce import Experiment, Scan

REPOSITORY = Path(__file__).resolve().parent.parent
SETUP_FILE = REPOSITORY / "tests" / "usansred-data" / "IPTS-30410" / "shared" / "setup-empty-cell.json"
SAMPLE = "S115_dry"

HEADER = """.. list-table::
   :header-rows: 1
   :widths: 10 18 18 18 18 18

   * - Scan
     - Share in main detector, empty cell
     - Share in main detector, sample
     - :math:`\\tau_{empty \\, cell}`
     - :math:`T` from both detectors
     - :math:`T` from transmission detector only"""


def main_detector_share(scan: Scan) -> float:
    """Fraction of the transmitted neutrons of a scan that are recorded by the main detector."""
    counts = scan.counts
    return counts.detector / (counts.detector + counts.transmission)


def transmitted_fraction(scan: Scan) -> float:
    """Transmitted fraction of a scan, ``(N_D + N_Tr) / N_M``, from both detectors."""
    counts = scan.counts
    return (counts.detector + counts.transmission) / counts.monitor


def transmission_detector_fraction(scan: Scan) -> float:
    """Fraction of the monitor counts recorded by the transmission detector alone, ``N_Tr / N_M``."""
    counts = scan.counts
    return counts.transmission / counts.monitor


def check_counts(scans: list[Scan], name: str) -> None:
    """Raise ``ValueError`` if the monitor, main-detector, or transmission-detector count of a scan is zero.

    ``usansred`` sets the count of a missing event-count file to zero and logs a warning; a zero
    count would make the table show wrong values, or divide by zero.
    """
    for scan in scans:
        counts = scan.counts
        zero = [label for label in ("monitor", "detector", "transmission") if getattr(counts, label) == 0]
        if zero:
            raise ValueError(
                f"Scan {scan.number} of {name} has zero {', '.join(zero)} counts; check its event-count files."
            )


def main() -> None:
    """Print the scan-by-scan transmission table and the summed transmission coefficient."""
    setup_file = Path(sys.argv[1]) if len(sys.argv) > 1 else SETUP_FILE
    if not setup_file.exists():
        raise FileNotFoundError(f"{setup_file} not found; run `git submodule update --init` first.")
    logging.disable(logging.INFO)  # keep the warnings and errors, e.g. about missing event-count files
    experiment = Experiment(config_file=str(setup_file))
    empty_cell = experiment.empty_cell
    sample = next(s for s in experiment.samples if s.name == SAMPLE)
    check_counts(empty_cell.scans, empty_cell.name)
    check_counts(sample.scans, SAMPLE)
    if len(sample.scans) != len(empty_cell.scans):
        raise ValueError(f"{SAMPLE} has {len(sample.scans)} scans, but {empty_cell.name} has {len(empty_cell.scans)}")

    print(
        f"Sample {SAMPLE} (runs {sample.scans[0].number} to {sample.scans[-1].number}), "
        f"empty cell {empty_cell.name} (runs {empty_cell.scans[0].number} to {empty_cell.scans[-1].number})\n"
    )
    print(HEADER)
    for k, (scan_s, scan_ec) in enumerate(zip(sample.scans, empty_cell.scans), start=1):
        tau_ec = transmitted_fraction(scan_ec)
        t_both = transmitted_fraction(scan_s) / tau_ec
        t_trans = transmission_detector_fraction(scan_s) / transmission_detector_fraction(scan_ec)
        print(f"   * - {k}")
        print(f"     - {main_detector_share(scan_ec):.3f}")
        print(f"     - {main_detector_share(scan_s):.3f}")
        print(f"     - {tau_ec:.4f}")
        print(f"     - {t_both:.3f}")
        print(f"     - {t_trans:.3f}")
    print(f"\nTransmission coefficient for sample {SAMPLE}, summed over all scans: {sample.transmission:.4f}")


if __name__ == "__main__":
    main()
