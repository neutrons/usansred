"""Generate the stitching example figures of the user documentation.

Reduces the reference dataset of the ``usansred-data`` repository with ``save_all_harmonics``
enabled, and plots, for samples ``S115_pc3`` and ``S115_dry``, the scaled profile of each
harmonic against the stitched profile. The figures are written to ``docs/source/user/media/``
and shown in ``docs/source/user/stitching.rst``.

Usage::

    git submodule update --init
    pixi run python scripts/plot_stitching_examples.py
"""

import csv
import logging
import re
import tempfile
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

from usansred.reduce import Experiment  # noqa: E402

REPOSITORY = Path(__file__).resolve().parent.parent
SETUP_FILE = REPOSITORY / "tests" / "usansred-data" / "IPTS-30410" / "shared" / "setup.json"
MEDIA_DIR = REPOSITORY / "docs" / "source" / "user" / "media"
SAMPLES = ["S115_pc3", "S115_dry"]
NUM_HARMONICS = 4

# Colors: four distinct hues for the harmonics, the combination of the reference categorical
# palette that best separates every pair, with and without simulated color-vision deficiency
# (worst pair: deltaE 16.2 under simulation, 19.6 in normal vision, on a white page). The
# stitched profile takes dark ink, which differs from all four hues in lightness.
HARMONIC_COLORS = ["#6250d6", "#008300", "#e87ba4", "#eda100"]  # violet, green, magenta, yellow
STITCHED_COLOR = "#0b0b0b"
TEXT_SECONDARY = "#52514e"
GRID_COLOR = "#e4e3df"


def read_profile(path: Path) -> tuple[list[float], list[float]]:
    """Read the Q and I columns of a reduced data file, keeping the points plottable on log axes.

    Parameters
    ----------
    path : Path
        Reduced data file with Q in ``1/angstrom``, intensity, and uncertainty columns.

    Returns
    -------
    tuple[list[float], list[float]]
        Q and intensity of the points with positive Q and positive intensity.
    """
    q, i = [], []
    with open(path, newline="") as file:
        for row in csv.reader(file):
            q_value, i_value = float(row[0]), float(row[1])
            if q_value > 0 and i_value > 0:
                q.append(q_value)
                i.append(i_value)
    return q, i


def stitching_outcome(log_file: Path, name: str) -> tuple[dict[int, float], int | None]:
    """Extract the scale factors and the harmonic that could not be joined from a reduction log.

    Only the stitch of the scaled harmonics is considered: messages about the
    background-subtracted stitch name the measurement ``background-subtracted sample <name>``
    and do not match.

    Parameters
    ----------
    log_file : Path
        The ``reduction_<name>.log`` file of the measurement.
    name : str
        Name of the measurement.

    Returns
    -------
    tuple[dict[int, float], int | None]
        Scale factor of each joined harmonic, and the harmonic whose join failed and stopped the
        stitching (None when all harmonics were joined).

    Raises
    ------
    RuntimeError
        If the log does not report the stitched orders, e.g. because stitching raised an
        unexpected exception or harmonic 1 has no valid points.
    """
    text = log_file.read_text()
    label = re.escape(f"sample {name}")
    match = re.search(rf"Stitched orders \[([\d, ]+)\] of {label} with scale factors \[([^\]]+)\]", text)
    if match is None:
        raise RuntimeError(f"{log_file} does not report the stitched orders of sample {name}.")
    orders = [int(order) for order in match.group(1).split(",")]
    factors = [float(factor) for factor in match.group(2).split(",")]
    # Every failed join is reported with this prefix, whatever the reason; only the order is needed
    stop = re.search(rf"Cannot join order (\d+) onto the stitched profile of {label}: ", text)
    return dict(zip(orders, factors)), (int(stop.group(1)) if stop else None)


def plot_sample(output_dir: Path, name: str) -> Path:
    """Plot the harmonics of a sample and its stitched profile in a single panel.

    Each harmonic is drawn as a line, at its own intensity scale before stitching, and the
    stitched profile as markers.

    Parameters
    ----------
    output_dir : Path
        Directory holding the reduced files and logs.
    name : str
        Name of the sample.

    Returns
    -------
    Path
        The written figure.
    """
    scale_factors, failed_harmonic = stitching_outcome(output_dir / f"reduction_{name}.log", name)
    q_stitched, i_stitched = read_profile(output_dir / f"UN_{name}_det_stitched.txt")

    figure, axis = plt.subplots(figsize=(7.0, 4.8), layout="constrained")
    for harmonic, color in zip(range(1, NUM_HARMONICS + 1), HARMONIC_COLORS):
        q, i = read_profile(output_dir / f"UN_{name}_det_{harmonic}.txt")
        if harmonic in scale_factors:
            status = "reference" if harmonic == 1 else f"joined, s = {scale_factors[harmonic]:.3g}"
        elif harmonic == failed_harmonic:
            status = "not joined"
        else:
            status = "not attempted"
        axis.plot(q, i, linewidth=2.0, color=color, label=f"harmonic {harmonic} ({status})")
    axis.plot(
        q_stitched,
        i_stitched,
        linestyle="none",
        marker="o",
        markersize=4.0,
        markerfacecolor=STITCHED_COLOR,
        markeredgecolor="white",
        markeredgewidth=0.6,
        label="stitched profile",
    )

    axis.set_xscale("log")
    axis.set_yscale("log")
    axis.set_xlabel("Q (1/Å)", color=TEXT_SECONDARY)
    axis.set_ylabel("I (counts per monitor count, per cm per sr)", color=TEXT_SECONDARY, fontsize=9)
    axis.grid(True, which="major", color=GRID_COLOR, linewidth=0.6)
    axis.tick_params(colors=TEXT_SECONDARY, labelsize=8)
    for spine in axis.spines.values():
        spine.set_color(GRID_COLOR)
    # The sample is named in the figure caption of the documentation, so no figure title
    axis.legend(loc="lower left", frameon=False, fontsize=8.5)

    MEDIA_DIR.mkdir(parents=True, exist_ok=True)
    figure_path = MEDIA_DIR / f"stitching_{name}.png"
    figure.savefig(figure_path, dpi=130, facecolor="white")
    plt.close(figure)
    return figure_path


def main() -> None:
    """Reduce the reference dataset with ``save_all_harmonics`` and write the example figures."""
    if not SETUP_FILE.exists():
        raise FileNotFoundError(f"{SETUP_FILE} not found; run `git submodule update --init` first.")
    with tempfile.TemporaryDirectory() as output_dir:
        experiment = Experiment(config_file=str(SETUP_FILE))
        experiment.config.save_all_harmonics = True
        experiment.reduce(output_dir=output_dir)
        for name in SAMPLES:
            logging.info("Wrote %s", plot_sample(Path(output_dir), name))


if __name__ == "__main__":
    main()
