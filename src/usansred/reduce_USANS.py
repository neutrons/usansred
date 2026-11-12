import argparse
import html
import json
import logging
import math
import os
import re
import sys
import time
import traceback
import warnings
from dataclasses import dataclass, field

import h5py
import numpy as np
import requests
from mantid.simpleapi import (
    ConvertTableToMatrixWorkspace,
    CropWorkspace,
    LoadEventNexus,
    Rebin,
    StepScan,
    mtd,
)
from matplotlib import use
from plot_publisher import plot1d, publish_plot
from plotly.offline import get_plotlyjs_version

from usansred import reduce
from usansred.io.filenames import reduced_data_filename
from usansred.io.read import read_config
from usansred.io.save import save_ascii, save_summed_spectra
from usansred.models.reduction_config import ReductionConfig, SampleConfig
from usansred.summary import generate_report
from usansred.utils.logging import get_logger

logger = get_logger(__name__)

use("agg")
np.seterr(all="ignore")
warnings.filterwarnings("ignore", module="numpy")


def write_file(path: str, text: str) -> None:
    """Write a text file, to a temporary file in the same directory renamed at the end.

    A reader of the file never sees it partly written, and a job stopped while writing leaves the
    earlier content in place.

    Parameters
    ----------
    path : str
        Path of the file.
    text : str
        Content of the file.
    """
    # The process ID keeps the temporary files of two overlapping calls apart
    temporary_file = f"{path}.{os.getpid()}.tmp"
    try:
        with open(temporary_file, "x") as fd:
            fd.write(text)
        os.replace(temporary_file, path)
    finally:
        if os.path.exists(temporary_file):
            os.remove(temporary_file)


def update_sequence_info(out_file: str, info: dict) -> None:
    """Add entries to a file of a sequence, ``scan_<N>.json`` or ``sample_<N>.json``, replacing those of the same key.

    The file is created if it does not exist, and written with ``write_file``.

    Parameters
    ----------
    out_file : str
        Path of the file.
    info : dict
        The entries.

    Raises
    ------
    ValueError
        If the file exists but does not hold a JSON object. It is left unchanged: starting a new file
        would lose the runs recorded in it, which step 2 would then exclude.
    """
    content = {}
    if os.path.isfile(out_file):
        try:
            with open(out_file) as fd:
                content = json.load(fd)
        except ValueError as error:
            raise ValueError(f"{out_file} cannot be read, and is left unchanged: correct it") from error
        if not isinstance(content, dict):
            raise ValueError(f"{out_file} does not hold a JSON object, and is left unchanged: correct it")
    content.update(info)
    write_file(out_file, json.dumps(content))


def get_sequence_info(seq_file):
    """Get the sequence information from local autoreduce folder json files for autoreduction"""
    if os.path.exists(seq_file):
        with open(seq_file) as json_file:
            seq_dict = json.load(json_file)
    else:
        return None

    return seq_dict


# Name of the setup file written by step 2, in the output directory
SETUP_FILENAME = "autoconfig.json"

# Name of the directory, in the output directory, where step 2 writes the reduced data
REDUCTION_DIRNAME = "auto"

# Files in which step 1 records the runs of a sequence, named after the first run of the sequence
SCAN_FILE_PATTERN = re.compile(r"^scan_(?P<first_run>.+)\.json$")

# Log holding the run number of the first run of the sequence that a run belongs to
FIRST_RUN_LOG = "BL1A:CS:Scan:USANS:FirstRun"

# Log holding the thickness of the sample, in cm
THICKNESS_LOG = "BL1A:CS:Scan:USANS:SampleThickness"

# Thickness of the sample, in cm, when the log THICKNESS_LOG gives none
DEFAULT_THICKNESS = 0.1


def read_first_run(filename: str) -> int | float | str | None:
    """Read the first run of the sequence that a run belongs to, from its event file.

    The log ``BL1A:CS:Scan:USANS:FirstRun`` is read with h5py, without loading the events, so that
    the runs outside a sequence can be skipped quickly. The value is returned as stored, so that a value
    that is not a run number can be reported as it is; ``as_run_number`` converts it.

    Parameters
    ----------
    filename : str
        Path of the event file of the run.

    Returns
    -------
    int | float | str | None
        The last value of the log, as stored, text decoded: the run number of the first run of the sequence,
        or a value that is not a positive integer, such as -1, for a run that is not part of any sequence.
        ``None`` if the file has no such log, or the log no value.

    Raises
    ------
    OSError
        If the file cannot be opened as an HDF5 file.
    """
    with h5py.File(filename, "r") as nexus:
        path = f"/entry/DASlogs/{FIRST_RUN_LOG}/value"
        if path not in nexus:
            return None
        values = np.atleast_1d(nexus[path][()]).ravel()
    if values.size == 0:
        return None
    value = values[-1]
    if isinstance(value, bytes):
        return value.decode(errors="replace")
    # A NumPy scalar becomes the Python int, float or str it holds
    return value.item() if isinstance(value, np.generic) else value


def as_run_number(value) -> int | None:
    """Convert a value of the log ``BL1A:CS:Scan:USANS:FirstRun`` to a run number.

    Parameters
    ----------
    value
        The value, as returned by ``read_first_run``.

    Returns
    -------
    int | None
        The value as an integer if it is a finite whole number, such as ``41713``, ``41713.0`` or ``"41713"``;
        ``None`` otherwise, such as for ``41713.5``, NaN or ``"Not Set"``. The sign is not checked.
    """
    if isinstance(value, bool):
        return None
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    if not (math.isfinite(number) and number.is_integer()):
        return None
    return int(number)


def sample_thickness(run) -> float:
    """Read the thickness of the sample, in cm, from the log ``BL1A:CS:Scan:USANS:SampleThickness``.

    Parameters
    ----------
    run : mantid.api.Run
        Run of the workspace of the events.

    Returns
    -------
    float
        The last value of the log, in cm. 0.1 cm if the log is missing, is not a number, or is not
        above 0, which is logged.
    """
    if not run.hasProperty(THICKNESS_LOG):
        logger.info(f"No log {THICKNESS_LOG}: the thickness is {DEFAULT_THICKNESS} cm")
        return DEFAULT_THICKNESS
    value = run.getProperty(THICKNESS_LOG).value
    # A time-series log holds a sequence of values, of which the last one is taken
    if not isinstance(value, (str, bytes)) and hasattr(value, "__len__"):
        value = value[-1] if len(value) > 0 else None
    try:
        thickness = float(value)
    except (TypeError, ValueError):
        logger.info(f"The log {THICKNESS_LOG} is not a number ({value!r}): the thickness is {DEFAULT_THICKNESS} cm")
        return DEFAULT_THICKNESS
    if not (math.isfinite(thickness) and thickness > 0):
        logger.info(f"The log {THICKNESS_LOG} is not above 0 ({thickness}): the thickness is {DEFAULT_THICKNESS} cm")
        return DEFAULT_THICKNESS
    return thickness


def record_run(outdir: str, first_run: int, run_number: str, iq_file: str, run_title: str, thickness: float) -> None:
    """Record a run in the files of its sequence, ``scan_<N>.json`` and ``sample_<N>.json``.

    ``scan_<N>.json`` gets an entry for the run, with its I(Q) file and its title. ``sample_<N>.json``
    gets the title, the thickness and the background flag 0, replacing the values written before.

    Parameters
    ----------
    outdir : str
        Output directory of the preprocessing script.
    first_run : int
        Run number of the first run of the sequence, the ``N`` of the file names.
    run_number : str
        Run number of the run.
    iq_file : str
        Path of the file of the scan of the main harmonic.
    run_title : str
        Title of the run.
    thickness : float
        Thickness of the sample, in cm.
    """
    update_sequence_info(
        os.path.join(outdir, f"scan_{first_run}.json"),
        {str(run_number): {"iq": iq_file, "title": run_title}},
    )
    update_sequence_info(
        os.path.join(outdir, f"sample_{first_run}.json"),
        {"title": run_title, "background": 0, "thickness": thickness},
    )


@dataclass
class RunSequence:
    """The information recorded by step 1 for one sequence of runs.

    Parameters
    ----------
    first_run : int
        Run number of the first run of the sequence, the ``N`` of ``scan_<N>.json``.
    runs : dict[str, dict]
        Content of ``scan_<N>.json``: one entry per run, keyed by the run number.
    sample : dict
        Content of ``sample_<N>.json``, or an empty dictionary if the file is missing.
    """

    first_run: int
    runs: dict[str, dict] = field(default_factory=dict)
    sample: dict = field(default_factory=dict)


def collect_sequence(outdir: str, first_run: int) -> RunSequence | None:
    """Collect one sequence of runs recorded by step 1, from ``scan_<N>.json`` and ``sample_<N>.json``.

    If ``sample_<N>.json`` is missing, the sample information is empty, and a warning is logged.
    The runs of ``scan_<N>.json`` numbered below ``N`` cannot be part of the sequence: they are left out,
    with a warning.

    Parameters
    ----------
    outdir : str
        Output directory of the preprocessing script.
    first_run : int
        Run number of the first run of the sequence, the ``N`` of the file names.

    Returns
    -------
    RunSequence | None
        The sequence, or ``None`` if ``scan_<N>.json`` is missing or holds no run of the sequence.
    """
    scan_filename = f"scan_{first_run}.json"
    runs = get_sequence_info(os.path.join(outdir, scan_filename))
    if runs is None:
        return None
    earlier = sorted(int(run) for run in runs if int(run) < first_run)
    if earlier:
        logger.warning(f"Runs {earlier} of {scan_filename} come before the first run, {first_run}, and are left out")
        runs = {run: entry for run, entry in runs.items() if int(run) >= first_run}
    if not runs:
        logger.warning(f"{scan_filename} holds no run of the sequence")
        return None

    sample_filename = f"sample_{first_run}.json"
    sample = get_sequence_info(os.path.join(outdir, sample_filename))
    if sample is None:
        logger.warning(f"{sample_filename} not found: the sample of {scan_filename} takes the default values")
        sample = {}
    return RunSequence(first_run=first_run, runs=runs, sample=sample)


def collect_sequences(outdir: str) -> list[RunSequence]:
    """Collect the sequences of runs recorded by step 1 in an output directory.

    There is one sequence for each file ``scan_<N>.json``, where ``N`` is an integer, ordered by ``N``,
    collected by ``collect_sequence``.
    A file whose ``N`` is not a positive integer, such as ``scan_-1.json``, is ignored: it holds the runs
    whose log ``BL1A:CS:Scan:USANS:FirstRun`` is not a run number, which are not part of any sequence.
    A file whose ``N`` is not an integer is ignored, with a warning.

    Parameters
    ----------
    outdir : str
        Output directory of the preprocessing script.

    Returns
    -------
    list[RunSequence]
        The sequences, ordered by the run number of their first run.
    """
    first_runs = []
    for filename in os.listdir(outdir):
        match = SCAN_FILE_PATTERN.match(filename)
        if match is None:
            continue
        try:
            first_run = int(match.group("first_run"))
        except ValueError:
            logger.warning(f"Ignoring {filename}: {match.group('first_run')} is not a run number")
            continue
        if first_run <= 0:
            runs = get_sequence_info(os.path.join(outdir, filename))
            logger.info(
                f"Ignoring {filename}: its {len(runs)} runs are not part of any sequence (FirstRun = {first_run})"
            )
            continue
        first_runs.append(first_run)
    # Order by run number, not by file name, which would put scan_10001.json before scan_9990.json
    first_runs.sort()

    sequences = []
    for first_run in first_runs:
        sequence = collect_sequence(outdir, first_run)
        if sequence is not None:
            sequences.append(sequence)
    return sequences


# Prefix of the titles of the runs recorded during a scan of the analyzer, removed from the sample names
ALIGN_PREFIX = "Align:"

# Characters of a title replaced with "_" in the sample name: path separators, and ":"
REPLACED_CHARACTERS = ("/", "\\", ":")


def sequence_title(sequence: RunSequence) -> str | None:
    """Find the title of a sequence: the title of its first run, the run with the lowest run number.

    If that run has no title (``scan_<N>.json`` written before the titles were recorded), the title of
    ``sample_<N>.json`` is used, and this is logged. The runs whose title differs from the title of the
    first run are listed in a warning; the runs without a title are left out of that comparison, which is
    logged.

    Parameters
    ----------
    sequence : RunSequence
        The sequence, as returned by ``collect_sequences``.

    Returns
    -------
    str | None
        The title, or ``None`` if neither the first run nor ``sample_<N>.json`` has one.
    """
    runs = sorted((int(run), entry) for run, entry in sequence.runs.items())
    first_run, first_entry = runs[0]
    title = first_entry.get("title")
    if title is None:
        title = sequence.sample.get("title")
        # Without a title in sample_<N>.json either, sequence_name warns that the sequence has no title
        if title is not None:
            logger.info(
                f"Run {first_run} has no title in scan_{sequence.first_run}.json: the sequence takes the title "
                f"of sample_{sequence.first_run}.json, {title!r}"
            )
        return title

    untitled = [run for run, entry in runs if "title" not in entry]
    if untitled:
        logger.info(f"Runs {untitled} of sequence {sequence.first_run} have no title, and are not compared")
    differing = [run for run, entry in runs if "title" in entry and entry["title"] != title]
    if differing:
        logger.warning(
            f"Runs {differing} of sequence {sequence.first_run} have a title different from {title!r}, "
            f"the title of run {first_run}"
        )
    return title


def clean_title(title: str) -> str:
    """Turn a title into a sample name.

    White space at the start and end, and a leading ``Align:``, are removed; ``/``, ``\\`` and ``:`` are
    replaced with ``_``. The removal of ``Align:`` and the replacements are logged.

    Parameters
    ----------
    title : str
        Title of a sequence, as returned by ``sequence_title``.

    Returns
    -------
    str
        The name, which may be empty.
    """
    name = title.strip()
    if name.startswith(ALIGN_PREFIX):
        name = name.removeprefix(ALIGN_PREFIX).strip()
        logger.info(f"{ALIGN_PREFIX} removed from the title {title!r}")
    if any(character in name for character in REPLACED_CHARACTERS):
        replaced = name
        for character in REPLACED_CHARACTERS:
            name = name.replace(character, "_")
        logger.info(f"'/', '\\' and ':' replaced with '_' in {replaced!r}: {name!r}")
    return name


def sequence_name(sequence: RunSequence, index: int, taken: set[str]) -> str:
    """Name the sample of a sequence after its title.

    The name is the title cleaned by ``clean_title``. If there is no title, or the name is empty,
    the name is ``Sample_<index>``, with a warning. If the name is in ``taken``, the first of
    ``<name>_2``, ``<name>_3``, ... that is not is used instead, with a warning, so that the files of
    one sample do not overwrite those of another.

    Parameters
    ----------
    sequence : RunSequence
        The sequence, as returned by ``collect_sequences``.
    index : int
        Position of the sequence in the setup, from 0.
    taken : set[str]
        Names of the other entries of the setup.

    Returns
    -------
    str
        The name, which is not in ``taken``.
    """
    title = sequence_title(sequence)
    name = clean_title(title) if title is not None else ""
    if not name:
        name = f"Sample_{index}"
        logger.warning(f"Sequence {sequence.first_run} has no title: its sample is named {name}")

    if name in taken:
        counter = 2
        while f"{name}_{counter}" in taken:
            counter += 1
        logger.warning(
            f"The name {name} is already taken by another entry: "
            f"the sample of sequence {sequence.first_run} is named {name}_{counter}"
        )
        name = f"{name}_{counter}"
    return name


def sequence_entry(sequence: RunSequence, name: str) -> SampleConfig:
    """Build the entry of a sequence in a setup file.

    The runs of the entry span the run numbers from ``N``, the first run of the sequence, to the highest
    run number of the sequence; the run numbers of that span missing from the sequence, ``N`` included,
    are excluded. ``Sample`` reads ``num_of_scans`` consecutive runs from ``start_scan_num`` and skips the
    excluded ones, so ``num_of_scans`` is the length of the whole span. Starting the span at ``N`` lets
    step 2 find the entry of a sequence by its ``start_scan_num``.

    Parameters
    ----------
    sequence : RunSequence
        The sequence, as returned by ``collect_sequence``.
    name : str
        Name of the sample.

    Returns
    -------
    SampleConfig
        The entry, with the thickness of ``sample_<N>.json``, in cm, or 0.1 cm without one.
    """
    sample_info = sequence.sample
    # Thickness in cm
    thickness = float(sample_info["thickness"]) if "thickness" in sample_info else DEFAULT_THICKNESS
    run_numbers = {int(run) for run in sequence.runs}
    first, last = sequence.first_run, max(run_numbers)
    return SampleConfig(
        name=name,
        start_scan_num=first,
        num_of_scans=last - first + 1,
        thickness=thickness,
        exclude=sorted(set(range(first, last + 1)) - run_numbers),
    )


def build_setup(sequences: list[RunSequence]) -> ReductionConfig:
    """Build the setup of the reduction of the sequences.

    Every sequence is a sample, built by ``sequence_entry``: the setup has no background and no empty
    cell. Each sample is named after the title of its sequence (see ``sequence_name``).

    Parameters
    ----------
    sequences : list[RunSequence]
        The sequences, as returned by ``collect_sequences``.

    Returns
    -------
    ReductionConfig
        The setup, with one sample per sequence, and only the files of harmonic 1 saved.
    """
    samples = []
    taken = set()
    for index, sequence in enumerate(sequences):
        name = sequence_name(sequence, index, taken)
        taken.add(name)
        samples.append(sequence_entry(sequence, name))
    return ReductionConfig(samples=samples, save_all_harmonics=False)


def update_setup(config: ReductionConfig, sequence: RunSequence) -> tuple[ReductionConfig, SampleConfig]:
    """Rewrite or insert the entry of a sequence in the setup of every sequence.

    The entry of the sequence is the sample whose ``start_scan_num`` is ``N``, the first run of the
    sequence. It is rewritten by ``sequence_entry``, and keeps its name. Without such a sample, the entry
    is inserted in order of first run, and named by ``sequence_name`` against the names of the other
    entries, at its position. The other entries are left unchanged.

    Parameters
    ----------
    config : ReductionConfig
        The setup of every sequence, as read from ``autoconfig.json``.
    sequence : RunSequence
        The sequence, as returned by ``collect_sequence``.

    Returns
    -------
    tuple[ReductionConfig, SampleConfig]
        The updated setup, and the entry of the sequence.
    """
    samples = list(config.samples)
    matches = [index for index, sample in enumerate(samples) if sample.start_scan_num == sequence.first_run]
    if matches:
        index = matches[0]
        if len(matches) > 1:
            names = [samples[other].name for other in matches[1:]]
            logger.warning(
                f"Samples {names} of {SETUP_FILENAME} also start at run {sequence.first_run}: "
                f"only {samples[index].name} is updated"
            )
        entry = sequence_entry(sequence, samples[index].name)
        samples[index] = entry
    else:
        index = sum(1 for sample in samples if sample.start_scan_num < sequence.first_run)
        taken = {sample.name for sample in samples}
        taken.update(other.name for other in (config.background, config.empty_cell) if other is not None)
        entry = sequence_entry(sequence, sequence_name(sequence, index, taken))
        samples.insert(index, entry)
        logger.info(f"Sequence {sequence.first_run} added to {SETUP_FILENAME} as {entry.name}")
    return config.model_copy(update={"samples": samples}), entry


def sequence_setup(entry: SampleConfig, config: ReductionConfig, existing: ReductionConfig | None) -> ReductionConfig:
    """Build the setup of the reduction of one sequence, written to ``autoconfig_<N>.json``.

    Parameters
    ----------
    entry : SampleConfig
        The entry of the sequence, as returned by ``update_setup``.
    config : ReductionConfig
        The setup of every sequence. A new setup copies its background, empty cell and
        ``save_all_harmonics``.
    existing : ReductionConfig | None
        The setup read from ``autoconfig_<N>.json``, or ``None`` if the file does not exist. Only its
        samples are replaced, so that the rest, including changes made by hand, is kept.

    Returns
    -------
    ReductionConfig
        The setup, with the entry as its only sample.
    """
    if existing is not None:
        return existing.model_copy(update={"samples": [entry]})
    return ReductionConfig(
        samples=[entry],
        background=config.background,
        empty_cell=config.empty_cell,
        save_all_harmonics=config.save_all_harmonics,
    )


def read_setup_file(setup_file: str) -> ReductionConfig | None:
    """Read a setup file written by step 2, if it exists.

    Parameters
    ----------
    setup_file : str
        Path of the setup file.

    Returns
    -------
    ReductionConfig | None
        The setup, or ``None`` if the file does not exist.

    Raises
    ------
    OSError
        If the file cannot be read.
    ValueError
        If the file is not valid JSON, or is rejected by the schema of the setup files. The file is left
        unchanged, so that changes made by hand are not discarded.
    """
    if not os.path.exists(setup_file):
        return None
    try:
        return read_config(setup_file)
    except (OSError, ValueError):
        logger.error(f"{setup_file} cannot be read, and is left unchanged: correct it, or remove it to rebuild it")
        raise


def write_config(config: ReductionConfig, setup_file: str) -> None:
    """Write a setup file, to a temporary file in the same directory renamed at the end.

    A reader of the file never sees it partly written.

    Parameters
    ----------
    config : ReductionConfig
        The setup.
    setup_file : str
        Path of the setup file.
    """
    write_file(setup_file, config.model_dump_json(indent=2, exclude_none=True) + "\n")


# Margin, in seconds, by which a log of the reduction may seem older than the start of the reduction
LOG_TIME_MARGIN = 2.0


def add_reduction_logs(error: BaseException, sequence_file: str, autodir: str, started: float) -> None:
    """Add to the error of a failed reduction the logs that it wrote, ``reduction_<name>.log``.

    ``Experiment.reduce`` logs the error of a failed sample to its log file and goes on, and the error it
    raises afterwards only states that the data are missing: the cause is in the log. Each log of a
    measurement of ``sequence_file`` written since ``started`` is added as a note, which is shown after
    the traceback.

    Parameters
    ----------
    error : BaseException
        The error raised by the reduction.
    sequence_file : str
        Path of the setup file reduced, ``autoconfig_<N>.json``.
    autodir : str
        Directory of the reduced data, where the logs are written.
    started : float
        Time at which the reduction started, as returned by ``time.time()``; older logs are left out.
    """
    try:
        config = read_config(sequence_file)
    except (OSError, ValueError):
        return
    measurements = [*config.samples, config.background, config.empty_cell]
    for name in dict.fromkeys(measurement.name for measurement in measurements if measurement is not None):
        log_file = os.path.join(autodir, f"reduction_{name}.log")
        try:
            # The margin allows for file systems that store modification times to the second
            if os.path.getmtime(log_file) < started - LOG_TIME_MARGIN:
                continue
            with open(log_file) as fd:
                log = fd.read()
        except OSError:
            continue
        error.add_note(f"\nLog of the reduction, {os.path.join(REDUCTION_DIRNAME, os.path.basename(log_file))}:\n{log}")


def reduce_sequence(sequence_file: str, setup_file: str, outdir: str) -> None:
    """Reduce the setup of one sequence into ``<outdir>/auto``, and write the summary report of every sequence.

    Parameters
    ----------
    sequence_file : str
        Path of the setup file of the sequence, ``autoconfig_<N>.json``.
    setup_file : str
        Path of the setup file of every sequence, ``autoconfig.json``, from which the summary is written,
        without the ``_unscaled`` files.
    outdir : str
        Output directory of the preprocessing script.
    """
    autodir = os.path.join(outdir, REDUCTION_DIRNAME)
    os.makedirs(autodir, exist_ok=True)

    exp = reduce.Experiment(config_file=sequence_file)
    started = time.time()
    try:
        exp.reduce(output_dir=autodir)
    # The reduction can raise any error; the logs written are added to it, and it is raised again unchanged
    except Exception as error:
        add_reduction_logs(error, sequence_file, autodir, started)
        raise

    # The summary leaves out the _unscaled files (#107)
    generate_report(config_file_path=setup_file, output_dir=exp.output_dir, include_unscaled=False)


def reduce_run_sequence(outdir: str, first_run: int) -> str | None:
    """Step 2: update the setup files of the sequence of a run, and reduce that sequence alone.

    The entry of the sequence is rewritten or inserted in ``autoconfig.json``, which lists every
    sequence (see ``update_setup``). If ``autoconfig.json`` is missing, it is first built from every
    ``scan_<N>.json`` of the output directory. ``autoconfig_<N>.json`` is created, or its samples
    replaced (see ``sequence_setup``), and is reduced into ``<outdir>/auto``. The files of the other
    sequences in ``<outdir>/auto`` are left unchanged.

    Parameters
    ----------
    outdir : str
        Output directory of the preprocessing script.
    first_run : int
        Run number of the first run of the sequence, the value of the log ``BL1A:CS:Scan:USANS:FirstRun``
        of the run.

    Returns
    -------
    str | None
        Path of the reduced data of harmonic 1 of the sequence, ``<outdir>/auto/<name>_det_1.txt``, or
        ``None`` if no run of the sequence is recorded, and nothing is reduced.

    Raises
    ------
    OSError, ValueError
        If ``autoconfig.json`` or ``autoconfig_<N>.json`` cannot be read (see ``read_setup_file``).
        Neither file is then written.
    """
    sequence = collect_sequence(outdir, first_run)
    if sequence is None:
        logger.error(f"No run of sequence {first_run} recorded in {outdir}: nothing to reduce")
        return None

    setup_file = os.path.join(outdir, SETUP_FILENAME)
    sequence_file = os.path.join(outdir, f"autoconfig_{first_run}.json")
    # Read both files before writing either, so that an unreadable file leaves both unchanged
    config = read_setup_file(setup_file)
    existing = read_setup_file(sequence_file)
    if config is None:
        logger.info(f"{SETUP_FILENAME} not found: building it from every sequence of {outdir}")
        config = build_setup(collect_sequences(outdir))

    config, entry = update_setup(config, sequence)
    write_config(config, setup_file)
    write_config(sequence_setup(entry, config, existing), sequence_file)
    reduce_sequence(sequence_file, setup_file, outdir)
    return os.path.join(outdir, REDUCTION_DIRNAME, reduced_data_filename(entry.name, "det_1"))


# Instrument name under which the figure is published to the live data server
INSTRUMENT = "USANS"


def raw_scan_plot(run_number: str, angle, counts, errors, scan_var: str) -> str | None:
    """Build the panel of the raw scan of a run: counts at the detector against the analyzer angle.

    Parameters
    ----------
    run_number : str
        Run number of the run.
    angle, counts, errors : array-like
        Angle of the analyzer, in arcsec, and counts at the detector with their errors.
    scan_var : str
        Name of the log of the scanned motor, the title of the x axis.

    Returns
    -------
    str | None
        The panel, as an HTML block, or ``None`` if it cannot be built, which is logged.
    """
    try:
        # Plain lists keep the data readable in the HTML block
        data = [[list(angle), list(counts), list(errors)]]
        return plot1d(run_number, data, x_title=scan_var, y_title="Counts", y_log=True, publish=False)
    except Exception:
        logger.exception(f"The panel of the raw scan of run {run_number} cannot be built")
        return None


def reduced_sequence_plot(run_number: str, det_1_file: str) -> str | None:
    """Build the panel of the reduced sequence of a run: I(Q) of harmonic 1, on log scales.

    Parameters
    ----------
    run_number : str
        Run number of the run.
    det_1_file : str
        Path of the reduced data of harmonic 1, ``<name>_det_1.txt``, with the columns Q in 1/angstrom,
        I and dI in 1/cm.

    Returns
    -------
    str | None
        The panel, as an HTML block, or ``None`` if the file cannot be read or the panel built, which
        is logged.
    """
    if not os.path.isfile(det_1_file):
        logger.warning(f"{det_1_file} not found: the figure of run {run_number} has no reduced sequence")
        return None
    name = os.path.basename(det_1_file).removesuffix(".txt")
    try:
        # The rows end with a comma, so only the first three columns are read
        q, intensity, error = np.genfromtxt(det_1_file, delimiter=",", usecols=(0, 1, 2), ndmin=2, unpack=True)
        return plot1d(
            run_number,
            [[q.tolist(), intensity.tolist(), error.tolist()]],
            data_names=[name],
            x_title="Q (1/A)",
            y_title="I (1/cm)",
            x_log=True,
            y_log=True,
            title=name,
            publish=False,
        )
    except Exception:
        logger.exception(f"The panel of {det_1_file} cannot be built")
        return None


# Headings of the block shown in the figure of a run when step 1 or step 2 fails
PREPROCESSING_FAILED = "Preprocessing Failed"
REDUCTION_FAILED = "Reduction Failed"


def failure_block(heading: str, error: BaseException) -> str:
    """Build the block shown in the figure of a run when step 1 or step 2 fails: a heading and the traceback.

    The traceback ends with the notes added to the error, such as the log of a failed reduction. It is
    escaped, so that characters such as ``<`` in a message are shown as text.

    Parameters
    ----------
    heading : str
        The heading, ``PREPROCESSING_FAILED`` or ``REDUCTION_FAILED``.
    error : BaseException
        The error raised.

    Returns
    -------
    str
        The block, as HTML.
    """
    trace = "".join(traceback.format_exception(error))
    return f"<h2>{heading}</h2>\n<pre>{html.escape(trace)}</pre>"


def save_figure(run_number: str, panels: list[str | None], outdir: str) -> str | None:
    """Save the figure of a run as ``<outdir>/USANS_<run>.html``.

    The page loads plotly.js from cdn.plot.ly, at the version of the installed ``plotly``. A failure to
    write the page is logged.

    Parameters
    ----------
    run_number : str
        Run number of the run.
    panels : list[str | None]
        The panels, as HTML blocks; ``None`` for a panel that could not be built.
    outdir : str
        Output directory of the preprocessing script.

    Returns
    -------
    str | None
        The figure, the panels joined, or ``None`` if there is no panel, and nothing is saved.
    """
    panels = [panel for panel in panels if panel is not None]
    if not panels:
        logger.warning(f"No panel to show for run {run_number}: no figure is saved or uploaded")
        return None
    figure = "\n".join(panels)
    page = (
        "<!DOCTYPE html>\n"
        '<html lang="en">\n'
        "<head>\n"
        '<meta charset="UTF-8">\n'
        f"<title>{INSTRUMENT} run {run_number}</title>\n"
        f'<script src="https://cdn.plot.ly/plotly-{get_plotlyjs_version()}.min.js"></script>\n'
        "</head>\n"
        "<body>\n"
        f"{figure}\n"
        "</body>\n"
        "</html>\n"
    )
    page_file = os.path.join(outdir, f"{INSTRUMENT}_{run_number}.html")
    try:
        with open(page_file, "w") as fd:
            fd.write(page)
    except OSError:
        logger.exception(f"{page_file} cannot be written")
    return figure


def upload_figure(run_number: str, figure: str) -> None:
    """Upload the figure of a run to the live data server, under the run.

    A failure of the request (an error answer, no connection, a timeout) is logged, and the script
    goes on. A missing or invalid publisher configuration raises an error.

    Parameters
    ----------
    run_number : str
        Run number of the run.
    figure : str
        The figure, as returned by ``save_figure``.
    """
    try:
        publish_plot(INSTRUMENT, run_number, files={"file": figure})
    except requests.RequestException:
        logger.exception(f"The figure of run {run_number} cannot be uploaded to the live data server")


def reduce_and_show(outdir: str, first_run: int, run_number: str, raw_scan: str | None, no_publish: bool) -> None:
    """Step 2, then the figure of the run: saved, and uploaded unless ``no_publish``.

    The figure has the panel of the raw scan and that of the reduced sequence. If step 2 fails, the
    reduced sequence is replaced with the heading ``Reduction Failed`` and the traceback of the error, so
    that the cause can be read on the live data server; the error is then raised again, so that the script
    ends with an error.

    Parameters
    ----------
    outdir : str
        Output directory of the preprocessing script.
    first_run : int
        Run number of the first run of the sequence.
    run_number : str
        Run number of the run.
    raw_scan : str | None
        The panel of the raw scan, as returned by ``raw_scan_plot``.
    no_publish : bool
        If true, the figure is saved but not uploaded.
    """
    try:
        det_1_file = reduce_run_sequence(outdir, first_run)
    # The reduction can raise any error; it is shown in the figure, then raised again unchanged
    except Exception as error:
        show_figure(run_number, [raw_scan, failure_block(REDUCTION_FAILED, error)], outdir, no_publish)
        raise
    reduced = reduced_sequence_plot(run_number, det_1_file) if det_1_file is not None else None
    show_figure(run_number, [raw_scan, reduced], outdir, no_publish)


def show_figure(run_number: str, panels: list[str | None], outdir: str, no_publish: bool) -> None:
    """Save the figure of a run, and upload it unless ``no_publish``.

    Parameters
    ----------
    run_number : str
        Run number of the run.
    panels : list[str | None]
        The panels, as HTML blocks; ``None`` for a panel that could not be built.
    outdir : str
        Output directory of the preprocessing script.
    no_publish : bool
        If true, the figure is saved but not uploaded.
    """
    figure = save_figure(run_number, panels, outdir)
    if figure is not None and not no_publish:
        upload_figure(run_number, figure)


# Prefix of the logs of the motors of the instrument
MOTOR_LOG_PREFIX = "BL1A:Mot:"


class PreprocessingError(RuntimeError):
    """A run that step 1 cannot preprocess, for a reason other than an error of Mantid."""


@dataclass
class RunFigure:
    """The panels of the figure of a run built so far, kept when step 1 or step 2 fails.

    Parameters
    ----------
    raw_scan : str | None
        The panel of the raw scan, as returned by ``raw_scan_plot``, or ``None`` until it is built.
    """

    raw_scan: str | None = None


def scanned_motors(run) -> list[str]:
    """Find the motors scanned during a run: those whose log varies.

    A log ``BL1A:Mot:<motor>`` varies when its standard deviation is above 1 % of the absolute value of
    its mean, or above 0 if its mean is 0. The readback logs, ending in ``.RBV``, are not considered.

    Parameters
    ----------
    run : mantid.api.Run
        The run information of the workspace of the events.

    Returns
    -------
    list[str]
        Names of the logs of the scanned motors, such as ``BL1A:Mot:ARN``, in the order of the logs of the run.
    """
    motors = []
    for item in run.getProperties():
        if item.name.startswith(MOTOR_LOG_PREFIX) and not item.name.endswith(".RBV"):
            stats = item.getStatistics()
            if (
                abs(stats.mean) > 0
                and abs(stats.standard_deviation / stats.mean) > 0.01
                or abs(stats.mean) == 0
                and abs(stats.standard_deviation) > 0.0
            ):
                motors.append(item.name)
    return motors


def preprocess_run(
    filename: str, outdir: str, first_run: int, run_number: str, load_monitors: bool, figure: RunFigure
) -> None:
    """Step 1: write the files of a run whose events are loaded in the workspace ``USANS``, and record the run.

    The run is recorded in ``scan_<N>.json`` and ``sample_<N>.json`` only once all its files are written,
    so that a run on which this step fails is not recorded, and step 2 excludes it (#107).

    Parameters
    ----------
    filename : str
        Path of the event file of the run.
    outdir : str
        Output directory of the preprocessing script.
    first_run : int
        Run number of the first run of the sequence, as returned by ``as_run_number``.
    run_number : str
        Run number of the run.
    load_monitors : bool
        Whether the monitors are loaded, in the workspace ``USANS_monitors``.
    figure : RunFigure
        Receives the panel of the raw scan as soon as it is built, so that it is shown even if a later
        part of this step fails.

    Raises
    ------
    PreprocessingError
        If no motor is scanned during the run.
    """

    # LoadEventNexus(filename, LoadMonitors=False, OutputWorkspace="USANS")
    # load_monitors = False

    file_prefix = os.path.split(filename)[1].split(".")[0]

    # if mtd['USANS'].getRun().hasProperty("BL1A:CS:Scan:USANS:Wavelength"):
    #    main_wl = mtd['USANS'].getRun().getProperty("BL1A:CS:Scan:USANS:Wavelength").value[0]
    # else:
    #    main_wl = "main_peak"

    # Get ROI from logs
    wavelength = [3.6, 1.8, 1.2, 0.9, 0.72, 0.6]
    roi_min = mtd["USANS"].getRun().getProperty("BL1A:Det:N1:Det1:TOF:ROI:1:Min").value[-1]
    # roi_step = mtd["USANS"].getRun().getProperty("BL1A:Det:N1:Det1:TOF:ROI:1:Size").value[-1]

    # Reference to the item in the wavelength array
    main_index = 1
    # Time-of-flight regions of the detector, in microseconds: the main region, then one per harmonic.
    # Local to the call, so that two calls in one process, as in the tests, do not add to the same list.
    peaks = []
    for i in range(1, 8):
        lower_bound = mtd["USANS"].getRun().getProperty("BL1A:Det:N1:Det1:TOF:ROI:%s:Min" % i).value[-1]
        tof_step = mtd["USANS"].getRun().getProperty("BL1A:Det:N1:Det1:TOF:ROI:%s:Size" % i).value[-1]
        if i > 1 and lower_bound == roi_min:
            main_index = i - 2
        peaks.append([lower_bound * 1000.0, (lower_bound + tof_step) * 1000.0])
    main_wl = wavelength[main_index]

    # Produce ASCII data
    Rebin(InputWorkspace="USANS", Params="0,10,17000", OutputWorkspace="USANS")
    file_path = os.path.join(outdir, "%s_detector_trans.txt" % file_prefix)
    save_summed_spectra("USANS", file_path, header="TOF, COUNTS, ERROR")

    CropWorkspace(
        InputWorkspace="USANS",
        StartWorkspaceIndex=0,
        EndWorkspaceIndex=1023,
        OutputWorkspace="USANS_detector",
    )
    file_path = os.path.join(outdir, "%s_detector.txt" % file_prefix)
    save_summed_spectra("USANS_detector", file_path, header="TOF, COUNTS, ERROR")

    CropWorkspace(
        InputWorkspace="USANS",
        StartWorkspaceIndex=1024,
        EndWorkspaceIndex=2047,
        OutputWorkspace="USANS_trans",
    )
    file_path = os.path.join(outdir, "%s_trans.txt" % file_prefix)
    save_summed_spectra("USANS_trans", file_path, header="TOF, COUNTS, ERROR")

    if load_monitors:
        Rebin(
            InputWorkspace="USANS_monitors",
            Params="0,10,17000",
            OutputWorkspace="USANS_monitors",
        )
        file_path = os.path.join(outdir, "%s_monitor.txt" % file_prefix)
        save_ascii("USANS_monitors", file_path, header="TOF, COUNTS, ERROR")

    # The motors scanned during the run: their logs are the x axis of the scans
    scan_vars = scanned_motors(mtd["USANS"].getRun())
    if not scan_vars:
        raise PreprocessingError(f"no log {MOTOR_LOG_PREFIX}* varies during the run, so the run is not a scan")
    for scan_var in scan_vars:
        short_name = scan_var.removeprefix(MOTOR_LOG_PREFIX)

        y_monitor = None
        if load_monitors:
            StepScan(
                InputWorkspace="USANS_monitors",
                OutputWorkspace="mon_scan_table",
            )
            ConvertTableToMatrixWorkspace(
                InputWorkspace="mon_scan_table",
                ColumnX=scan_var,
                ColumnY="Counts",
                ColumnE="Error",
                OutputWorkspace="USANS_scan_monitor",
            )
            file_path = os.path.join(outdir, "%s_monitor_scan_%s.txt" % (file_prefix, short_name))
            save_ascii("USANS_scan_monitor", file_path, header=f"{scan_var}, COUNTS, ERROR")
            y_monitor = mtd["USANS_scan_monitor"].readY(0)

        iq_file_path_simple = os.path.join(outdir, "%s_iq_%s_%s.txt" % (file_prefix, short_name, main_wl))
        iq_fd_simple = open(iq_file_path_simple, "w")

        iq_file_path = os.path.join(outdir, "%s_iq_%s.txt" % (file_prefix, short_name))
        iq_fd = open(iq_file_path, "w")

        start_time = mtd["USANS"].getRun().getProperty("start_time").value
        experiment = mtd["USANS"].getRun().getProperty("experiment_identifier").value
        run_title = mtd["USANS"].getRun().getProperty("run_title").value
        sequence_index = mtd["USANS"].getRun().getProperty("BL1A:CS:Scan:USANS:Index").value[-1]
        meta_wavelength = mtd["USANS"].getRun().getProperty("BL1A:CS:Scan:USANS:Wavelength").value[-1]
        print(f"Wavelength: {wavelength[main_index]} [{meta_wavelength}]")

        iq_fd.write("# Experiment %s Run %s\n" % (experiment, run_number))
        iq_fd.write("# Run start time: %s\n" % start_time)
        iq_fd.write("# Title: %s\n" % run_title)
        iq_fd.write("# Sequence ID: %s\n" % first_run)
        iq_fd.write("# Sequence index: %s\n" % sequence_index)
        iq_fd.write("# Selected wavelength: %s\n" % wavelength[main_index])
        iq_fd.write(
            "# %-8s %-10s %-10s %-10s %-10s %-10s %-10s %-5s\n"
            % ("Q", "I(Q)", "dI(Q)", "dQ", "N(Q)", "dN(Q)", "Mon(Q)", "Lambda")
        )
        iq_fd_simple.write("# Experiment %s Run %s\n" % (experiment, run_number))
        iq_fd_simple.write("# Run start time: %s\n" % start_time)
        iq_fd_simple.write("# Title: %s\n" % run_title)
        iq_fd_simple.write("# Sequence ID: %s\n" % first_run)
        iq_fd_simple.write("# Sequence index: %s\n" % sequence_index)
        iq_fd_simple.write("# Selected wavelength: %s\n" % wavelength[main_index])
        iq_fd_simple.write("# %-8s %-10s %-10s\n" % ("Q", "I(Q)", "dI(Q)"))

        for i in range(len(peaks)):
            peak = peaks[i]
            CropWorkspace(
                InputWorkspace="USANS_detector",
                OutputWorkspace="peak_detector",
                XMin=peak[0],
                XMax=peak[1],
            )
            StepScan(InputWorkspace="peak_detector", OutputWorkspace="scan_table")
            ConvertTableToMatrixWorkspace(
                InputWorkspace="scan_table",
                ColumnX=scan_var,
                ColumnY="Counts",
                ColumnE="Error",
                OutputWorkspace="USANS_scan_detector",
            )
            mtd["USANS_scan_detector"].getAxis(1).getUnit().setLabel("Counts", "Counts")
            x_data = mtd["USANS_scan_detector"].readX(0)  # rotation angle (arcsec) of the analyzer
            y_data = mtd["USANS_scan_detector"].readY(0)  # total counts at detector for each analyzer angle
            e_data = mtd["USANS_scan_detector"].readE(0)

            if i == 0:
                # Kept for record_run, since file_path is reassigned below
                main_scan_file = os.path.join(outdir, "%s_detector_%s.txt" % (file_prefix, main_wl))
                save_ascii("USANS_scan_detector", main_scan_file, header=f"{scan_var}, COUNTS, ERROR")
                # json_file_path = os.path.join(outdir, "%s_plot_data.json" % file_prefix)
                # SavePlot1DAsJson(
                #     InputWorkspace="USANS_scan_detector", JsonFilename=json_file_path, PlotName="main_output"
                # )

                figure.raw_scan = raw_scan_plot(run_number, x_data, y_data, e_data, scan_var)

                for i_theta in range(len(x_data)):
                    q = 2.0 * math.pi * math.sin(x_data[i_theta] * math.pi / 180.0 / 3600.0) / wavelength[main_index]
                    # if q<=0:
                    #    continue

                    # Write I(q) file
                    i_q = y_data[i_theta] / y_monitor[i_theta]
                    di_q = math.sqrt(
                        (e_data[i_theta] / y_monitor[i_theta]) ** 2 + y_data[i_theta] ** 2 / y_monitor[i_theta] ** 3
                    )
                    iq_fd_simple.write("%-10.6g %-10.6g %-10.6g\n" % (q, i_q, di_q))

            else:
                file_path = os.path.join(
                    outdir,
                    "%s_detector_scan_%s_peak_%s.txt" % (file_prefix, short_name, i),
                )
                save_ascii("USANS_scan_detector", file_path, header=f"{scan_var}, COUNTS, ERROR")
                for i_theta in range(len(x_data)):
                    q = 2.0 * math.pi * math.sin(x_data[i_theta] * math.pi / 180.0 / 3600.0) / wavelength[i - 1]
                    # if q<=0:
                    #    continue

                    # Write complete I(q) file
                    i_q = y_data[i_theta] / y_monitor[i_theta]
                    di_q = math.sqrt(
                        (e_data[i_theta] / y_monitor[i_theta]) ** 2 + y_data[i_theta] ** 2 / y_monitor[i_theta] ** 3
                    )
                    if i_q > 0:
                        iq_fd.write(
                            "%-10.6g %-10.6g %-10.6g %-10.6g %-10.6g %-10.6g %-10.6g %-5.4g\n"
                            % (
                                q,
                                i_q,
                                di_q,
                                0,
                                y_data[i_theta],
                                e_data[i_theta],
                                y_monitor[i_theta],
                                wavelength[i - 1],
                            )
                        )

            CropWorkspace(
                InputWorkspace="USANS_trans",
                OutputWorkspace="peak_trans",
                XMin=peak[0],
                XMax=peak[1],
            )
            StepScan(InputWorkspace="peak_trans", OutputWorkspace="scan_table")
            ConvertTableToMatrixWorkspace(
                InputWorkspace="scan_table",
                ColumnX=scan_var,
                ColumnY="Counts",
                OutputWorkspace="USANS_scan_trans",
            )

            if i == 0:
                file_path = os.path.join(outdir, "%s_trans_%s.txt" % (file_prefix, main_wl))
            else:
                file_path = os.path.join(
                    outdir,
                    "%s_trans_scan_%s_peak_%s.txt" % (file_prefix, short_name, i),
                )
            save_ascii("USANS_scan_trans", file_path, header=f"{scan_var}, COUNTS, ERROR")

        iq_fd.close()
        iq_fd_simple.close()

        # Record the run in its sequence only now that all its files are written, so that a run on
        # which step 1 fails is not recorded, and step 2 excludes it (#107). The first run is the value
        # validated by as_run_number, so that the file names are scan_<N>.json with N an integer.
        record_run(
            outdir,
            first_run,
            run_number,
            main_scan_file,
            run_title,
            sample_thickness(mtd["USANS"].getRun()),
        )


def parse_arguments(argv: list[str] | None = None) -> argparse.Namespace:
    """Read the command line of the preprocessing script.

    Parameters
    ----------
    argv : list[str] | None
        The arguments, or ``None`` for those of the command line.

    Returns
    -------
    argparse.Namespace
        ``event_file``, ``outdir`` and ``no_publish``.
    """
    parser = argparse.ArgumentParser(description="Preprocess a USANS run, then reduce its sequence of runs.")
    parser.add_argument("event_file", help="event file of the run")
    parser.add_argument("outdir", help="output directory")
    parser.add_argument(
        "--no_publish",
        action="store_true",
        help=f"do not upload the figure to the live data server; {INSTRUMENT}_<run>.html is still saved",
    )
    return parser.parse_args(argv)


def main():
    args = parse_arguments()
    if not (os.path.isfile(args.event_file)):
        print(f"Data file {args.event_file} not found")
        sys.exit()
    filename = args.event_file
    outdir = args.outdir

    # The runs outside a sequence are neither preprocessed nor reduced (#107)
    value = read_first_run(filename)
    first_run = as_run_number(value)
    if first_run is None or first_run <= 0:
        if value is None:
            reason = f"it has no log {FIRST_RUN_LOG}"
        else:
            # repr() quotes text, so that "Not Set" is told apart from a number
            shown = repr(value) if isinstance(value, str) else value
            reason = f"its log {FIRST_RUN_LOG} is {shown}, which is not a run number"
        logger.info(f"{filename} is not processed: {reason}, so the run is not part of any sequence")
        return

    # Don't load monitors unless we really need them
    try:
        LoadEventNexus(filename, LoadMonitors=True, OutputWorkspace="USANS")
        load_monitors = True
    except:  # noqa E722
        LoadEventNexus(filename, LoadMonitors=False, OutputWorkspace="USANS")
        load_monitors = False
    run_number = mtd["USANS"].getRun().getProperty("run_number").value

    # A failure of step 1 or step 2 is shown in the figure of the run, then raised again (#107)
    figure = RunFigure()
    try:
        preprocess_run(filename, outdir, first_run, run_number, load_monitors, figure)
    # Step 1 can raise any error; it is shown in the figure, then raised again unchanged
    except Exception as error:
        show_figure(run_number, [figure.raw_scan, failure_block(PREPROCESSING_FAILED, error)], outdir, args.no_publish)
        raise
    reduce_and_show(outdir, first_run, run_number, figure.raw_scan, args.no_publish)


if __name__ == "__main__":
    main()
