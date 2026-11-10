"""Tests of the preprocessing script (#107).

Step 1 preprocesses one run and records it in the files of its sequence. Step 2 goes from the files of
the sequences to the reduction.
"""

import json
import logging
import shutil
import sys
from pathlib import Path

import h5py
import numpy as np
import pytest

from usansred import reduce_USANS
from usansred.io.read import read_config
from usansred.reduce_USANS import (
    FIRST_RUN_LOG,
    THICKNESS_LOG,
    build_setup,
    collect_sequences,
    main,
    read_first_run,
    record_run,
    reduce_run_sequence,
    reduce_sequence,
    sample_thickness,
    write_config,
)

DATA_DIR = Path(__file__).parent.parent / "data" / "preprocessing"

# First run and number of recorded runs of each sequence of the fixtures, from IPTS-30410
SEQUENCES = {41713: 6, 41741: 6, 41832: 6}

# Titles of the sequences of the fixtures, as recorded in sample_<N>.json
TITLES = {41713: "Align:S115_dry", 41741: "Align:EmptyPCell", 41832: "Align:S115_pc3"}

# Runs of the fixture scan_-1.json, recorded just before each sequence
NO_SEQUENCE_RUNS = {"41712", "41740", "41831"}


@pytest.fixture
def outdir(tmp_path: Path) -> Path:
    """An output directory of the preprocessing script, holding the sequence files of the fixtures."""
    for path in DATA_DIR.glob("*.json"):
        shutil.copy(path, tmp_path)
    return tmp_path


class LogCapture:
    """The messages logged by the preprocessing script."""

    def __init__(self, caplog: pytest.LogCaptureFixture):
        self.caplog = caplog

    @property
    def messages(self) -> list[str]:
        """Messages at level INFO and above."""
        return self.caplog.messages

    def records_at_warning(self) -> list[str]:
        """Messages at level WARNING and above."""
        return [record.getMessage() for record in self.caplog.records if record.levelno >= logging.WARNING]


@pytest.fixture
def log(monkeypatch, caplog) -> LogCapture:
    """Captures the messages logged by the preprocessing script, at level INFO and above."""
    # The logger of the module does not propagate to the root logger, where caplog listens
    monkeypatch.setattr(reduce_USANS.logger, "propagate", True)
    caplog.set_level(logging.INFO)
    return LogCapture(caplog)


def write_scan_file(
    outdir: Path,
    first_run: int,
    num_of_runs: int = 1,
    runs: list[int] | None = None,
    titles: str | list[str | None] | None = None,
) -> None:
    """Write a scan_<N>.json file for a sequence starting at ``first_run``.

    The sequence holds ``runs``, in the order given, or else ``num_of_runs`` consecutive runs. ``titles``
    is the title of every run, or a list with the title of each run (``None`` for a run without a title),
    or ``None`` for runs without a title, as written before the titles were recorded.
    """
    if runs is None:
        runs = list(range(first_run, first_run + num_of_runs))
    if not isinstance(titles, list):
        titles = [titles] * len(runs)
    entries = {}
    for run, title in zip(runs, titles, strict=True):
        entries[str(run)] = {"iq": f"USANS_{run}_detector_3.6.txt"}
        if title is not None:
            entries[str(run)]["title"] = title
    (outdir / f"scan_{first_run}.json").write_text(json.dumps(entries))


def sample_names(outdir: Path) -> list[str]:
    """Names of the samples of the setup built from the sequence files of ``outdir``."""
    return [sample.name for sample in build_setup(collect_sequences(str(outdir))).samples]


def write_sample_file(outdir: Path, first_run: int, **sample_info) -> None:
    """Change the values of the sample_<N>.json file of a fixture sequence."""
    sample_file = outdir / f"sample_{first_run}.json"
    sample_file.write_text(json.dumps(json.loads(sample_file.read_text()) | sample_info))


def write_event_file(path: Path, first_run: list[int] | None) -> Path:
    """Write an HDF5 file holding only the log FirstRun of an event file, with the values given.

    Without values, the file has no such log.
    """
    with h5py.File(path, "w") as nexus:
        logs = nexus.create_group("entry/DASlogs")
        if first_run is not None:
            logs.create_dataset(f"{FIRST_RUN_LOG}/value", data=np.array(first_run, dtype=np.int32))
    return path


class TestReadFirstRun:
    def test_no_sequence(self, tmp_path):
        event_file = write_event_file(tmp_path / "USANS_41712.nxs.h5", [-1])

        assert read_first_run(str(event_file)) == -1

    def test_sequence(self, tmp_path):
        event_file = write_event_file(tmp_path / "USANS_41714.nxs.h5", [41713])

        assert read_first_run(str(event_file)) == 41713

    def test_last_value(self, tmp_path):
        event_file = write_event_file(tmp_path / "USANS_41713.nxs.h5", [-1, 41713])

        assert read_first_run(str(event_file)) == 41713

    def test_no_value(self, tmp_path):
        event_file = write_event_file(tmp_path / "USANS_41713.nxs.h5", [])

        assert read_first_run(str(event_file)) is None

    def test_no_log(self, tmp_path):
        event_file = write_event_file(tmp_path / "USANS_41713.nxs.h5", None)

        assert read_first_run(str(event_file)) is None

    def test_not_an_hdf5_file(self, tmp_path):
        event_file = tmp_path / "USANS_41713.nxs.h5"
        event_file.write_text("not an HDF5 file")

        with pytest.raises(OSError, match="file signature not found"):
            read_first_run(str(event_file))


class TestMainSkipsRuns:
    @pytest.mark.parametrize(
        ("first_run", "reason"),
        [
            ([-1], f"its log {FIRST_RUN_LOG} is -1, which is not a run number"),
            ([41845, 0], f"its log {FIRST_RUN_LOG} is 0, which is not a run number"),
            ([-5], f"its log {FIRST_RUN_LOG} is -5, which is not a run number"),
            (None, f"it has no log {FIRST_RUN_LOG}"),
        ],
        ids=["minus_one", "zero", "negative", "no_log"],
    )
    def test_run_outside_a_sequence(self, tmp_path, monkeypatch, log, first_run, reason):
        event_file = write_event_file(tmp_path / "USANS_41712.nxs.h5", first_run)
        outdir = tmp_path / "shared"
        outdir.mkdir()

        def load_event_nexus(*_args, **_kwargs):
            pytest.fail("the events of a run outside a sequence are loaded")

        monkeypatch.setattr(reduce_USANS, "LoadEventNexus", load_event_nexus)
        monkeypatch.setattr(sys, "argv", ["reduce_USANS.py", str(event_file), str(outdir)])

        main()

        assert list(outdir.iterdir()) == []
        assert any("is not processed" in message and reason in message for message in log.messages)


class StandInRun:
    """Stands in for mantid.api.Run, holding the logs given as keyword arguments."""

    class Property:
        def __init__(self, value):
            self.value = value

    def __init__(self, **logs):
        self.logs = logs

    def hasProperty(self, name: str) -> bool:  # noqa: N802
        return name in self.logs

    def getProperty(self, name: str) -> "StandInRun.Property":  # noqa: N802
        return self.Property(self.logs[name])


class TestSampleThickness:
    def test_number(self):
        assert sample_thickness(StandInRun(**{THICKNESS_LOG: [0.3]})) == 0.3

    def test_last_value(self):
        assert sample_thickness(StandInRun(**{THICKNESS_LOG: [0.2, 0.3]})) == 0.3

    def test_numeric_string(self):
        assert sample_thickness(StandInRun(**{THICKNESS_LOG: ["0.3"]})) == 0.3

    @pytest.mark.parametrize(
        "value", [["Not Set"], [], [0.0], [-0.3], [float("nan")]], ids=["not_set", "empty", "zero", "negative", "nan"]
    )
    def test_not_a_thickness(self, log, value):
        assert sample_thickness(StandInRun(**{THICKNESS_LOG: value})) == 0.1
        assert any(THICKNESS_LOG in message for message in log.messages)

    def test_no_log(self, log):
        assert sample_thickness(StandInRun()) == 0.1
        assert any(THICKNESS_LOG in message for message in log.messages)


class TestRecordRun:
    def test_title_in_scan_file(self, tmp_path):
        record_run(str(tmp_path), 41713, "41713", "USANS_41713_detector_3.6.txt", "Align:S115_dry", 0.3)

        scan = json.loads((tmp_path / "scan_41713.json").read_text())
        assert scan == {"41713": {"iq": "USANS_41713_detector_3.6.txt", "title": "Align:S115_dry"}}

    def test_thickness_in_sample_file(self, tmp_path):
        record_run(str(tmp_path), 41713, "41713", "USANS_41713_detector_3.6.txt", "Align:S115_dry", 0.3)

        sample = json.loads((tmp_path / "sample_41713.json").read_text())
        assert sample == {"title": "Align:S115_dry", "background": 0, "thickness": 0.3}

    def test_second_run(self, tmp_path):
        record_run(str(tmp_path), 41713, "41713", "USANS_41713_detector_3.6.txt", "Align:S115_dry", 0.1)
        record_run(str(tmp_path), 41713, "41714", "USANS_41714_detector_3.6.txt", "Align:S115_dry", 0.1)

        scan = json.loads((tmp_path / "scan_41713.json").read_text())
        assert list(scan) == ["41713", "41714"]

    def test_replaces_thickness_edited_by_hand(self, tmp_path):
        sample_file = tmp_path / "sample_41713.json"
        sample_file.write_text(json.dumps({"title": "Align:S115_dry", "background": 0, "thickness": 0.5}))

        record_run(str(tmp_path), 41713, "41714", "USANS_41714_detector_3.6.txt", "Align:S115_dry", 0.1)

        assert json.loads(sample_file.read_text())["thickness"] == 0.1

    def test_step_2_reads_recorded_runs(self, tmp_path):
        for run in range(41713, 41719):
            record_run(str(tmp_path), 41713, str(run), f"USANS_{run}_detector_3.6.txt", "Align:S115_dry", 0.3)

        (sample,) = build_setup(collect_sequences(str(tmp_path))).samples

        assert (sample.start_scan_num, sample.num_of_scans, sample.thickness) == (41713, 6, 0.3)


class TestCollectSequences:
    def test_one_sequence_per_scan_file(self, outdir):
        sequences = collect_sequences(str(outdir))

        assert [sequence.first_run for sequence in sequences] == list(SEQUENCES)
        assert [len(sequence.runs) for sequence in sequences] == list(SEQUENCES.values())

    def test_ignores_runs_outside_a_sequence(self, outdir, log):
        sequences = collect_sequences(str(outdir))

        assert NO_SEQUENCE_RUNS.isdisjoint(run for sequence in sequences for run in sequence.runs)
        assert any("scan_-1.json" in message and "3 runs" in message for message in log.messages)

    def test_ignores_first_run_zero(self, outdir, log):
        # In IPTS-30410, runs 41845-41855 have FirstRun = 0
        write_scan_file(outdir, 0, runs=[41845, 41846, 41847])

        sequences = collect_sequences(str(outdir))

        assert [sequence.first_run for sequence in sequences] == list(SEQUENCES)
        assert any(
            "scan_0.json" in message and "3 runs" in message and "FirstRun = 0" in message for message in log.messages
        )

    def test_orders_by_run_number(self, tmp_path):
        write_scan_file(tmp_path, 10001)
        write_scan_file(tmp_path, 9990)

        sequences = collect_sequences(str(tmp_path))

        assert [sequence.first_run for sequence in sequences] == [9990, 10001]

    def test_ignores_file_without_run_number(self, outdir, log):
        (outdir / "scan_old.json").write_text(json.dumps({"41713": {"iq": "USANS_41713_detector_1.8.txt"}}))

        sequences = collect_sequences(str(outdir))

        assert [sequence.first_run for sequence in sequences] == list(SEQUENCES)
        assert any("scan_old.json" in message for message in log.messages)

    def test_no_scan_file(self, tmp_path):
        assert collect_sequences(str(tmp_path)) == []

    def test_reads_sample_files(self, outdir):
        sequences = collect_sequences(str(outdir))

        assert [sequence.sample["title"] for sequence in sequences] == list(TITLES.values())

    def test_missing_sample_file(self, outdir, log):
        (outdir / "sample_41741.json").unlink()

        sequences = collect_sequences(str(outdir))

        assert sequences[1].sample == {}
        assert any("sample_41741.json" in message for message in log.messages)


class TestBuildSetup:
    def test_every_sequence_is_a_sample(self, outdir):
        config = build_setup(collect_sequences(str(outdir)))

        assert [sample.name for sample in config.samples] == ["S115_dry", "EmptyPCell", "S115_pc3"]
        assert [sample.start_scan_num for sample in config.samples] == list(SEQUENCES)
        assert [sample.thickness for sample in config.samples] == [0.1, 0.1, 0.1]
        assert config.background is None
        assert config.empty_cell is None

    def test_sequence_flagged_as_background_is_a_sample(self, outdir):
        write_sample_file(outdir, 41741, background=1)

        config = build_setup(collect_sequences(str(outdir)))

        assert len(config.samples) == len(SEQUENCES)
        assert config.background is None

    def test_single_sequence(self, tmp_path):
        write_scan_file(tmp_path, 41713, num_of_runs=6)

        config = build_setup(collect_sequences(str(tmp_path)))

        assert len(config.samples) == 1
        assert config.background is None

    def test_keeps_every_run(self, outdir):
        config = build_setup(collect_sequences(str(outdir)))

        assert [sample.num_of_scans for sample in config.samples] == list(SEQUENCES.values())
        assert [sample.exclude for sample in config.samples] == [[], [], []]

    def test_excludes_missing_runs(self, tmp_path):
        write_scan_file(tmp_path, 100, runs=[100, 101, 103])

        (sample,) = build_setup(collect_sequences(str(tmp_path))).samples

        assert sample.start_scan_num == 100
        assert sample.num_of_scans == 4
        assert sample.exclude == [102]

    def test_span_starts_at_first_run(self, tmp_path):
        """Run 100, the first run of the sequence, was not recorded."""
        write_scan_file(tmp_path, 100, runs=[101, 102])

        (sample,) = build_setup(collect_sequences(str(tmp_path))).samples

        assert (sample.start_scan_num, sample.num_of_scans, sample.exclude) == (100, 3, [100])

    def test_leaves_out_runs_before_first_run(self, tmp_path, log):
        write_scan_file(tmp_path, 100, runs=[99, 100, 101])

        (sample,) = build_setup(collect_sequences(str(tmp_path))).samples

        assert (sample.start_scan_num, sample.num_of_scans, sample.exclude) == (100, 2, [])
        assert any("[99]" in message and "scan_100.json" in message for message in log.records_at_warning())

    def test_order_of_runs_in_scan_file(self, tmp_path):
        write_scan_file(tmp_path, 100, runs=[103, 100, 101])
        write_scan_file(tmp_path, 200, runs=[200, 201, 203])

        first, second = build_setup(collect_sequences(str(tmp_path))).samples

        assert (first.start_scan_num, first.num_of_scans, first.exclude) == (100, 4, [102])
        assert (second.start_scan_num, second.num_of_scans, second.exclude) == (200, 4, [202])

    def test_thickness_from_sample_file(self, outdir):
        write_sample_file(outdir, 41741, thickness=0.3)

        config = build_setup(collect_sequences(str(outdir)))

        assert [sample.thickness for sample in config.samples] == [0.1, 0.3, 0.1]

    def test_thickness_as_string(self, outdir):
        write_sample_file(outdir, 41741, thickness="0.3")

        config = build_setup(collect_sequences(str(outdir)))

        assert config.samples[1].thickness == 0.3

    def test_default_thickness_without_sample_file(self, outdir):
        (outdir / "sample_41741.json").unlink()

        config = build_setup(collect_sequences(str(outdir)))

        assert config.samples[1].thickness == 0.1

    def test_saves_only_harmonic_1(self, outdir):
        config = build_setup(collect_sequences(str(outdir)))

        assert config.save_all_harmonics is False


class TestSampleNames:
    def test_names_from_sample_files(self, outdir, log):
        """The fixtures were written before the run titles were recorded in scan_<N>.json."""
        names = sample_names(outdir)

        assert names == [title.removeprefix("Align:") for title in TITLES.values()]
        assert any("sample_41713.json" in message and "Align:S115_dry" in message for message in log.messages)

    def test_name_from_first_run(self, outdir):
        write_scan_file(outdir, 41741, num_of_runs=6, titles="EmptyCell")

        assert sample_names(outdir)[1] == "EmptyCell"

    def test_first_run_is_lowest_run_number(self, tmp_path):
        write_scan_file(tmp_path, 100, runs=[102, 100, 101], titles=["Third", "First", "Second"])

        assert sample_names(tmp_path) == ["First"]

    def test_removes_align_prefix(self, tmp_path, log):
        write_scan_file(tmp_path, 100, titles="Align:S115_dry")

        assert sample_names(tmp_path) == ["S115_dry"]
        assert any("Align:" in message and "removed" in message for message in log.messages)

    def test_removes_white_space(self, tmp_path):
        write_scan_file(tmp_path, 100, titles="  Align: S115 dry \t")

        assert sample_names(tmp_path) == ["S115 dry"]

    def test_replaces_separators(self, tmp_path, log):
        write_scan_file(tmp_path, 100, titles="a/b\\c:d")

        assert sample_names(tmp_path) == ["a_b_c_d"]
        assert any("replaced" in message and "a_b_c_d" in message for message in log.messages)

    @pytest.mark.parametrize("title", [None, "", "   ", "Align:"], ids=["none", "empty", "white_space", "align"])
    def test_no_title(self, tmp_path, log, title):
        write_scan_file(tmp_path, 100, titles="First")
        write_scan_file(tmp_path, 200, titles=title)

        assert sample_names(tmp_path) == ["First", "Sample_1"]
        assert any("200" in message and "Sample_1" in message for message in log.records_at_warning())

    def test_repeated_names(self, tmp_path, log):
        for first_run in (100, 200, 300):
            write_scan_file(tmp_path, first_run, titles="EmptyCell")

        assert sample_names(tmp_path) == ["EmptyCell", "EmptyCell_2", "EmptyCell_3"]
        warnings = log.records_at_warning()
        assert any("200" in message and "EmptyCell_2" in message for message in warnings)
        assert any("300" in message and "EmptyCell_3" in message for message in warnings)

    def test_repeated_names_after_cleaning(self, tmp_path):
        write_scan_file(tmp_path, 100, titles="a/b")
        write_scan_file(tmp_path, 200, titles="Align:a:b")

        assert sample_names(tmp_path) == ["a_b", "a_b_2"]

    def test_counter_skips_name_of_a_title(self, tmp_path):
        write_scan_file(tmp_path, 100, titles="EmptyCell")
        write_scan_file(tmp_path, 200, titles="EmptyCell_2")
        write_scan_file(tmp_path, 300, titles="EmptyCell")

        assert sample_names(tmp_path) == ["EmptyCell", "EmptyCell_2", "EmptyCell_3"]

    def test_title_repeating_a_counter(self, tmp_path):
        write_scan_file(tmp_path, 100, titles="EmptyCell")
        write_scan_file(tmp_path, 200, titles="EmptyCell")
        write_scan_file(tmp_path, 300, titles="EmptyCell_2")

        assert sample_names(tmp_path) == ["EmptyCell", "EmptyCell_2", "EmptyCell_2_2"]

    def test_fallback_repeating_a_title(self, tmp_path):
        write_scan_file(tmp_path, 100, titles="Sample_1")
        write_scan_file(tmp_path, 200)

        assert sample_names(tmp_path) == ["Sample_1", "Sample_1_2"]

    def test_differing_titles(self, tmp_path, log):
        write_scan_file(tmp_path, 100, num_of_runs=4, titles=["S115_dry", "S115_dry", "S115_wet", "S115_pc3"])

        assert sample_names(tmp_path) == ["S115_dry"]
        assert any("[102, 103]" in message and "'S115_dry'" in message for message in log.records_at_warning())

    def test_runs_without_title_not_compared(self, tmp_path, log):
        write_scan_file(tmp_path, 100, num_of_runs=3, titles=["S115_dry", None, "S115_dry"])

        assert sample_names(tmp_path) == ["S115_dry"]
        assert any("[101]" in message and "no title" in message for message in log.messages)
        assert not any("different" in message for message in log.records_at_warning())


class TestWriteConfig:
    def test_writes_setup_file(self, outdir):
        config = build_setup(collect_sequences(str(outdir)))
        setup_file = str(outdir / "autoconfig.json")

        write_config(config, setup_file)

        assert read_config(setup_file) == config
        assert read_config(setup_file).save_all_harmonics is False

    def test_writes_excluded_runs(self, tmp_path):
        write_scan_file(tmp_path, 100, runs=[100, 101, 103])
        config = build_setup(collect_sequences(str(tmp_path)))
        setup_file = tmp_path / "autoconfig.json"

        write_config(config, str(setup_file))

        assert json.loads(setup_file.read_text())["samples"][0]["exclude"] == [102]
        assert read_config(str(setup_file)) == config

    def test_replaces_file_and_leaves_no_temporary_file(self, outdir):
        setup_file = outdir / "autoconfig.json"
        setup_file.write_text("earlier content")
        files_before = set(outdir.iterdir())

        write_config(build_setup(collect_sequences(str(outdir))), str(setup_file))

        assert set(outdir.iterdir()) == files_before
        assert len(read_config(str(setup_file)).samples) == len(SEQUENCES)


class TestReduceSequence:
    def test_reduces_sequence_and_reports_every_sequence(self, tmp_path, monkeypatch):
        reduction = StandInReduction(monkeypatch)
        sequence_file, setup_file = str(tmp_path / "autoconfig_100.json"), str(tmp_path / "autoconfig.json")

        reduce_sequence(sequence_file, setup_file, str(tmp_path))

        autodir = str(tmp_path / "auto")
        assert reduction.reduced == [(sequence_file, autodir)]
        assert reduction.reports == [{"config_file_path": setup_file, "output_dir": autodir}]


class StandInReduction:
    """Stands in for usansred.reduce.Experiment and generate_report, recording their calls."""

    def __init__(self, monkeypatch: pytest.MonkeyPatch):
        self.reduced = []
        self.reports = []
        reduction = self

        class StandInExperiment:
            def __init__(self, config_file: str):
                self.config_file = config_file
                self.output_dir = None

            def reduce(self, output_dir: str | None = None) -> None:
                self.output_dir = output_dir
                reduction.reduced.append((self.config_file, output_dir))

        monkeypatch.setattr(reduce_USANS.reduce, "Experiment", StandInExperiment)
        monkeypatch.setattr(reduce_USANS, "generate_report", lambda **kwargs: reduction.reports.append(kwargs))


def read_setup(outdir: Path, first_run: int | None = None) -> dict:
    """Content of autoconfig.json, or of autoconfig_<N>.json for the sequence starting at ``first_run``."""
    filename = "autoconfig.json" if first_run is None else f"autoconfig_{first_run}.json"
    return json.loads((outdir / filename).read_text())


def write_setup(outdir: Path, setup: dict, first_run: int | None = None) -> None:
    """Write autoconfig.json, or autoconfig_<N>.json, by hand."""
    filename = "autoconfig.json" if first_run is None else f"autoconfig_{first_run}.json"
    (outdir / filename).write_text(json.dumps(setup))


def entry(name: str, start_scan_num: int, num_of_scans: int, exclude: list[int] | None = None) -> dict:
    """An entry of a setup file, as step 2 writes it, with the default thickness."""
    sample = {"name": name, "start_scan_num": start_scan_num, "num_of_scans": num_of_scans, "thickness": 0.1}
    return sample | {"exclude": exclude or []}


@pytest.mark.usefixtures("reduction")
class TestReduceRunSequence:
    @pytest.fixture
    def reduction(self, monkeypatch) -> StandInReduction:
        return StandInReduction(monkeypatch)

    def test_builds_missing_setup_from_every_sequence(self, outdir):
        reduce_run_sequence(str(outdir), 41741)

        assert [sample["start_scan_num"] for sample in read_setup(outdir)["samples"]] == list(SEQUENCES)
        assert read_setup(outdir, 41741) == {"samples": [entry("EmptyPCell", 41741, 6)], "save_all_harmonics": False}
        assert not (outdir / "autoconfig_41713.json").exists()
        assert not (outdir / "autoconfig_41832.json").exists()

    def test_reduces_only_the_sequence(self, outdir, reduction):
        reduce_run_sequence(str(outdir), 41741)

        autodir = str(outdir / "auto")
        assert reduction.reduced == [(str(outdir / "autoconfig_41741.json"), autodir)]
        assert reduction.reports == [{"config_file_path": str(outdir / "autoconfig.json"), "output_dir": autodir}]

    def test_inserts_in_order_of_first_run(self, tmp_path):
        for first_run, title in ((100, "First"), (300, "Third"), (200, "Second")):
            write_scan_file(tmp_path, first_run, num_of_runs=2, titles=title)
            reduce_run_sequence(str(tmp_path), first_run)

        assert read_setup(tmp_path)["samples"] == [
            entry("First", 100, 2),
            entry("Second", 200, 2),
            entry("Third", 300, 2),
        ]

    def test_rewrites_only_the_entry_of_the_sequence(self, outdir):
        reduce_run_sequence(str(outdir), 41713)
        setup = read_setup(outdir)
        setup["samples"][1]["thickness"] = 0.5
        setup["samples"][2]["name"] = "Renamed"
        write_setup(outdir, setup)
        write_scan_file(outdir, 41713, num_of_runs=7)

        reduce_run_sequence(str(outdir), 41713)

        samples = read_setup(outdir)["samples"]
        assert samples[0] == entry("S115_dry", 41713, 7)
        assert samples[1:] == setup["samples"][1:]
        assert read_setup(outdir, 41713)["samples"] == [entry("S115_dry", 41713, 7)]

    def test_run_of_earlier_sequence_updates_its_entry(self, outdir):
        reduce_run_sequence(str(outdir), 41832)
        write_scan_file(outdir, 41713, num_of_runs=7)

        reduce_run_sequence(str(outdir), 41713)

        assert [sample["num_of_scans"] for sample in read_setup(outdir)["samples"]] == [7, 6, 6]
        assert read_setup(outdir, 41713)["samples"] == [entry("S115_dry", 41713, 7)]
        assert read_setup(outdir, 41832)["samples"] == [entry("S115_pc3", 41832, 6)]

    def test_name_fixed_once_inserted(self, tmp_path):
        write_scan_file(tmp_path, 100, num_of_runs=2, titles="First")
        reduce_run_sequence(str(tmp_path), 100)
        write_scan_file(tmp_path, 100, num_of_runs=3, titles="Changed")

        reduce_run_sequence(str(tmp_path), 100)

        assert read_setup(tmp_path)["samples"] == [entry("First", 100, 3)]
        assert read_setup(tmp_path, 100)["samples"] == [entry("First", 100, 3)]

    def test_names_at_insertion(self, tmp_path, log):
        write_scan_file(tmp_path, 100, titles="S115")
        reduce_run_sequence(str(tmp_path), 100)
        write_scan_file(tmp_path, 200, titles="S115")
        reduce_run_sequence(str(tmp_path), 200)
        write_scan_file(tmp_path, 300)

        reduce_run_sequence(str(tmp_path), 300)

        assert [sample["name"] for sample in read_setup(tmp_path)["samples"]] == ["S115", "S115_2", "Sample_2"]
        assert any("S115_2" in message for message in log.records_at_warning())

    def test_name_not_taken_by_background(self, tmp_path):
        write_setup(
            tmp_path,
            {"samples": [entry("S115", 100, 1)], "background": entry("Empty", 50, 1) | {"thickness": 0.1}},
        )
        write_scan_file(tmp_path, 200, titles="Empty")

        reduce_run_sequence(str(tmp_path), 200)

        assert read_setup(tmp_path)["samples"][1]["name"] == "Empty_2"

    def test_same_run_twice_gives_same_files(self, outdir):
        reduce_run_sequence(str(outdir), 41741)
        setup, sequence_setup = (outdir / "autoconfig.json").read_text(), (outdir / "autoconfig_41741.json").read_text()

        reduce_run_sequence(str(outdir), 41741)

        assert (outdir / "autoconfig.json").read_text() == setup
        assert (outdir / "autoconfig_41741.json").read_text() == sequence_setup

    def test_first_run_recorded_later(self, tmp_path):
        write_scan_file(tmp_path, 100, runs=[101, 102], titles="S115")
        reduce_run_sequence(str(tmp_path), 100)
        assert read_setup(tmp_path)["samples"] == [entry("S115", 100, 3, exclude=[100])]
        write_scan_file(tmp_path, 100, runs=[101, 102, 100], titles="S115")

        reduce_run_sequence(str(tmp_path), 100)

        assert read_setup(tmp_path)["samples"] == [entry("S115", 100, 3)]
        assert read_setup(tmp_path, 100)["samples"] == [entry("S115", 100, 3)]

    def test_new_sequence_setup_copies_options(self, outdir):
        background = entry("Empty", 41741, 6)
        empty_cell = {"name": "Cell", "start_scan_num": 41741, "num_of_scans": 6}
        write_setup(
            outdir,
            {
                "samples": [entry("S115_dry", 41713, 6)],
                "background": background,
                "empty_cell": empty_cell,
                "save_all_harmonics": True,
            },
        )

        reduce_run_sequence(str(outdir), 41832)

        sequence_setup = read_setup(outdir, 41832)
        assert sequence_setup["samples"] == [entry("S115_pc3", 41832, 6)]
        assert sequence_setup["background"] == background
        assert sequence_setup["empty_cell"] == empty_cell | {"exclude": []}
        assert sequence_setup["save_all_harmonics"] is True

    def test_existing_sequence_setup_keeps_all_but_samples(self, outdir):
        reduce_run_sequence(str(outdir), 41832)
        sequence_setup = read_setup(outdir, 41832)
        sequence_setup["background"] = entry("Empty", 41741, 6)
        sequence_setup["save_all_harmonics"] = True
        write_setup(outdir, sequence_setup, 41832)
        write_scan_file(outdir, 41832, num_of_runs=7)

        reduce_run_sequence(str(outdir), 41832)

        assert read_setup(outdir, 41832) == sequence_setup | {"samples": [entry("S115_pc3", 41832, 7)]}

    @pytest.mark.parametrize("first_run", [None, 41713], ids=["autoconfig", "autoconfig_N"])
    @pytest.mark.parametrize("content", ["{not json", '{"samples": []}'], ids=["invalid_json", "rejected"])
    def test_unreadable_setup_file(self, outdir, reduction, log, first_run, content):
        filename = "autoconfig.json" if first_run is None else f"autoconfig_{first_run}.json"
        (outdir / filename).write_text(content)
        files_before = set(outdir.iterdir())

        with pytest.raises(ValueError):  # noqa: PT011 - the messages of json and pydantic differ
            reduce_run_sequence(str(outdir), 41713)

        assert (outdir / filename).read_text() == content
        assert set(outdir.iterdir()) == files_before
        assert reduction.reduced == []
        assert any(filename in message and "cannot be read" in message for message in log.records_at_warning())

    def test_no_temporary_file_left(self, outdir):
        reduce_run_sequence(str(outdir), 41741)
        reduce_run_sequence(str(outdir), 41741)

        assert not [path for path in outdir.iterdir() if path.suffix == ".tmp"]

    def test_sequence_not_recorded(self, outdir, reduction, log):
        files_before = set(outdir.iterdir())

        reduce_run_sequence(str(outdir), 41900)

        assert set(outdir.iterdir()) == files_before
        assert reduction.reduced == []
        assert any("41900" in message and "nothing to reduce" in message for message in log.records_at_warning())
