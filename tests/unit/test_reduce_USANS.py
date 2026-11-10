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
    reduce_sequences,
    sample_thickness,
    write_setup_file,
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
        [([-1], f"its log {FIRST_RUN_LOG} is -1"), (None, f"it has no log {FIRST_RUN_LOG}")],
        ids=["no_sequence", "no_log"],
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

    def test_saves_all_harmonics(self, outdir):
        config = build_setup(collect_sequences(str(outdir)))

        assert config.save_all_harmonics is True


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


class TestWriteSetupFile:
    def test_writes_auto_json(self, outdir):
        config = build_setup(collect_sequences(str(outdir)))

        setup_file = write_setup_file(config, str(outdir))

        assert Path(setup_file) == outdir / "auto.json"
        assert read_config(setup_file) == config
        assert read_config(setup_file).save_all_harmonics is True

    def test_writes_excluded_runs(self, tmp_path):
        write_scan_file(tmp_path, 100, runs=[100, 101, 103])
        config = build_setup(collect_sequences(str(tmp_path)))

        setup_file = write_setup_file(config, str(tmp_path))

        assert json.loads(Path(setup_file).read_text())["samples"][0]["exclude"] == [102]
        assert read_config(setup_file) == config

    def test_writes_no_csv(self, outdir):
        write_setup_file(build_setup(collect_sequences(str(outdir))), str(outdir))

        assert not (outdir / "auto.csv").exists()


class TestReduceSequences:
    def test_report_written_with_reduced_data(self, outdir, monkeypatch):
        reduced_into = []
        reports = []

        class StandInExperiment:
            """Stands in for usansred.reduce.Experiment, recording the output directory of the reduction."""

            def __init__(self, config_file: str):
                self.config_file = config_file
                self.output_dir = None

            def reduce(self, output_dir: str | None = None) -> None:
                self.output_dir = output_dir
                reduced_into.append(output_dir)

        monkeypatch.setattr(reduce_USANS.reduce, "Experiment", StandInExperiment)
        monkeypatch.setattr(reduce_USANS, "generate_report", lambda **kwargs: reports.append(kwargs))
        setup_file = write_setup_file(build_setup(collect_sequences(str(outdir))), str(outdir))

        reduce_sequences(setup_file, str(outdir))

        autodir = str(outdir / "auto")
        assert reduced_into == [autodir]
        assert reports == [{"config_file_path": setup_file, "output_dir": autodir}]
