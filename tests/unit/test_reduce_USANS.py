"""Tests of step 2 of the preprocessing script: from the sequence files to the reduction (#107)."""

import json
import logging
import shutil
from pathlib import Path

import pytest

from usansred import reduce_USANS
from usansred.io.read import read_config
from usansred.reduce_USANS import (
    build_setup,
    collect_sequences,
    reduce_sequences,
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


@pytest.fixture
def log(monkeypatch, caplog) -> pytest.LogCaptureFixture:
    """Captures the messages logged by the preprocessing script, at level INFO and above."""
    # The logger of the module does not propagate to the root logger, where caplog listens
    monkeypatch.setattr(reduce_USANS.logger, "propagate", True)
    caplog.set_level(logging.INFO)
    return caplog


def write_scan_file(outdir: Path, first_run: int, num_of_runs: int = 1, runs: list[int] | None = None) -> None:
    """Write a scan_<N>.json file for a sequence starting at ``first_run``.

    The sequence holds ``runs``, in the order given, or else ``num_of_runs`` consecutive runs.
    """
    if runs is None:
        runs = list(range(first_run, first_run + num_of_runs))
    entries = {str(run): {"iq": f"USANS_{run}_detector_3.6.txt"} for run in runs}
    (outdir / f"scan_{first_run}.json").write_text(json.dumps(entries))


def write_sample_file(outdir: Path, first_run: int, **sample_info) -> None:
    """Change the values of the sample_<N>.json file of a fixture sequence."""
    sample_file = outdir / f"sample_{first_run}.json"
    sample_file.write_text(json.dumps(json.loads(sample_file.read_text()) | sample_info))


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

        assert [sample.name for sample in config.samples] == ["Sample0", "Sample1", "Sample2"]
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

    @pytest.mark.xfail(strict=True, reason="#107: the sequences are named after the run titles in Stage 5")
    def test_names_from_sample_files(self, outdir):
        config = build_setup(collect_sequences(str(outdir)))

        # The titles are written to the setup file with some changes (Align: removed), so only
        # the part after the prefix is compared
        for sample, title in zip(config.samples, TITLES.values()):
            assert title.removeprefix("Align:") in sample.name


class TestWriteSetupFile:
    def test_writes_auto_json(self, outdir):
        config = build_setup(collect_sequences(str(outdir)))

        setup_file = write_setup_file(config, str(outdir))

        assert Path(setup_file) == outdir / "auto.json"
        assert read_config(setup_file) == config

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
