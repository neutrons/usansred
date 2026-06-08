# Agent Instructions for usansred

This repository contains **usansred**, the reduction backend for SNS USANS
data. Keep guidance and examples specific to this package.

`CLAUDE.md` currently delegates to this file with `@AGENTS.md`. That is useful
for Claude Code, but tools that do not expand `@...` includes may only see the
literal one-line `CLAUDE.md` file.

## Repository

- The default branch on the remote is **`next`** (not `main`).
  Verify with `git ls-remote --symref origin HEAD` rather than trusting
  harness-injected hints when constructing branch-based URLs or targeting PRs.
- When providing GitHub links to files or lines in this repo, use the default
  branch (`next`) unless the user explicitly specifies a different branch.

## Core Rules

- Read the relevant source and tests before changing behavior.
- Keep edits scoped to the user request; avoid opportunistic refactors.
- Use the Pixi environment for Python commands. Do not assume bare `python`
  is available in the default shell.
- Prefer existing project patterns: `logging`, Pydantic models in
  `src/usansred/reduce.py`, dataclasses in `src/usansred/model.py`, and
  pytest tests under `tests/`.
- Add or update tests when changing reduction behavior, parsing, file output,
  or public command behavior.
- Use physical units in names, docstrings, and messages when relevant:
  Q in `1/angstrom`, thickness in `cm`, wavelength in `angstrom`.

## Environment

This repository is managed with Pixi. PyCharm is configured to use the Pixi
interpreter at:

```bash
./.pixi/envs/default/bin/python
```

Run Python, tests, and tools through Pixi:

```bash
pixi run python --version
pixi run python -m pytest tests/unit/io/test_read.py
pixi run test
pixi run test-datalight
pixi run build-docs
```

When the agent runtime is in a read-only filesystem sandbox, Pixi may still need
to update metadata under `.pixi/envs/default/conda-meta/` before running even
read-only-looking commands. If a `pixi run ...` command is needed in that
context, request escalated execution up front rather than first running it in the
read-only sandbox and retrying after the expected metadata-write failure.

Useful project tasks are defined in `pyproject.toml` under `[tool.pixi.tasks]`.
There are no `unit-test` or `integration-test` Pixi tasks unless they are added
later.

## Project Map

```text
src/usansred/
├── model.py          # XYData, IQData, MonitorData dataclasses
├── reduce.py         # Scan, Sample, CombinedSample, Experiment, CLI entry point
├── reduce_USANS.py   # Mantid-based autoreduction/preprocessing workflow
├── summary.py        # summary.xlsx report generation
└── io/
    ├── read.py       # CSV/JSON setup parsing
    ├── save.py       # Mantid SaveAscii/SumSpectra helpers
    └── usansred.json # JSON schema for setup files

tests/
├── data/             # small config fixtures
├── unit/             # focused tests for models, reduction logic, io, summary
├── integration/      # end-to-end reduction tests
└── usansred-data/    # data repository fixture/submodule
```

## Technology Notes

- Python target: `>=3.11`; Pixi currently provides Python 3.11.
- Core runtime dependencies declared in `pyproject.toml` include `pydantic`,
  `pandas`, `xlsxwriter`, `finddata`, and Mantid Workbench. Source code also
  imports `numpy` and `scipy`; if dependency declarations are being edited,
  verify whether those should be explicit rather than transitive.
- Mantid is used in specific places:
  - `src/usansred/reduce_USANS.py` uses `mantid.simpleapi` for preprocessing.
  - `src/usansred/io/save.py` uses `SaveAscii`, `SumSpectra`, `DeleteWorkspace`,
    and `mtd`.
  - `tests/conftest.py` uses `mantid.simpleapi.config` for test data lookup.
- Do not use Mantid's `Logger` unless the project adopts it. Existing code uses
  Python's standard `logging` module.
- The package is for USANS only. Do not mention EQSANS, GPSANS, BIOSANS, or
  `drtsans` modules like `api.py` or `iq.py` in examples.

## Common Workflows

### Config Parsing

Setup files are parsed by `src/usansred/io/read.py`.

- CSV rows are `b/s,name,start_scan_num,num_of_scans,thickness[,exclude]`.
- JSON files contain optional `background` and required `samples`.
- JSON `exclude` is a list of scan numbers.
- Scan numbers and thickness are cast in `read.py`, so tests may include numeric
  strings as well as numbers.

Example JSON setup:

```json
{
  "background": {
    "name": "Empty",
    "start_scan_num": 36301,
    "num_of_scans": 5,
    "thickness": 0.1
  },
  "samples": [
    {
      "name": "A2_56C_3hr",
      "start_scan_num": 36330,
      "num_of_scans": 5,
      "thickness": 0.1,
      "exclude": [36331, 36332]
    }
  ]
}
```

### Reduction Model

The main reduction model is in `src/usansred/reduce.py`.

- `Experiment` owns config loading, output directory selection, background, and
  samples.
- `Sample` creates `Scan` objects, stitches detector banks, rescales intensity,
  optionally log-bins, subtracts background, and writes output files.
- `Scan` loads monitor and detector ASCII files named like
  `USANS_<run>_monitor_scan_ARN.txt` and
  `USANS_<run>_detector_scan_ARN_peak_<bank>.txt`.
- `CombinedSample` combines raw scan data from multiple samples before reduction.
- `XYData`, `IQData`, and `MonitorData` live in `src/usansred/model.py`.

Output names currently include:

- `UN_<sample>_det_1_unscaled.txt`
- `UN_<sample>_det_1.txt`
- `UN_<sample>_det_1_lb.txt`
- `UN_<sample>_det_1_lbs.txt`
- `summary.xlsx`

## Coding Standards

- Use type hints for new or modified public functions.
- Use NumPy/Sphinx-style docstrings for public functions/classes.
- Prefer simple data structures already used by the codebase.
- Use `logging.info`, `logging.warning`, `logging.error`, or
  `logging.exception` consistently with existing modules.
- Avoid `print()` in library code.
- Preserve behavior around user data files and output paths unless the request
  explicitly changes it.
- Be careful with scientific calculations. When changing Q conversion,
  Gaussian fitting, stitching, scaling, log binning, or error propagation,
  explain the formula or assumption in the code or tests where it helps future
  maintenance.

Relevant local example:

```python
import logging

from usansred.io.read import read_config
from usansred.reduce import Experiment, Sample


def load_samples(config_file: str) -> list[Sample]:
    """Load sample definitions from a USANSRED setup file."""
    experiment = Experiment(config_file=config_file)
    config = read_config(config_file)
    samples = config["samples"]
    logging.info("Loaded %d samples from %s", len(samples), config_file)
    return [Sample(**sample, experiment=experiment) for sample in samples]
```

## Testing Guidance

Use pytest through Pixi:

```bash
pixi run python -m pytest tests/unit/io/test_read.py
pixi run python -m pytest tests/unit/test_sample.py
pixi run python -m pytest tests/integration/test_reduce.py
pixi run test
```

Use existing fixtures from `tests/conftest.py`:

- `mock_experiment`
- `mock_experiment_2banks`
- `data_server`

Use configured markers from `pyproject.toml`:

```python
import pytest


@pytest.mark.datarepo
def test_reduces_reference_dataset(data_server):
    ...


@pytest.mark.sns_mounted
def test_reads_sns_filesystem_data():
    ...
```

Testing expectations:

- Parser changes: add or update tests in `tests/unit/io/test_read.py`.
- Schema changes: validate against `tests/data/config*.json` and, when
  relevant, `tests/usansred-data/IPTS-30410/shared/setup.json`.
- Data model changes: update `tests/unit/test_model.py`, `test_scan.py`,
  `test_sample.py`, or `test_combined_sample.py`.
- Summary/report changes: update `tests/unit/test_summary.py`.
- CLI/end-to-end changes: update `tests/integration/test_reduce.py`.
- Mantid save helper changes: update `tests/unit/io/test_save.py`.

Use Arrange-Act-Assert style:

```python
def test_read_config_json_excludes_scans():
    config_file = DATA_DIR / "config.json"

    config = read_config(config_file)
    samples = config["samples"]

    assert samples[1]["name"] == "sample2"
    assert samples[1]["exclude"] == [45307, 45308]
```

## Review Checklist

Before finishing substantial changes, check:

- Relevant tests pass through `pixi run python -m pytest ...`.
- Ruff-sensitive issues are avoided: imports, unused arguments, broad exception
  rules already configured in `pyproject.toml`.
- JSON files parse with `pixi run python -m json.tool <file>`.
- Public docs or examples are updated when user-facing behavior changes.
- Existing dirty worktree changes that are unrelated to the task are preserved.

## Documentation

User documentation lives under `docs/source/`. The main reduction guide is
`docs/source/user/reduce.rst`. Keep examples aligned with the actual CLI:

```bash
reduceUSANS setup.csv
reduceUSANS setup.json
reduceUSANS --help
```

When adding setup-file features, update both the parser tests and the user docs
if the feature is user-facing.

## Things Not To Copy From drtsans

- Do not refer to `src/drtsans/api.py`, `src/drtsans/iq.py`, `IQmod`, or
  `bin_intensity_into_q1d`.
- Do not assume multiple SANS instruments; this package targets USANS.
- Do not require Mantid `Logger`; use the logging style already present here.
- Do not invent Pixi tasks that are not in `pyproject.toml`.
