"""Unit tests for usansred.io module.

Note: We need to patch the Experiment and Sample model_post_init methods
to avoid issues with missing attributes during testing.
"""

import json
from importlib import resources
from pathlib import Path
from unittest.mock import patch

import pytest
from jsonschema import Draft202012Validator

from usansred.io.read import read_config
from usansred.reduce import Experiment, Sample
from usansred.utils import cast_to_bool

DATA_DIR = Path(__file__).parent.parent.parent / "data"


def _write_json_config(tmp_path: Path, config: dict) -> Path:
    config_file = tmp_path / "setup.json"
    config_file.write_text(json.dumps(config), encoding="utf-8")
    return config_file


@pytest.mark.parametrize(
    ("value", "expected"),
    [
        (True, True),
        (False, False),
        (1, True),
        (0, False),
        ("1", True),
        ("0", False),
        ("True", True),
        ("False", False),
        ("", False),
    ],
)
def test_cast_to_bool(value, expected):
    """Test casting common setup-file boolean values."""
    assert cast_to_bool(value) is expected


def test_read_config_csv():
    """Test reading configuration from a CSV file."""

    csv_file = DATA_DIR / "config.csv"
    config = read_config(csv_file)
    background = config.background
    samples = config.samples

    with patch.object(Experiment, "model_post_init", return_value=None):
        experiment = Experiment(config_file="dummy.csv")

    with patch.object(Sample, "model_post_init", return_value=None):
        background = Sample(**background.model_dump(), experiment=experiment) if background else None
        samples = [Sample(**s.model_dump(), experiment=experiment) for s in samples]

    assert background is not None
    assert len(samples) == 2
    assert samples[0].name == "sample1"
    assert samples[1].name == "sample2"
    assert samples[1].exclude == [12345, 67890]


def test_read_config_csv_applies_schema_defaults(tmp_path):
    """Schema defaults (save_all_harmonics, binning, exclude) are injected for CSV configs."""

    csv_file = tmp_path / "setup.csv"
    csv_file.write_text("s,sample1,45306,6,0.1\n", encoding="utf-8")

    config = read_config(csv_file)

    assert config.save_all_harmonics is False
    assert config.binning.log_binning is False
    assert config.binning.steps_per_decade == 33
    assert config.samples[0].exclude == []


def test_read_config_json():
    """Test reading configuration from a JSON file."""

    json_file = DATA_DIR / "config.json"
    config = read_config(json_file)
    background = config.background
    samples = config.samples

    with patch.object(Experiment, "model_post_init", return_value=None):
        experiment = Experiment(config_file="dummy.json")

    with patch.object(Sample, "model_post_init", return_value=None):
        background = Sample(**background.model_dump(), experiment=experiment) if background else None
        samples = [Sample(**s.model_dump(), experiment=experiment) for s in samples]

    assert background is not None
    assert background.name == "example_background"
    assert background.exclude == [45336]
    assert len(samples) == 2
    assert samples[0].name == "sample1"
    assert samples[1].name == "sample2"
    assert samples[1].exclude == [45307, 45308]


def test_read_config_json_applies_schema_defaults(tmp_path):
    """Test that schema defaults are inserted before JSON configuration parsing."""

    json_file = _write_json_config(
        tmp_path,
        {
            "samples": [
                {
                    "name": "sample1",
                    "start_scan_num": "45306",
                    "num_of_scans": 6,
                    "thickness": 0.1,
                }
            ]
        },
    )

    config = read_config(json_file)

    assert config.save_all_harmonics is False
    assert config.binning.log_binning is False
    assert config.binning.steps_per_decade == 33
    assert config.samples[0].exclude == []


def test_usansred_schema_treats_save_all_harmonics_as_optional():
    """The bundled schema validates configs that omit save_all_harmonics."""

    schema_path = resources.files("usansred.io").joinpath("usansred.json")
    with schema_path.open("r", encoding="utf-8") as schema_file:
        schema = json.load(schema_file)

    config = {
        "samples": [
            {
                "name": "sample1",
                "start_scan_num": "45306",
                "num_of_scans": 6,
                "thickness": 0.1,
            }
        ]
    }

    Draft202012Validator(schema).validate(config)


def test_read_config_json_missing_samples_raises(tmp_path):
    """Test that missing required properties without defaults raise an exception."""

    json_file = _write_json_config(
        tmp_path,
        {
            "background": {
                "name": "example_background",
                "start_scan_num": "45335",
                "num_of_scans": 6,
                "thickness": 0.1,
            }
        },
    )

    with pytest.raises(ValueError, match="Field required"):
        read_config(json_file)


def test_read_config_json_missing_sample_required_property_raises(tmp_path):
    """Test that missing required sample properties without defaults raise an exception."""

    json_file = _write_json_config(
        tmp_path,
        {
            "samples": [
                {
                    "start_scan_num": "45306",
                    "num_of_scans": 6,
                    "thickness": 0.1,
                }
            ]
        },
    )

    with pytest.raises(ValueError, match="Field required"):
        read_config(json_file)


def test_read_config_json_unexpected_property_raises(tmp_path):
    """Test that schema validation rejects unexpected JSON properties."""

    json_file = _write_json_config(
        tmp_path,
        {
            "samples": [
                {
                    "name": "sample1",
                    "start_scan_num": "45306",
                    "num_of_scans": 6,
                    "thickness": 0.1,
                    "unexpected": True,
                }
            ]
        },
    )

    with pytest.raises(ValueError, match="Extra inputs are not permitted"):
        read_config(json_file)


def test_read_config_json_no_background():
    """Test reading configuration from a JSON file with no background sample."""

    json_file = DATA_DIR / "config-no-bg.json"
    config = read_config(json_file)
    background = config.background
    samples = config.samples

    with patch.object(Experiment, "model_post_init", return_value=None):
        experiment = Experiment(config_file="dummy.json")

    with patch.object(Sample, "model_post_init", return_value=None):
        background = Sample(**background.model_dump(), experiment=experiment) if background else None
        samples = [Sample(**s.model_dump(), experiment=experiment) for s in samples]

    assert background is None
    assert len(samples) == 2
    assert samples[0].name == "sample1"
    assert samples[1].name == "sample2"
    assert samples[1].exclude == [45307, 45308]
