"""Unit tests for usansred.io module.

Note: We need to patch the Experiment and Sample model_post_init methods
to avoid issues with missing attributes during testing.
"""

from pathlib import Path
from unittest.mock import patch

from usansred.io.read import read_config
from usansred.reduce import Experiment, Sample

DATA_DIR = Path(__file__).parent.parent.parent / "data"


def test_read_config_csv():
    """Test reading configuration from a CSV file."""

    csv_file = DATA_DIR / "example-config.csv"
    background, samples = read_config(csv_file)

    with patch.object(Experiment, "model_post_init", return_value=None):
        experiment = Experiment(config="dummy.csv")

    with patch.object(Sample, "model_post_init", return_value=None):
        background = Sample(**background, experiment=experiment) if background else None
        samples = [Sample(**s, experiment=experiment) for s in samples]

    assert background is not None
    assert len(samples) == 2
    assert samples[0].name == "sample1"
    assert samples[1].name == "sample2"
    assert samples[1].exclude == [12345, 67890]


def test_read_config_json():
    """Test reading configuration from a JSON file."""

    json_file = DATA_DIR / "example-config.json"
    background, samples = read_config(json_file)

    with patch.object(Experiment, "model_post_init", return_value=None):
        experiment = Experiment(config="dummy.json")

    with patch.object(Sample, "model_post_init", return_value=None):
        background = Sample(**background, experiment=experiment) if background else None
        samples = [Sample(**s, experiment=experiment) for s in samples]

    assert background is not None
    assert background.name == "example_background"
    assert len(samples) == 2
    assert samples[0].name == "sample1"
    assert samples[1].name == "sample2"
    assert samples[1].exclude == [45307, 45308]


def test_read_config_json_no_background():
    """Test reading configuration from a JSON file with no background sample."""

    json_file = DATA_DIR / "example-config-no-bg.json"
    background, samples = read_config(json_file)

    with patch.object(Experiment, "model_post_init", return_value=None):
        experiment = Experiment(config="dummy.json")

    with patch.object(Sample, "model_post_init", return_value=None):
        background = Sample(**background, experiment=experiment) if background else None
        samples = [Sample(**s, experiment=experiment) for s in samples]

    assert background is None
    assert len(samples) == 2
    assert samples[0].name == "sample1"
    assert samples[1].name == "sample2"
    assert samples[1].exclude == [45307, 45308]
