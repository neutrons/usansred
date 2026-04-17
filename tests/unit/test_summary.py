import logging
import pytest

import usansred.summary
from usansred.summary import format_sheet_name, generate_report, get_filenames_from_samples


@pytest.fixture(scope="function")
def reset_sheet_name_suffix(monkeypatch):
    """Reset the global suffix counter before each test.
    
    This ensures that tests are independent and do not affect each other's results."""
    monkeypatch.setattr(usansred.summary, "suffix", 0)


@pytest.mark.parametrize(
    ("input_filename", "expected_output"),
    [
        ("example[1].txt", "example_1_.txt"),
        ("verylongfilename_exceeding_twenty_chars.txt", "verylongfilename_exc1"),
        ("normal_filename.txt", "normal_filename.txt"),
    ],
)
def test_format_sheet_name(input_filename, expected_output, reset_sheet_name_suffix):
    assert format_sheet_name(input_filename) == expected_output


@pytest.mark.parametrize(
    ("sample_name", "expected_filenames"),
    [
        (
            "sample1",
            [
                "UN_sample1_det_1.txt",
                "UN_sample1_det_1_lb.txt",
                "UN_sample1_det_1_lbs.txt",
                "UN_sample1_det_1_unscaled.txt",
            ],
        ),
        ("", ValueError),
        (None, ValueError),
    ],
)
def test_get_filenames_from_samples(sample_name, expected_filenames):
    if isinstance(expected_filenames, type) and issubclass(expected_filenames, Exception):
        with pytest.raises(expected_filenames):
            get_filenames_from_samples(sample_name)
    else:
        assert get_filenames_from_samples(sample_name) == expected_filenames


def test_generate_report_reads_from_output_dir(tmp_path, caplog):
    """Test that generate_report looks for output files in the output_dir"""
    data_dir = tmp_path / "data"
    output_dir = tmp_path / "output"
    data_dir.mkdir()
    output_dir.mkdir()

    # Write a minimal CSV config into data_dir
    config = data_dir / "config.csv"
    config.write_text("b/s,name,start_scan,num_scans,thickness\ns,mysample,1,1,1.0\n")

    # Place a reduced file in output_dir
    sample_txt = output_dir / "UN_mysample_det_1.txt"
    sample_txt.write_text("0.001,10.0,0.5\n0.002,8.0,0.4\n")

    with caplog.at_level(logging.INFO):
        generate_report(str(config), data_dir=str(data_dir), output_dir=str(output_dir))

    assert (output_dir / "summary.xlsx").exists()
    assert any("Reading sample file UN_mysample_det_1.txt" in m for m in caplog.messages), (
        "Expected the sample file to be read from output_dir, but it was not found. "
        f"Log messages: {caplog.messages}"
    )
