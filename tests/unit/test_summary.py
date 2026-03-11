import pytest

from usansred.summary import format_sheet_name, get_filenames_from_samples


@pytest.mark.parametrize(
    ("input_filename", "expected_output"),
    [
        ("example[1].txt", "example_1_.txt"),
        ("verylongfilename_exceeding_twenty_chars.txt", "verylongfilename_exc1"),
        ("normal_filename.txt", "normal_filename.txt"),
    ],
)
def test_format_sheet_name(input_filename, expected_output):
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
