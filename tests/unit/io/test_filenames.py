"""Tests for the names of the reduced data files."""

import pytest

from usansred.io.filenames import (
    LEGACY_PREFIX,
    check_measurement_name,
    legacy_filenames,
    reduced_data_filename,
    reduced_data_filenames,
)


def test_reduced_data_filename():
    assert reduced_data_filename("S115_pc3", "det_2_unscaled") == "S115_pc3_det_2_unscaled.txt"
    assert reduced_data_filename("S", "det_stitched") == "S_det_stitched.txt"


def test_legacy_prefix():
    """Names written by usansred up to version 1.9.0 are the current names with the prefix ``UN_``."""
    assert LEGACY_PREFIX + reduced_data_filename("S", "det_1") == "UN_S_det_1.txt"


def test_reduced_data_filenames_single_harmonic():
    filenames = reduced_data_filenames("S115_pc3", num_of_harmonics=1)

    assert filenames == [
        "S115_pc3_det_1.txt",
        "S115_pc3_det_1_background_subtracted.txt",
        "S115_pc3_det_1_unscaled.txt",
        "S115_pc3_det_stitched.txt",
        "S115_pc3_det_background_subtracted_stitched.txt",
    ]


def test_reduced_data_filenames_all_harmonics():
    filenames = reduced_data_filenames("S", num_of_harmonics=4)

    assert len(filenames) == 3 * 4 + 2
    assert len(set(filenames)) == len(filenames)
    for harmonic in range(1, 5):
        assert f"S_det_{harmonic}.txt" in filenames
        assert f"S_det_{harmonic}_background_subtracted.txt" in filenames
        assert f"S_det_{harmonic}_unscaled.txt" in filenames
    assert filenames[-2:] == ["S_det_stitched.txt", "S_det_background_subtracted_stitched.txt"]


def test_reduced_data_filenames_selected_categories():
    scaled = reduced_data_filenames("S", num_of_harmonics=2, detector_data=False, background_subtracted_data=False)
    unscaled = reduced_data_filenames("S", num_of_harmonics=2, scaled_data=False, background_subtracted_data=False)
    subtracted = reduced_data_filenames("S", num_of_harmonics=2, detector_data=False, scaled_data=False)

    assert scaled == ["S_det_1.txt", "S_det_2.txt", "S_det_stitched.txt"]
    assert unscaled == ["S_det_1_unscaled.txt", "S_det_2_unscaled.txt"]
    assert subtracted == [
        "S_det_1_background_subtracted.txt",
        "S_det_2_background_subtracted.txt",
        "S_det_background_subtracted_stitched.txt",
    ]


@pytest.mark.parametrize("name", ["../other", "/tmp/x", "a/b", "a\\b"])
def test_reduced_data_filename_rejects_path_separators(name):
    """A separator in the name would place the file outside the output directory."""
    with pytest.raises(ValueError, match="name must not contain path separators"):
        reduced_data_filename(name, "det_1")


def test_reduced_data_filenames_rejects_path_separators():
    with pytest.raises(ValueError, match="name must not contain path separators"):
        reduced_data_filenames("../other", num_of_harmonics=1)


def test_check_measurement_name_accepts_name_without_separators():
    assert check_measurement_name("UN_S1's.v2") == "UN_S1's.v2"


def test_legacy_filenames():
    """The names written by 1.9.0 are the current names with the prefix; earlier versions also wrote
    log-binned files and bank_<n>/ subdirectories."""
    v190, older = legacy_filenames("S", num_of_harmonics=3)

    assert v190 == [LEGACY_PREFIX + filename for filename in reduced_data_filenames("S", num_of_harmonics=3)]
    assert older == [
        "UN_S_det_1_lb.txt",
        "UN_S_det_1_lbs.txt",
        "bank_2/UN_S.txt",
        "bank_2/UN_S_unscaled.txt",
        "bank_3/UN_S.txt",
        "bank_3/UN_S_unscaled.txt",
    ]


def test_legacy_filenames_categories():
    v190, older = legacy_filenames("S", num_of_harmonics=2, scaled_data=False, background_subtracted_data=False)

    assert v190 == ["UN_S_det_1_unscaled.txt", "UN_S_det_2_unscaled.txt"]
    assert older == ["bank_2/UN_S_unscaled.txt"]


def test_legacy_filenames_rejects_path_separators():
    with pytest.raises(ValueError, match="name must not contain path separators"):
        legacy_filenames("../other", num_of_harmonics=1)
