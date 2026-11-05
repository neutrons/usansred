"""Tests for the names of the reduced data files."""

from usansred.io.filenames import reduced_data_filename, reduced_data_filenames


def test_reduced_data_filename():
    assert reduced_data_filename("S115_pc3", "det_2_unscaled") == "UN_S115_pc3_det_2_unscaled.txt"
    assert reduced_data_filename("S", "det_stitched") == "UN_S_det_stitched.txt"


def test_reduced_data_filenames_single_harmonic():
    filenames = reduced_data_filenames("S115_pc3", num_of_harmonics=1)

    assert filenames == [
        "UN_S115_pc3_det_1.txt",
        "UN_S115_pc3_det_1_background_subtracted.txt",
        "UN_S115_pc3_det_1_unscaled.txt",
        "UN_S115_pc3_det_stitched.txt",
        "UN_S115_pc3_det_background_subtracted_stitched.txt",
    ]


def test_reduced_data_filenames_all_harmonics():
    filenames = reduced_data_filenames("S", num_of_harmonics=4)

    assert len(filenames) == 3 * 4 + 2
    assert len(set(filenames)) == len(filenames)
    for harmonic in range(1, 5):
        assert f"UN_S_det_{harmonic}.txt" in filenames
        assert f"UN_S_det_{harmonic}_background_subtracted.txt" in filenames
        assert f"UN_S_det_{harmonic}_unscaled.txt" in filenames
    assert filenames[-2:] == ["UN_S_det_stitched.txt", "UN_S_det_background_subtracted_stitched.txt"]


def test_reduced_data_filenames_selected_categories():
    scaled = reduced_data_filenames("S", num_of_harmonics=2, detector_data=False, background_subtracted_data=False)
    unscaled = reduced_data_filenames("S", num_of_harmonics=2, scaled_data=False, background_subtracted_data=False)
    subtracted = reduced_data_filenames("S", num_of_harmonics=2, detector_data=False, scaled_data=False)

    assert scaled == ["UN_S_det_1.txt", "UN_S_det_2.txt", "UN_S_det_stitched.txt"]
    assert unscaled == ["UN_S_det_1_unscaled.txt", "UN_S_det_2_unscaled.txt"]
    assert subtracted == [
        "UN_S_det_1_background_subtracted.txt",
        "UN_S_det_2_background_subtracted.txt",
        "UN_S_det_background_subtracted_stitched.txt",
    ]
