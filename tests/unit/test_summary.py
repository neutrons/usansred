import logging
import re
import zipfile

import pytest

import usansred.summary
from usansred.summary import (
    SHEET_NAME_MAX_LENGTH,
    format_sheet_name,
    generate_report,
    get_filenames_from_samples,
)


@pytest.fixture(autouse=True)
def reset_sheet_name_suffix(monkeypatch):
    """Reset the global suffix counter before each test.

    This ensures that tests are independent and do not affect each other's results."""
    monkeypatch.setattr(usansred.summary, "suffix", 0)


@pytest.mark.parametrize(
    ("input_filename", "expected_output"),
    [
        # The extension and the UN_ prefix are always removed; short names are otherwise kept
        ("UN_S115_pc3_det_1.txt", "S115_pc3_det_1"),
        ("UN_S115_pc3_det_1_unscaled.txt", "S115_pc3_det_1_unscaled"),
        ("UN_S115_pc3_det_stitched.txt", "S115_pc3_det_stitched"),
        # Rule 1 alone makes the name fit
        ("UN_S115_pc3_det_1_background_subtracted.txt", "S115_pc3_det_1_bkgr_subtracted"),
        # Rules 1 to 3; exactly 31 characters
        ("UN_S115_pc3_det_background_subtracted_stitched.txt", "S115_pc3_bkgr_subtracted_stitch"),
        # Rules 1 and 3 (rule 2 has nothing to replace)
        ("UN_A2_56C_3hr_det_1_background_subtracted.txt", "A2_56C_3hr_1_bkgr_subtracted"),
        # Rules 1 to 4
        ("UN_A2_56C_3hr_det_background_subtracted_stitched.txt", "A2_56C_3hr_bkgr_sub_stitch"),
        # Invalid characters are replaced
        ("UN_example[1]_det_1.txt", "example_1__det_1"),
        # Names without the UN_ prefix or the .txt extension are accepted
        ("normal_filename", "normal_filename"),
    ],
)
def test_format_sheet_name(input_filename, expected_output):
    """Sheet names drop ``UN_`` and ``.txt``, and long names are shortened rule by rule."""
    assert format_sheet_name(input_filename) == expected_output


def test_format_sheet_name_stops_at_first_rule_that_fits():
    """Rules are applied only until the name fits, so the later rules are not applied.

    ``S115_dry2_det_1_background_subtracted`` has 37 characters. Rule 1 brings it to exactly 31,
    so ``_det`` is kept. With one more character in the measurement name, rule 1 is not enough
    and rule 3 drops ``_det``.
    """
    assert format_sheet_name("UN_S115_dry2_det_1_background_subtracted.txt") == "S115_dry2_det_1_bkgr_subtracted"
    assert format_sheet_name("UN_S115_dry23_det_1_background_subtracted.txt") == "S115_dry23_1_bkgr_subtracted"


def test_format_sheet_name_keeps_measurement_name_intact():
    """The rules change only the part of the name from the last ``_det_`` onward.

    The measurement name below contains ``_det`` and ``background``, which must survive.
    """
    name = format_sheet_name("UN_my_detbackground_det_stitched.txt")

    assert name == "my_detbackground_det_stitched"
    long_name = format_sheet_name("UN_bg_det_x_det_background_subtracted_stitched.txt")
    assert long_name.startswith("bg_det_x_")
    assert long_name == "bg_det_x_bkgr_subtracted_stitch"


def test_format_sheet_name_falls_back_to_cut_and_counter():
    """A name still too long after all rules is cut and given a counter, ending at 31 characters.

    The cut applies to the name after all rules, ``a_very_long_measurement_name_bkgr_sub_stitch``.
    """
    name = format_sheet_name("UN_a_very_long_measurement_name_det_background_subtracted_stitched.txt")

    assert len(name) == SHEET_NAME_MAX_LENGTH
    assert name == "a_very_long_measurement_name_b1"


def test_format_sheet_name_makes_names_unique():
    """Names already used, compared without regard to case, get a counter."""
    used_names = {"original"}
    long_name = "UN_a_very_long_measurement_name_det_background_subtracted_stitched.txt"

    first = format_sheet_name(long_name, used_names)
    second = format_sheet_name(long_name, used_names)
    clash = format_sheet_name("Original.txt", used_names)

    assert first != second
    assert len(second) <= SHEET_NAME_MAX_LENGTH
    assert clash.lower() != "original"
    assert {first.lower(), second.lower(), clash.lower()} <= used_names


@pytest.mark.parametrize(
    ("sample_name", "expected_filenames"),
    [
        (
            "sample1",
            [
                "UN_sample1_det_1.txt",
                "UN_sample1_det_1_background_subtracted.txt",
                "UN_sample1_det_1_unscaled.txt",
                "UN_sample1_det_stitched.txt",
                "UN_sample1_det_background_subtracted_stitched.txt",
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


def test_generate_report_reads_from_output_dir(tmp_path, caplog, monkeypatch):
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

    monkeypatch.setattr(usansred.summary.logger, "propagate", True)

    with caplog.at_level(logging.INFO):
        generate_report(str(config), data_dir=str(data_dir), output_dir=str(output_dir))

    assert (output_dir / "summary.xlsx").exists()
    assert any("Reading sample file UN_mysample_det_1.txt" in m for m in caplog.messages), (
        f"Expected the sample file to be read from output_dir, but it was not found. Log messages: {caplog.messages}"
    )


def _workbook_xml(path) -> dict[str, str]:
    """Read the XML parts of an ``.xlsx`` workbook.

    Parameters
    ----------
    path : pathlib.Path
        The workbook.

    Returns
    -------
    dict[str, str]
        XML content keyed by part name, e.g. ``xl/workbook.xml``.
    """
    with zipfile.ZipFile(path) as archive:
        return {name: archive.read(name).decode() for name in archive.namelist() if name.endswith(".xml")}


def _chart_with_title(parts: dict[str, str], title: str) -> str:
    """Return the XML of the chart whose title is ``title``.

    Parameters
    ----------
    parts : dict[str, str]
        XML parts of the workbook, from ``_workbook_xml``.
    title : str
        Chart title, e.g. ``Background Subtracted``.

    Returns
    -------
    str
        The chart XML.
    """
    [chart] = [xml for name, xml in parts.items() if name.startswith("xl/charts/") and f"<a:t>{title}</a:t>" in xml]
    return chart


def _write_profile(path, num_points: int) -> None:
    """Write a reduced data file with ``num_points`` positive (Q, I, E) rows.

    Parameters
    ----------
    path : pathlib.Path
        File to write.
    num_points : int
        Number of rows.
    """
    rows = [f"{1e-5 * (k + 1)},{100.0 / (k + 1)},{1.0 / (k + 1)}," for k in range(num_points)]
    path.write_text("\n".join(rows) + "\n")


def test_generate_report_includes_stitched_profiles(tmp_path, monkeypatch):
    """Stitched profiles get their own sheets and appear in the summary charts.

    The stitched profile has 150 points, more than the 99 rows the charts used to read, and the
    chart ranges must reach its last row. The background-subtracted stitched profile goes to
    the "Background Subtracted" chart. Series carry the full file name, stitched series are
    drawn without a line, and the y axis of the summary charts reads ``I (1/cm)``.
    """
    monkeypatch.setattr(usansred.summary.logger, "propagate", True)
    config = tmp_path / "setup.json"
    config.write_text('{"samples": [{"name": "S115_pc3", "start_scan_num": 1, "num_of_scans": 1, "thickness": 0.1}]}')
    output_dir = tmp_path / "reduced"
    output_dir.mkdir()
    _write_profile(output_dir / "UN_S115_pc3_det_1.txt", 65)
    _write_profile(output_dir / "UN_S115_pc3_det_1_background_subtracted.txt", 65)
    _write_profile(output_dir / "UN_S115_pc3_det_stitched.txt", 150)
    _write_profile(output_dir / "UN_S115_pc3_det_background_subtracted_stitched.txt", 150)

    generate_report(str(config), output_dir=str(output_dir))

    parts = _workbook_xml(output_dir / "summary.xlsx")
    sheet_names = re.findall(r'<sheet name="([^"]*)"', parts["xl/workbook.xml"])
    assert sheet_names == [
        "Unscaled",
        "Original",
        "BG Subtracted",
        "S115_pc3_det_1",
        "S115_pc3_det_1_bkgr_subtracted",
        "S115_pc3_det_stitched",
        "S115_pc3_bkgr_subtracted_stitch",
    ]

    original = _chart_with_title(parts, "Original Data")
    assert "<c:v>UN_S115_pc3_det_stitched</c:v>" in original
    assert "S115_pc3_det_stitched!$D$2:$D$151" in original
    assert "S115_pc3_det_stitched!$E$2:$E$151" in original
    assert "S115_pc3_det_1!$D$2:$D$66" in original

    subtracted = _chart_with_title(parts, "Background Subtracted")
    assert "<c:v>UN_S115_pc3_det_background_subtracted_stitched</c:v>" in subtracted
    assert "S115_pc3_bkgr_subtracted_stitch!$E$2:$E$151" in subtracted
    assert "UN_S115_pc3_det_stitched<" not in subtracted

    # Only the stitched series have their line removed
    series = re.findall(r"<c:ser>.*?</c:ser>", original, flags=re.DOTALL)
    no_line = "<a:ln><a:noFill/></a:ln>"
    assert [no_line in ser for ser in series] == ["det_stitched" in ser for ser in series]
    assert any(no_line in ser for ser in series)

    # No unscaled file is present, so the empty "Unscaled Data" chart is not written
    for title in ("Original Data", "Background Subtracted"):
        assert "<a:t>I (1/cm)</a:t>" in _chart_with_title(parts, title)


def test_generate_report_without_stitched_profiles(tmp_path, caplog, monkeypatch):
    """Without stitched files (e.g. ``save_all_harmonics`` off), the report is still produced.

    The missing stitched files are reported at info level and skipped.
    """
    monkeypatch.setattr(usansred.summary.logger, "propagate", True)
    config = tmp_path / "setup.csv"
    config.write_text("b/s,name,start_scan,num_scans,thickness\ns,S1,1,1,1.0\n")
    output_dir = tmp_path / "reduced"
    output_dir.mkdir()
    _write_profile(output_dir / "UN_S1_det_1.txt", 10)

    with caplog.at_level(logging.INFO):
        generate_report(str(config), output_dir=str(output_dir))

    sheet_names = re.findall(r'<sheet name="([^"]*)"', _workbook_xml(output_dir / "summary.xlsx")["xl/workbook.xml"])
    assert sheet_names == ["Unscaled", "Original", "BG Subtracted", "S1_det_1"]
    assert "Sample UN_S1_det_stitched.txt file path does not exist!" in caplog.messages
    assert "Sample UN_S1_det_background_subtracted_stitched.txt file path does not exist!" in caplog.messages
