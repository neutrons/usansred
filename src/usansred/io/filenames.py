"""filenames.py: names of the reduced data files written for a measurement."""

import os

# Prefix of the reduced data file names written by usansred up to version 1.9.0
LEGACY_PREFIX = "UN_"

# Characters a measurement name may not contain. The name is part of the file names, so a separator
# would place files outside the output directory, e.g. "../x" or "/tmp/x".
PATH_SEPARATORS = ("/", "\\")


def check_measurement_name(name: str) -> str:
    """Check that a measurement name can be part of a file name.

    Parameters
    ----------
    name : str
        Name of the measurement.

    Returns
    -------
    str
        ``name``, unchanged.

    Raises
    ------
    ValueError
        If ``name`` contains a path separator, ``/`` or ``\\``.
    """
    if any(separator in name for separator in PATH_SEPARATORS):
        raise ValueError(f"name must not contain path separators ('/' or '\\'): {name!r}")
    return name


def reduced_data_filename(name: str, suffix: str) -> str:
    """Name of one reduced data file of a measurement, ``<name>_<suffix>.txt``.

    Every reduced data file name is built here, so that the naming scheme is defined in one place.

    Parameters
    ----------
    name : str
        Name of the measurement.
    suffix : str
        Part of the name after the measurement name, without the extension, e.g. ``det_1``,
        ``det_2_unscaled``, or ``det_background_subtracted_stitched``.

    Returns
    -------
    str
        The file name, without directory.

    Raises
    ------
    ValueError
        If ``name`` contains a path separator, ``/`` or ``\\``.
    """
    return f"{check_measurement_name(name)}_{suffix}.txt"


def reduced_data_filenames(
    name: str,
    num_of_harmonics: int,
    detector_data: bool = True,
    scaled_data: bool = True,
    background_subtracted_data: bool = True,
) -> list[str]:
    """Names of the reduced data files that a reduction can write for a measurement.

    The flags select the categories of file, as in ``Sample.dump_reduced_data_to_csv``. For each
    harmonic ``n`` from 1 to ``num_of_harmonics``, in this order:

    - ``<name>_det_<n>.txt`` (``scaled_data``)
    - ``<name>_det_<n>_background_subtracted.txt`` (``background_subtracted_data``)
    - ``<name>_det_<n>_unscaled.txt`` (``detector_data``)

    followed by the profiles with all harmonics stitched together:

    - ``<name>_det_stitched.txt`` (``scaled_data``)
    - ``<name>_det_background_subtracted_stitched.txt`` (``background_subtracted_data``)

    Whether each file is actually written depends on the reduction, e.g. on ``save_all_harmonics``
    and on whether a background was subtracted.

    Parameters
    ----------
    name : str
        Name of the measurement.
    num_of_harmonics : int
        Number of harmonics (detector banks) to list files for.
    detector_data : bool
        Include the monitor-normalized files, ``_unscaled.txt``.
    scaled_data : bool
        Include the scaled files and the stitched scaled profile.
    background_subtracted_data : bool
        Include the background-subtracted files and the stitched background-subtracted profile.

    Returns
    -------
    list[str]
        The file names, without directory.

    Raises
    ------
    ValueError
        If ``name`` contains a path separator, ``/`` or ``\\``.
    """
    filenames = []
    for harmonic in range(1, num_of_harmonics + 1):
        if scaled_data:
            filenames.append(reduced_data_filename(name, f"det_{harmonic}"))
        if background_subtracted_data:
            filenames.append(reduced_data_filename(name, f"det_{harmonic}_background_subtracted"))
        if detector_data:
            filenames.append(reduced_data_filename(name, f"det_{harmonic}_unscaled"))
    if scaled_data:
        filenames.append(reduced_data_filename(name, "det_stitched"))
    if background_subtracted_data:
        filenames.append(reduced_data_filename(name, "det_background_subtracted_stitched"))
    return filenames


def legacy_bank_dirname(harmonic: int) -> str:
    """Subdirectory of the output directory that usansred 1.6.0 to 1.8.0 wrote a higher harmonic to.

    Parameters
    ----------
    harmonic : int
        The harmonic (detector bank), from 2.

    Returns
    -------
    str
        The directory name, ``bank_<harmonic>``.
    """
    return f"bank_{harmonic}"


def legacy_filenames(
    name: str,
    num_of_harmonics: int,
    detector_data: bool = True,
    scaled_data: bool = True,
    background_subtracted_data: bool = True,
) -> tuple[list[str], list[str]]:
    """Paths of the reduced data files that usansred 1.9.0 and earlier wrote for a measurement.

    All these versions start the names with ``LEGACY_PREFIX``. The paths are relative to the
    output directory, and the flags select the categories of file as in ``reduced_data_filenames``.

    The files are returned in two lists:

    - Names written by usansred 1.9.0: ``LEGACY_PREFIX`` followed by each name of
      ``reduced_data_filenames``. Versions 1.0.0 to 1.8.0 wrote the first-harmonic files under
      these names too. These are also the current names of a measurement named
      ``LEGACY_PREFIX + name``.
    - Names written only by versions 1.0.0 to 1.8.0, which no current name can equal:

      - ``UN_<name>_det_1_lb.txt`` and ``UN_<name>_det_1_lbs.txt``, the log-binned scaled data
        (``scaled_data``);
      - ``bank_<n>/UN_<name>.txt`` (``scaled_data``) and ``bank_<n>/UN_<name>_unscaled.txt``
        (``detector_data``) for each harmonic ``n`` from 2 to ``num_of_harmonics``, written by
        versions 1.6.0 to 1.8.0 with ``save_all_harmonics``.

    Parameters
    ----------
    name : str
        Name of the measurement.
    num_of_harmonics : int
        Number of harmonics (detector banks) to list files for.
    detector_data : bool
        Include the monitor-normalized files.
    scaled_data : bool
        Include the scaled files, including the log-binned ones.
    background_subtracted_data : bool
        Include the background-subtracted files.

    Returns
    -------
    tuple[list[str], list[str]]
        The names written by version 1.9.0, and the paths written only by earlier versions.

    Raises
    ------
    ValueError
        If ``name`` contains a path separator, ``/`` or ``\\``.
    """
    v190 = [
        LEGACY_PREFIX + filename
        for filename in reduced_data_filenames(
            name,
            num_of_harmonics,
            detector_data=detector_data,
            scaled_data=scaled_data,
            background_subtracted_data=background_subtracted_data,
        )
    ]
    prefixed = LEGACY_PREFIX + check_measurement_name(name)
    older = []
    if scaled_data:
        older += [f"{prefixed}_det_1_lb.txt", f"{prefixed}_det_1_lbs.txt"]
    for harmonic in range(2, num_of_harmonics + 1):
        if scaled_data:
            older.append(os.path.join(legacy_bank_dirname(harmonic), f"{prefixed}.txt"))
        if detector_data:
            older.append(os.path.join(legacy_bank_dirname(harmonic), f"{prefixed}_unscaled.txt"))
    return v190, older
