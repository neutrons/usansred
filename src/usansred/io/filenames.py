"""filenames.py: names of the reduced data files written for a measurement."""

# Prefix of the reduced data file names written by usansred up to version 1.1.0
LEGACY_PREFIX = "UN_"


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
    """
    return f"{name}_{suffix}.txt"


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
