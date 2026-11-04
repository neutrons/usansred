"""filenames.py: names of the reduced data files written for a measurement."""


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

    - ``UN_<name>_det_<n>.txt`` (``scaled_data``)
    - ``UN_<name>_det_<n>_background_subtracted.txt`` (``background_subtracted_data``)
    - ``UN_<name>_det_<n>_unscaled.txt`` (``detector_data``)

    followed by the profiles with all harmonics stitched together:

    - ``UN_<name>_det_stitched.txt`` (``scaled_data``)
    - ``UN_<name>_det_background_subtracted_stitched.txt`` (``background_subtracted_data``)

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
            filenames.append(f"UN_{name}_det_{harmonic}.txt")
        if background_subtracted_data:
            filenames.append(f"UN_{name}_det_{harmonic}_background_subtracted.txt")
        if detector_data:
            filenames.append(f"UN_{name}_det_{harmonic}_unscaled.txt")
    if scaled_data:
        filenames.append(f"UN_{name}_det_stitched.txt")
    if background_subtracted_data:
        filenames.append(f"UN_{name}_det_background_subtracted_stitched.txt")
    return filenames
