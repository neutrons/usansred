"""
Utility functions for USANS reduction.
As we add utilities, perhaps  we can split them into separater modules
"""

from mantid.simpleapi import DeleteWorkspace, SaveAscii, SumSpectra, mtd


def save_ascii(input_workspace, file_path, header=None, **saveascii):
    """Save a workspace to an ASCII file.

    Parameters
    ----------
    input_workspace : str or Workspace
        Input workspace to be summed before saving.
    file_path : str or path-like
        Destination file path for the ASCII output.
    header : str, optional
        Header text to insert or replace in the output file.
        If provided, and it does not start with ``#``, it is prefixed with ``# ``.
    **saveascii
        Additional keyword arguments passed through to ``mantid.simpleapi.SaveAscii``.
        If argument ``WriteSpectrumID`` is not supplied, it defaults to ``False``.
    """
    if "WriteSpectrumID" not in saveascii:
        saveascii["WriteSpectrumID"] = False  # spectrum number notwritten for single-spectrum workspaces
    SaveAscii(InputWorkspace=input_workspace, Filename=str(file_path), **saveascii)

    if header is None:
        return

    header_line = header if header.startswith("#") else f"# {header}"
    with open(file_path, "r", encoding="utf-8") as f:
        lines = f.readlines()
    replaced = False
    for i, line in enumerate(lines):
        if line.startswith("#"):
            lines[i] = header_line.rstrip("\n") + "\n"
            replaced = True
            break
    # Fallback: if no comment line exists, prepend one.
    if not replaced:
        lines.insert(0, header_line.rstrip("\n") + "\n")
    with open(file_path, "w", encoding="utf-8") as f:
        f.writelines(lines)


def save_summed_spectra(input_workspace, file_path, header=None):
    """Save the summed spectra of a workspace to an ASCII file.

    Parameters
    ----------
    input_workspace : str or Workspace
        Input workspace to be summed before saving.
    file_path : str or path-like
        Destination file path for the ASCII output.
    header : str, optional
        Header text to insert or replace in the output file.
        If provided, and it does not start with ``#``, it is prefixed with ``# ``.
    **saveascii
        Additional keyword arguments passed through to ``mantid.simpleapi.SaveAscii``.
        If argument ``WriteSpectrumID`` is not supplied, it defaults to ``False``.
    """
    workspace = SumSpectra(InputWorkspace=input_workspace, OutputWorkspace=mtd.unique_name())
    save_ascii(workspace, file_path, header=header)
    DeleteWorkspace(workspace)
