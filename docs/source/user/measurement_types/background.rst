.. measurement_types_background

Background
==========

A *background* is an optional measurement, for example of the solvent or buffer in the sample
cell, that is subtracted from every sample. A setup file contains at most one background.

Setup-file entry
----------------

In the JSON setup file, the background is the optional ``background`` key. It takes the same
keys as a :doc:`sample <sample>`:

.. code-block:: json

   {
     "background": {
       "name": "Empty",
       "start_scan_num": 36301,
       "num_of_scans": 5,
       "thickness": 0.1
     }
   }

- ``name``: a name for your own reference, also used in the names of the output files.
- ``start_scan_num``: the first scan (run) number of the background.
- ``num_of_scans``: the number of consecutive scans, including the first one.
- ``thickness``: the background thickness in cm.
- ``exclude`` (optional): scan numbers to leave out of the reduction.

In the legacy CSV format, the background is a row starting with ``b``, for example
``b,Empty,36301,5,0.1``. If a CSV file contains more than one ``b`` row, only the last one is
used.

Reduction
---------

The background is reduced in the same way as a sample, with its own thickness and, when an
:doc:`empty cell <empty_cell>` is present, its own transmission coefficient. It is reduced before
the samples. If its reduction fails, the whole reduction stops.

Nothing is subtracted from the background itself. In particular, the empty cell is not
subtracted from the background, because its contribution would cancel out when the background is
subtracted from the samples (see :doc:`empty_cell`).

Subtraction from the samples
----------------------------

When a background is present, it is subtracted from every sample, and it takes precedence over
the empty cell. For each harmonic:

- the background intensities are interpolated onto the momentum-transfer (Q) values of the
  sample, in ``1/angstrom``;
- the interpolated background is subtracted from the sample intensity;
- the uncertainties of the sample and of the interpolated background are added in quadrature,
  treating the two measurements as independent.

Harmonic *n* of the background is only subtracted from harmonic *n* of the sample.

Output files
------------

For a background named ``<background>``, the following files are written to the output folder for
the first harmonic, and for harmonics 2 to 4 when ``save_all_harmonics`` is enabled:

- ``UN_<background>_det_<n>_unscaled.txt``: stitched, monitor-normalized detector data before
  scaling.
- ``UN_<background>_det_<n>.txt``: the stitched data after scaling.

No ``_background_subtracted.txt`` file is written for the background, since nothing is subtracted
from it.
