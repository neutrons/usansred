.. measurement_types_sample

Sample
======

A *sample* is the measurement of interest. A setup file must contain at least one sample.

Setup-File Entry
----------------

In the JSON setup file, samples are listed under the required ``samples`` key:

.. code-block:: json

   {
     "samples": [
       {
         "name": "A2_56C_3hr",
         "start_scan_num": 36330,
         "num_of_scans": 5,
         "thickness": 0.1,
         "exclude": [36331, 36332]
       }
     ]
   }

- ``name``: a name for your own reference, also used in the names of the output files.
- ``start_scan_num``: the first scan (run) number of the sample.
- ``num_of_scans``: the number of consecutive scans, including the first one.
- ``thickness``: the sample thickness in cm.
- ``exclude`` (optional): scan numbers to leave out of the reduction.

The scans used are runs ``start_scan_num`` through ``start_scan_num + num_of_scans - 1``, minus
those listed in ``exclude``. In the example above, runs ``36330``, ``36333``, and ``36334`` are
used.

In the legacy CSV format, each sample is a row starting with ``s``, for example
``s,A2_56C_3hr,36330,5,0.1,36331;36332``. See :doc:`../reduce` for the full CSV description.

Subtraction
-----------

After the sample is reduced, one other measurement is subtracted from it:

- the :doc:`background <background>`, if the setup file has a ``background`` entry;
- otherwise the :doc:`empty cell <empty_cell>`, if the setup file has an ``empty_cell`` entry;
- otherwise nothing.

Subtraction is performed harmonic by harmonic: harmonic *n* of the background (or empty cell) is
subtracted from harmonic *n* of the sample, never from a different harmonic, because only
same-order curves share a comparable momentum-transfer axis.

Output Files
------------

For a sample named ``<sample>``, the following files are written to the output folder for the
first harmonic, and, when ``save_all_harmonics`` is enabled, for each higher harmonic (2 to 4) that
has data:

- ``UN_<sample>_det_<n>_unscaled.txt``: stitched, monitor-normalized detector data before scaling.
- ``UN_<sample>_det_<n>.txt``: the stitched data after scaling.
- ``UN_<sample>_det_<n>_background_subtracted.txt``: the data after subtraction of the
  background or empty cell. Only written when one of them was subtracted.

The sample is also included in ``summary.xlsx``.
