.. measurement_types_empty_cell

Empty Cell / Empty Beam
=======================

An *empty-cell* run (a measurement of the sample cell without sample) and an *empty-beam* run
(a measurement with nothing in the beam) are treated identically by ``usansred``.
Both are configured with the optional ``empty_cell`` entry of the JSON setup file.
The legacy CSV format does not support an empty cell.
Note that the ``empty_cell`` entry has no ``thickness`` key because there is no sample in the beam.

.. code-block:: json

   {
     "empty_cell": {
       "name": "EmptyPCell",
       "start_scan_num": 36301,
       "num_of_scans": 5
     }
   }

**Transmission coefficients.** When an ``empty_cell`` entry is present, the transmission
coefficient of each sample (and of the background, if present) is computed from the raw event
counts as

.. math::

   T = \frac{\text{transmitted counts of the sample}}{\text{transmitted counts of the empty cell}}

and the reduced intensity is scaled by :math:`1 / (\Delta\Omega \cdot t \cdot T)`, where
:math:`\Delta\Omega` is the analyzer solid-angle acceptance and :math:`t` is the sample
thickness in cm. Without an ``empty_cell`` entry, the transmission coefficient defaults to 1.

.. warning::

   Older versions of ``usansred`` computed the transmission coefficient but did not apply it.
   For setup files that include an ``empty_cell`` entry, reduced intensities therefore change
   with respect to results obtained with older versions. Setup files without an ``empty_cell``
   entry are unaffected.

**Empty-cell subtraction.** In the absence of a ``background`` entry, the empty cell is itself
reduced (before any of the samples) and its reduced curve is subtracted from each sample, using
the same mechanism as background subtraction. When a ``background`` entry is present, the empty
cell is *not* reduced or subtracted, because the empty-cell signal cancels out in the background
subtraction:

.. math::

   (\text{sample} - \text{empty cell}) - (\text{background} - \text{empty cell})
   = \text{sample} - \text{background}

Subtracting the empty cell from both the sample and the background would double-subtract it.
The empty cell is still used to compute the transmission coefficients in that case.

**Output files.** Reduced output files (``UN_*_det_1*.txt``) are written for the samples and for
the background, but never for the empty cell. The background-subtracted file
(``UN_*_det_1_background_subtracted.txt``) is only written for measurements from which a
background or empty cell was actually subtracted. Higher harmonics are written only when
``save_all_harmonics`` is enabled.
