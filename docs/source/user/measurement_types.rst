.. measurement_types

Measurement Types
=================

A setup file describes three types of measurement. Each type plays a different role in the
reduction:

- :doc:`Sample <measurement_types/sample>`: the measurement of interest. One or more samples
  are required.
- :doc:`Background <measurement_types/background>`: an optional measurement, for example of the
  solvent or buffer in the sample cell, that is subtracted from every sample.
- :doc:`Empty Cell <measurement_types/empty_cell>`: an optional measurement of the empty sample
  cell (or of the empty beam), used as the reference for transmission coefficients and, in the
  absence of a background, subtracted from every sample.

The table below summarizes how each type is configured and treated:

.. list-table::
   :header-rows: 1
   :widths: 30 20 20 20

   * -
     - Sample
     - Background
     - Empty Cell
   * - JSON setup-file key
     - ``samples`` (list)
     - ``background``
     - ``empty_cell``
   * - CSV row type
     - ``s``
     - ``b``
     - not supported
   * - Number allowed
     - one or more
     - zero or one
     - zero or one
   * - ``thickness`` key
     - required, in cm
     - required, in cm
     - none
   * - Transmission coefficient computed
     - yes, if an empty cell is present
     - yes, if an empty cell is present
     - no (reference, equal to 1)
   * - Subtracted from it
     - background, else empty cell
     - nothing
     - nothing
   * - Reduced output files written
     - yes
     - yes
     - no

.. toctree::
   :maxdepth: 1

   measurement_types/sample
   measurement_types/background
   measurement_types/empty_cell

Order of reduction
------------------

The measurements are reduced in the following order:

1. The empty cell, only when no background is present (see :doc:`measurement_types/empty_cell`).
2. The background, if present.
3. Each sample, in the order in which it appears in the setup file.

If the reduction of the empty cell or of the background fails, the whole reduction stops,
because the samples cannot be corrected without them. If the reduction of a sample fails, the
error is written to that sample's log file and the reduction continues with the next sample.
Each measurement writes its log to ``reduction_<name>.log`` in the output folder.
