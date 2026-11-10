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

Order of Reduction
------------------

The measurements are reduced in the following order:

1. The empty cell, only when no background is present (see :doc:`measurement_types/empty_cell`).
2. The background, if present.
3. Each sample, in the order in which it appears in the setup file.

If the reduction of the empty cell or of the background fails, the whole reduction stops,
because the samples cannot be corrected without them. If the reduction of a sample fails, the
error is written to that sample's log file and the reduction continues with the next sample.
Each reduced measurement writes its log to ``reduction_<name>.log`` in the output folder.


Collecting Run Sequences
------------------------

A measurement is recorded as a sequence of runs, one for each segment of the analyzer scan. Two
logs of each run identify the sequence that the run belongs to, and its place in it:

- ``BL1A:CS:Scan:USANS:FirstRun``: the run number of the first run of the sequence.
- ``BL1A:CS:Scan:USANS:Index``: the segment of the scan recorded by the run, starting at 0.

The table below lists runs 50386 to 50405 of proposal IPTS-*****, with the last value of each log.

.. list-table::
   :header-rows: 1
   :widths: 15 35 15 15

   * - Run
     - Title
     - FirstRun
     - Index
   * - 50386
     - Empty Cell
     - -1
     - 3
   * - 50387
     - Empty Cell
     - 50387
     - 0
   * - 50388
     - Empty Cell
     - 50387
     - 1
   * - 50389
     - Empty Cell
     - 50387
     - 2
   * - 50390
     - Empty Cell
     - 50387
     - 3
   * - 50391
     - Empty Cell
     - 50387
     - 4
   * - 50392
     - Empty Cell
     - -1
     - 4
   * - 50393
     - Empty Air
     - -1
     - 4
   * - 50394
     - Empty Air
     - 50394
     - 0
   * - 50395
     - Empty Air
     - 50394
     - 1
   * - 50396
     - Empty Air
     - 50394
     - 2
   * - 50397
     - Empty Air
     - 50394
     - 3
   * - 50398
     - Empty Air
     - -1
     - 3
   * - 50399
     - UA1-Ca8mM
     - -1
     - 3
   * - 50400
     - UA1-Ca8mM
     - 50400
     - 0
   * - 50401
     - UA1-Ca8mM
     - 50400
     - 1
   * - 50402
     - UA1-Ca8mM
     - 50400
     - 2
   * - 50403
     - UA1-Ca8mM
     - 50400
     - 3
   * - 50404
     - UA1-Ca8mM
     - 50400
     - 4
   * - 50405
     - UA1-Ca8mM
     - -1
     - 4

Each sequence starts with a run whose ``FirstRun`` is -1, and which carry the title of the
sequence. The sequence may also finish with another ``FirstRun = -1``.
For example, runs 50386 and 50392 bracket the sequence of the empty cell, runs 50387 to
50391. A run whose ``FirstRun`` is -1 repeats the ``Index`` of the last run of the preceding
sequence: runs 50392 and 50393 both hold index 4, the index of run 50391.

The runs of a measurement are collected as follows:

- A measurement starts at the first run of its sequence, the run whose number is the value of
  ``FirstRun``, and includes every run of the sequence. In the table, the empty cell is measured by
  runs 50387 to 50391, and the empty air by runs 50394 to 50397.
- Runs whose ``FirstRun`` is -1 are not part of any measurement, and are neither preprocessed nor
  reduced. Neither are runs without the ``FirstRun`` log.

In a setup file, ``start_scan_num`` is therefore the first run of the sequence, and
``num_of_scans`` the number of runs in it. For the empty cell of the table:

.. code-block:: json

   {
     "start_scan_num": 50387,
     "num_of_scans": 5
   }
