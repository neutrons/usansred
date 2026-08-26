.. using_reduce_script


Reducing One (or more) Experiments
==================================

After the experiment, the raw data should be stored in your experiment folder:

.. code-block:: bash

   $ cd /SNS/USANS/IPTS-XXXXXX/shared/autoreduce/
   $ usansred


Activating the Environment
--------------------------

| In order to reduce the data, you need to activate the ``usansred`` pixi environment first.
| Instructions for activating the environment can be found in :doc:`Environments <environments>`.
| This will allow you to run the ``reduceUSANS`` command in the terminal, which is the main script for reducing USANS data.

| Alternatively, if you want to open a Python interpreter with the ``usansred`` environment activated,
| you can simple run ``usansred`` in the terminal, which will automatically activate the environment and open a Python interpreter for you.

``usansred_qa`` and ``usansred_dev`` environments are also available, but they are intended for QA and development purposes, respectively.
For regular data reduction, simply running ``usansred`` is sufficient.

Defining the Setup File
-----------------------

First, create the setup file **in the same folder as the raw data**.

The setup file is a file that contains the information about the samples to be reduced.
Two formats are supported: JSON and CSV (comma separated values).
The CSV is supported for backward compatibility and it only supports the background and sample information,
while the JSON format also supports additional configuration flags such as `save_all_harmonics`.

The JSON format provides the same information in a structured layout.
A JSON setup file contains a required samples entry, optional background entry, and optional configuration flags.
Each background and sample object contains descriptive keys for each field.

.. code-block:: javascript

   {
     "samples": [
       {
         "name": "<string>",                    // required; sample name
         "start_scan_num": "<integer|string>",  // required; run or scan number
         "num_of_scans": "<integer|string>",    // required; number of scans
         "thickness": "<number|string>",        // required; thickness in cm
         "exclude": ["<integer|string>"]        // scan numbers to skip during reduction; default: []
       }
     ],

     "background": {
       "name": "<string>",                    // required if background is present; background name
       "start_scan_num": "<integer|string>",  // required if background is present; run or scan number
       "num_of_scans": "<integer|string>",    // required if background is present; number of scans
       "thickness": "<number|string>",        // required if background is present; thickness in cm
       "exclude": ["<integer|string>"]        // scan numbers to skip during reduction; default: []
     },
     "empty_cell": {
       "name": "<string>",                    // required if empty cell is present; empty cell name
       "start_scan_num": "<integer|string>",  // required if empty cell is present; run or scan number
       "num_of_scans": "<integer|string>",    // required if empty cell is present; number of scans
       "exclude": ["<integer|string>"]        // scan numbers to skip during reduction; default: []
     },
     "save_all_harmonics": "<boolean>"         // optional; save reduced data for higher harmonics; default: false
   }

For example, create a file named ``setup.json`` with the following content:

.. code-block:: json

   {
     "samples": [
       {
         "name": "A2_50C_3hr",
         "start_scan_num": 36308,
         "num_of_scans": 5,
         "thickness": 0.1
       },
       {
         "name": "A2_52C_3hr",
         "start_scan_num": 36316,
         "num_of_scans": 5,
         "thickness": 0.1
       },
       {
         "name": "A2_54C_3hr",
         "start_scan_num": 36323,
         "num_of_scans": 5,
         "thickness": 0.1
       },
       {
         "name": "A2_56C_3hr",
         "start_scan_num": 36330,
         "num_of_scans": 5,
         "thickness": 0.1,
         "exclude": [36331, 36332]
       }
     ],
     "background": {
       "name": "Empty",
       "start_scan_num": 36301,
       "num_of_scans": 5,
       "thickness": 0.1
     },
     "save_all_harmonics": false
   }

Empty Cell / Empty Beam
-----------------------

An *empty-cell* run (a measurement of the sample cell without sample) and an *empty-beam* run
(a measurement with nothing in the beam) are treated identically by ``usansred``.
Both are configured with the optional ``empty_cell`` entry of the JSON setup file.
Note that the ``empty_cell`` entry has no ``thickness`` key:
there is no sample in the beam, so an effective thickness of 1 cm is assumed internally.

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

JSON Schema
-----------

Your reduction ``config.json`` must conform to the schema defined by the ``ReductionConfig`` Pydantic model.
Below is the content of the generated JSON schema, which serves as a reference for the expected structure, data types, and value constraints of the setup file:

.. literalinclude:: ../../../src/usansred/io/usansred.json
   :language: json

CSV Format (Legacy)
-------------------

The old CSV format provides only part of the information that can be encoded in the JSON file.
Information is entered in rows, with items in a row separated by `,`:

1. Sample type: either `b` for background (empty sample) or `s` for sample.
2. Sample name: a name for your own reference.
3. Starting scan number: the first scan number associated with this sample.
4. Number of scans: the total number of scans associated with this sample, including the first one.
   For instance: ``36308,5`` instructs ``reduceUSANS`` to reduce together runs ``36308``, ``36309``, ``36310``,  ``36311``, and ``36312``.
5. Sample thickness: the thickness of the sample in centimeters.
6. (Optional) Exclude scans: a list of scan numbers to be excluded from the reduction, separated by semicolons.
   For example, ``36308;36310`` will exclude scans ``36308`` and ``36310`` from the reduction.

An example ``setup.csv`` might look like:

.. code-block:: bash

   b,Empty,36301,5,0.1
   s,A2_50C_3hr,36308,5,0.1
   s,A2_52C_3hr,36316,5,0.1
   s,A2_54C_3hr,36323,5,0.1
   s,A2_56C_3hr,36330,5,0.1,36331;36332

Note that the main difference is how excluded scans are represented:
- in JSON, they are represented as a list of integers under the key ``exclude``,
- in CSV, they are represented as a semicolon-separated string of scan numbers in the last (6th) column.

Reducing the Data
-----------------

Run the reducing script by passing the path to the JSON or CSV setup file.

.. code-block:: bash

   (usansred) $ reduceUSANS setup.json
   # or
   (usansred) $ reduceUSANS setup.csv

Additional CLI options for ``reduceUSANS`` can be viewed in the terminal by running:

.. code-block:: bash

   (usansred) $ reduceUSANS --help
   usage: reduceUSANS [-h] [-o OUTPUT] path

   USANS Data Reduction

   positional arguments:
     path                         Path to the configuration file

   options:
     -h, --help                   show this help message and exit
     -o OUTPUT, --output OUTPUT   Output folder for reduced data (default: current folder)

Tab completion for ``reduceUSANS`` is registered automatically when entering
the Pixi environment in Bash or Zsh:

.. code-block:: bash

   $ pixi shell
   (usansred) $ reduceUSANS --out<TAB>

If completion does not appear after updating ``usansred``, exit and re-enter
the Pixi environment.

Summary
-------

Once reduction is finished, subdirectory ``result/`` is created containing the following files:

- ``summary.xlsx`` containing sketchy plots of the data for a quick review.
- Reduced data files. For example:

  + ``UN_X5D2_8_det_1_unscaled.txt`` (**_unscaled.txt**) is the stitched,
    monitor-normalized detector data before scaling. It is written when detector
    data is present.
  + ``UN_X5D2_8_det_1.txt`` (**.txt**) is the stitched data (scaled).
  + ``UN_X5D2_8_det_1_background_subtracted.txt`` (**_background_subtracted.txt**)
    is the data after background (or empty-cell) subtraction.
    It is only written when a background or empty cell was actually subtracted.

The ``_det_1`` infix denotes the first harmonic (first detector bank), the only one written
by default. When ``save_all_harmonics`` is enabled in the JSON setup file, all three
categories of file are written for every harmonic, using the same names with the harmonic
number in place of the ``1``:

.. code-block:: text

   UN_X5D2_8_det_1_unscaled.txt   UN_X5D2_8_det_1.txt   UN_X5D2_8_det_1_background_subtracted.txt
   UN_X5D2_8_det_2_unscaled.txt   UN_X5D2_8_det_2.txt   UN_X5D2_8_det_2_background_subtracted.txt
   UN_X5D2_8_det_3_unscaled.txt   UN_X5D2_8_det_3.txt   UN_X5D2_8_det_3_background_subtracted.txt
   UN_X5D2_8_det_4_unscaled.txt   UN_X5D2_8_det_4.txt   UN_X5D2_8_det_4_background_subtracted.txt

Background subtraction is performed harmonic by harmonic: harmonic *n* of the background (or
empty cell) is subtracted from harmonic *n* of the sample, never from a different harmonic,
because only same-order curves share a comparable momentum-transfer axis.

The ``summary.xlsx`` report covers the first-harmonic files only.
