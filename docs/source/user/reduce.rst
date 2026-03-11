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

| The setup file is a file that contains the information about the samples to be reduced.
| Two formats are supported: CSV (comma separated values) and JSON.

The columns for the CSV format are as follows:

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

The JSON format provides the same information in a different layout. A JSON setup file contains a background entry, and a samples entry,which is a list of sample objects.
Each background and sample object contains the same information as the columns in the CSV format, but with descriptive keys:

- ``name`` (string)
- ``start_scan_num`` (integer)
- ``num_of_scans`` (integer)
- ``thickness`` (float)
- ``exclude`` (list of integers, optional)

Note that the main difference is how excluded scans are represented:
- in CSV, they are represented as a semicolon-separated string of scan numbers in the last (6th) column,
- in JSON, they are represented as a list of integers under the key ``exclude``.

For example, create a file named ``setup.json`` with the following content:

.. code-block:: json

   {
     "background": {
       "name": "Empty",
       "start_scan_num": 36301,
       "num_of_scans": 5,
       "thickness": 0.1
     },
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
     ]
   }

Reducing the Data
-----------------

Run the reducing script by passing the path to the JSON or CSV setup file.

.. code-block:: bash

   (usansred) $ reduceUSANS setup.csv
   # or
   (usansred) $ reduceUSANS setup.json

Additional CLI options for ``reduceUSANS`` can be viewed in the terminal by running:

.. code-block:: bash

   (usansred) $ reduceUSANS --help

Summary
-------

Once reduction is finished, subdirectory ``result/`` is created containing the following files:

- ``summary.xlsx`` containing sketchy plots of the data for a quick review.
- Reduced data files. For example:

  + ``UN_X5D2_8_det_1.txt`` (**.txt**) is the stitched data (scaled).
  + ``UN_X5D2_8_det_1_lb.txt`` (**_lb.txt**) is the data after log binning.
  + ``UN_X5D2_8_det_1_lbs.txt`` (**_lbs.txt**) is the log binned data after background subtraction.
