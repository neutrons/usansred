.. using_reduce_script

Reducing One (or more) Experiments
==================================
After the experiment, the raw data should be stored in your experiment folder:

.. code-block:: bash

   $> cd /SNS/USANS/IPTS-XXXXXX/shared/autoreduce/
   $> usansred

Command `usansred` activates the `usansred` conda environment,
making available the reducing script `reduceUSANS` as a terminal command.

Now create the setup file **in the same folder as the raw data**.
The setup file is a text file in Comman Separated Values (CSV) format. An example of setup file:

.. code-block:: bash

   b,Empty,36301,5,0.1
   s,A2_50C_3hr,36308,5,0.1
   s,A2_52C_3hr,36316,5,0.1
   s,A2_54C_3hr,36323,5,0.1
   s,A2_56C_3hr,36330,5,0.1

This file will reduce four different samples. An empty-sample will be substracted to each
sample to remove the background signal.

- First column: either empty-sample `b` (background) or sample `s`. These are all the allowed options.
- Second column: name of the sample, for your own reference.
- Third column: starting run number associate with this sample.
  The run numbers associated to each sample are found in the `OnCat data portal <https://oncat.ornl.gov>`_.
- Forth column: total number of runs associated with this sample, including the first.
  For instance, `36308,5` instructs `reduceUSANS` to reduce together runs
  `36308`, `36309`, `36310`,  `36311`, and `36312`.
- Fifth column: thickness of the sample, in centimeters.

Edit CSV files with your preferred text editor (vim, atom, nano, etc) or Excel
but make sure you save it as a CSV file, not a Excel spread sheet (xls/xlsx).

Run the reducing script by passing the path to the CSV setup file.

.. code-block:: bash

   (usansred) $> reduceUSANS setup.csv

Once reduction is finished, subdirectory `result/` is created containing the following files:

- `summary.xlsx` containing sketchy plots of the data for a quick review.
- Reduced data files. For example:

  + `UN_X5D2_8_det_1.txt` (**.txt**) is the stitched data (scaled).
  + `UN_X5D2_8_det_1_lb.txt` (**_lb.txt**) is the data after log binning.
  + `UN_X5D2_8_det_1_lbs.txt` (**_lbs.txt**) is the log binned data after background subtraction.
