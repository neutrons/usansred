.. _release_notes:

Release Notes
=============

..
   1.1.0
   -----
   (date of release, YYY-MM-DD)

   **Of interest to the User**:

   **Of interest to the Developer:**
..

1.1.0
-----
(2025-04-01)

**Of interest to the Developer:**

- PR 19: Explicitly denotes the encoding regardless of Windows and Linux. It won't change anything in Linux.
- PR 18: update Mantid dependency to 6.12
- PR 17: instructions to add/replace data files
- PR 17: change to absolute tolerance while subtracting background
- PR 15: update Mantid dependency to 6.11
- PR 14: transition from pip to conda when installing dependency finddata
- PR 13: Take average of intensity values with duplicate Q AND solve the issue with bg interpolation when bg q and sample q values are too close or identical
- PR 12: switch from mantid to mantidworkbench conda package

1.0.0
-----
2024-05-06

This is the first release of USANSRED.

- repository implements the `OpenSSF Best Practices Badge Program <https://www.bestpractices.dev/en/criteria/0>`_
- codecov reports for every accepted change, uploaded to `codecov usansred <https://app.codecov.io/gh/neutrons/usansred>`_
- generation of conda packages, uploaded to the `neutrons channel <https://anaconda.org/neutrons/usansred/files>`_
- `online documentation <https://usansred.readthedocs.io/en/latest/>`_ for the user and the developer.
- executable script `reduceUSANS` is the entry point for all reduction jobs

1.0.1
-----
2025-02-28

- Changed the algorithm in background subtraction when binning is not applied. Changed to absolute tolerance when matching the sample data points with background data points, so that low Q values with very small differences can be compared and subtracted.
