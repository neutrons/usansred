.. _release_notes:

Release Notes
=============

<Next Release>
--------------
(date of release)

**Of interest to the User**:

- PR #XYZ: one-liner description

**Of interest to the Developer:**

- PR 18: update Mantid dependency to 6.12
- PR 17: instructions to add/replace data files
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
