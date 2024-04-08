.. conda_environments

Conda Environments
==================

Three conda environments are available in the analysis nodes, beamline machines, as well as the
jupyter notebook severs. On a terminal:

.. code-block:: bash

   $> conda activate <environment>

where `<environment>` is one of `usansred`, `usansred-qa`, and `usansred-dev`

usansred Environment
--------------------
Activates the latest stable release of `usansred`. Typically users will reduce their data in this environment.

usansred-qa Environment
-----------------------
Activates a release-candidate environment.
Instrument scientists and computational instrument scientists will carry out testing on this environment
to prevent bugs being introduce in the next stable release.

usansred-dev Environment
------------------------
Activates the environment corresponding to the latest changes in the source code.
Instrument scientists and computational instrument scientists will test the latest changes to `usansred` in this
environment.
