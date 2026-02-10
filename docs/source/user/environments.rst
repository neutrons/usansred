.. environments

Pixi Environments
==================

Three pixi environments are available in the analysis nodes, beamline machines, as well as the
jupyter notebook severs.

To activate an environment, in a terminal run the command:

.. code-block:: bash

   $ nsd-pixi-shell.sh <environment>
   # which is a wrapper around the command:
   $ pixi shell --manifest-path /usr/local/pixi/<environment>

where ``<environment>`` is one of ``usansred``, ``usansred_qa``, and ``usansred_dev``

usansred Environment
--------------------
The latest stable release of ``usansred``. Typically users will reduce their data in this environment.

usansred-qa Environment
-----------------------
The release-candidate environment.
Instrument scientists and computational instrument scientists will carry out testing on this environment
to prevent bugs being introduce in the next stable release.

usansred-dev Environment
------------------------
The environment corresponding to the latest changes in the source code.
Instrument scientists and computational instrument scientists will test the latest changes to ``usansred`` in this
environment.
