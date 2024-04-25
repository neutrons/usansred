.. developer_documentation

Developer Documentation
=======================

.. toctree::
   :maxdepth: 1

Local Environment
-----------------
For purposes of development, create conda environment `usansred` with file `environment.yml`, and then
install the package in development mode with `pip`:

.. code-block:: bash

   $> cd cd /path/to/usansred/
   $> conda create env --solver libmamba --file ./environment.yml
   $> conda activate usansred
   (usansred)$> pip install -e ./

By installing the package in development mode, one doesn't need to re-install package `usanred` in conda
environment `usansred` after every change to the source code.

pre-commit Hooks
----------------


Updating mantid dependency
--------------------------
The mantid version and the mantid conda channel (`mantid/label/main` or `mantid/label/nightly`) **must** be
synchronized across these files:

- environment.yml
- conda.recipe/meta.yml
- .github/workflows/package.yml

Using the Data Repository usansred-data
---------------------------------------
To run the integration tests in your local environment, it is necessary first to download the data files.
Because of their size, the files are stored in the Git LFS repository
`usansred-data <https://code.ornl.gov/sns-hfir-scse/infrastructure/test-data/usansred-data>`_.

It is necessary to have package `git-lfs` installed in your machine.

.. code-block:: bash

   $> sudo apt install git-lfs

After this step, initialize or update the data repository:

.. code-block:: bash

   $> cd /path/to/usanred
   $> git submodule update --init

This will either clone `usansred-data` into `/path/to/usanred/tests/usansred-data` or
bring the `usansred-data`'s refspec in sync with the refspec listed within file `/path/to/usanred/.gitmodules`.

An intro to Git LFS in the context of the Neutron Data Project is found in the
`Confluence pages <https://ornl-neutrons.atlassian.net/wiki/spaces/NDPD/pages/19103745/Using+git-lfs+for+test+data>`_
(login required).


Coverage reports
----------------

GitHuh actions create reports for unit and integration tests, then combine into one report and upload it to
`Codecov <https://app.codecov.io/gh/neutrons/usansred>`_.


Building the documentation
--------------------------
A repository webhook is setup to automatically trigger the latest documentation build by GitHub actions.
To manually build the documentation:

.. code-block:: bash

   $> conda activate usansred
   (usansred)$> cd /path/to/usansred/docs
   (usansred)$> make docs

After this, point your browser to
`file:///path/to/usansred/docs/build/html/index.html`
