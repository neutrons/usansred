# USANSRED
Backend code for the reduction of USANS data.

5. Having code coverage, `codecov.yaml` is **strongly recommended**, please refer to [Code coverage](https://coverage.readthedocs.io/en/coverage-5.5/) for more information.


6. Adjust the demo Github action yaml files for CI/CD. For more information about Github action, please refer to [Github action](https://docs.github.com/en/actions).

    6.1 Specify package name at: .github/workflows/package.yml#L34

    6.2 Specify package name at: .github/workflows/package.yml#L46


8. Adjust `pyproject.toml` to match your project. For more information about `pyproject.toml`, please refer to [pyproject.toml](https://www.python.org/dev/peps/pep-0518/).

    8.1 Specify package name at: pyproject.toml#L2

    8.2 Specify package description at: pyproject.toml#L3

    8.3 Specify package name at: pyproject.toml#L40

    8.4 We strongly recommended using a single `pyproject.toml` file to manage all the project metadata, including the project name, version, author, license, etc.

    8.5 Python is moving away from `setup.cfg`/`setup.py`, and we would like to follow the trend for our new projects.


9. Specify package name at  src/usansred



11. Clear the content of this file and add your own README.md as the project README file. We recommend putting badges of the project status at the top of the README file. For more information about badges, please refer to [shields.io](https://shields.io/).


# Developer's corner

## Updating mantid dependency
The mantid version and the mantid conda channel (`mantid/label/main` or `mantid/label/nightly`) **must** be
synchronized across these files:
- environment.yml
- conda.recipe/meta.yml
- .github/workflows/package.yml

## Read *the* Docs
A repository webhook is setup to automatically trigger the latest documentation build.

### Manual build
You can manually trigger a build in your working directory with the following:
```
cd docs
make clean html
```
This creates the html files at `docs/build/html`