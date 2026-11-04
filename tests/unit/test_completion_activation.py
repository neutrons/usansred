"""Tests for reduceUSANS shell-completion activation."""

import os
import subprocess
import tomllib
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
ACTIVATION_SCRIPT = PROJECT_ROOT / "scripts" / "activate_reduceUSANS_argcomplete.sh"
PYPROJECT = PROJECT_ROOT / "pyproject.toml"


def test_pixi_activation_registers_argcomplete_script() -> None:
    pyproject = tomllib.loads(PYPROJECT.read_text(encoding="utf-8"))

    assert pyproject["tool"]["pixi"]["target"]["unix"]["activation"]["scripts"] == [
        "scripts/activate_reduceUSANS_argcomplete.sh"
    ]


def test_wheel_installs_conda_activation_script() -> None:
    pyproject = tomllib.loads(PYPROJECT.read_text(encoding="utf-8"))

    assert (
        pyproject["tool"]["hatch"]["build"]["targets"]["wheel"]["shared-data"][
            "scripts/activate_reduceUSANS_argcomplete.sh"
        ]
        == "etc/conda/activate.d/usansred_argcomplete.sh"
    )


def test_activation_script_registers_reduce_usans_for_bash() -> None:
    command = f"source {ACTIVATION_SCRIPT}; complete -p reduceUSANS"
    # Drop the guard variable, which is inherited when the tests run from an already activated
    # shell and would make the script skip the registration
    env = {key: value for key, value in os.environ.items() if key != "USANSRED_ARGCOMPLETE_REGISTERED"}

    result = subprocess.run(
        ["bash", "--noprofile", "--norc", "-ic", command],
        check=True,
        text=True,
        capture_output=True,
        env=env,
    )

    assert "_python_argcomplete" in result.stdout
    assert "reduceUSANS" in result.stdout
