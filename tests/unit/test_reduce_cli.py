"""Unit tests for the reduceUSANS command-line parser."""

import argparse

import usansred.reduce
from usansred.reduce import parse_args


def test_parse_args_logbin() -> None:
    args = parse_args(["--logbin", "setup.csv"])

    assert args.path == "setup.csv"
    assert args.logbin is True


def test_parse_args_output() -> None:
    args = parse_args(["--output", "reduced", "setup.json"])

    assert args.path == "setup.json"
    assert args.output == "reduced"


def test_parse_args_enables_argcomplete_before_parsing(monkeypatch) -> None:
    events: list[str] = []

    def autocomplete(parser: argparse.ArgumentParser) -> None:
        assert isinstance(parser, argparse.ArgumentParser)
        events.append("autocomplete")

    def parse_known_args(
        _self: argparse.ArgumentParser,
        _args: list[str] | None = None,
        _namespace: argparse.Namespace | None = None,
    ) -> tuple[argparse.Namespace, list[str]]:
        events.append("parse")
        return argparse.Namespace(path="setup.json", logbin=False, output=""), []

    monkeypatch.setattr(usansred.reduce.argcomplete, "autocomplete", autocomplete)
    monkeypatch.setattr(argparse.ArgumentParser, "parse_known_args", parse_known_args)

    parse_args(["setup.json"])

    assert events == ["autocomplete", "parse"]
