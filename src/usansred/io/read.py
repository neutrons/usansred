# TODO: Modify to allow multiple backgrounds
import json
import logging
import os
from copy import deepcopy
from importlib import resources

from jsonschema import Draft202012Validator, validators
from jsonschema import ValidationError as JsonSchemaValidationError


def _validator_with_defaults(validator_class: type[Draft202012Validator]) -> type[Draft202012Validator]:
    """Create a JSON schema validator that inserts schema defaults."""
    validate_properties = validator_class.VALIDATORS["properties"]
    validate_required = validator_class.VALIDATORS["required"]

    def set_defaults(validator, properties, instance, schema):  # noqa ANN001
        if isinstance(instance, dict):
            for property_name, subschema in properties.items():
                if "default" in subschema and property_name not in instance:
                    instance[property_name] = deepcopy(subschema["default"])

        yield from validate_properties(validator, properties, instance, schema)

    def validate_required_defaults(validator, required, instance, schema):  # noqa ANN001
        if not isinstance(instance, dict):
            yield from validate_required(validator, required, instance, schema)
            return

        properties = schema.get("properties", {})
        for property_name in required:
            if property_name in instance:
                continue

            subschema = properties.get(property_name, {})
            if "default" in subschema:
                instance[property_name] = deepcopy(subschema["default"])
                continue

            yield JsonSchemaValidationError(
                f"Required property '{property_name}' is missing from input configuration "
                "and no default value is defined in the schema."
            )

    return validators.extend(
        validator_class,
        {
            "properties": set_defaults,
            "required": validate_required_defaults,
        },
    )


DefaultValidatingDraft202012Validator = _validator_with_defaults(Draft202012Validator)


def _read_schema() -> dict:
    """Read the JSON schema for USANSRED setup files."""
    schema_path = resources.files("usansred.io").joinpath("usansred.json")
    with schema_path.open("r", encoding="utf-8") as schema_file:
        return json.load(schema_file)


def _format_json_path(error: JsonSchemaValidationError) -> str:
    """Format a JSON schema error path for user-facing messages."""
    if not error.path:
        return ""

    path = "".join(f"[{item}]" if isinstance(item, int) else f".{item}" for item in error.path)
    return path.lstrip(".")


def _format_validation_error(error: JsonSchemaValidationError) -> str:
    """Format a JSON schema validation error for setup-file users."""
    path = _format_json_path(error)
    prefix = f"{path}: " if path else ""

    if (
        error.validator == "additionalProperties"
        and error.validator_value is False
        and isinstance(error.instance, dict)
    ):
        allowed_properties = set(error.schema.get("properties", {}))
        extra_properties = sorted(set(error.instance) - allowed_properties)
        extras = ", ".join(repr(property_name) for property_name in extra_properties)
        return f"{prefix}Additional properties are not allowed: {extras}"

    return f"{prefix}{error.message}"


def _validate_config(config: dict) -> None:
    """Validate a reduction configuration and insert schema defaults."""
    schema = _read_schema()
    validator = DefaultValidatingDraft202012Validator(schema)
    errors = sorted(validator.iter_errors(config), key=lambda error: list(error.path))
    if not errors:
        return

    error = errors[0]
    raise ValueError(_format_validation_error(error))


def config_from_csv(csv_path: str) -> dict:
    """Reads the reduction configuration from a CSV file.

    Returns
    -------
    dict
        A dictionary with keys ``"background"`` (a dict or ``None``) and
        ``"samples"`` (a list of dicts).
    """
    import csv

    background = None
    samples = []
    with open(csv_path, newline="", encoding="utf-8-sig") as f:
        reader = csv.reader(f, delimiter=",")

        for row in reader:
            # skip comments and empty rows
            if not row or row[0].startswith("#"):
                continue

            exclude_runs = []
            if len(row) == 6:
                exclude_runs = [int(x) for x in row[5].split(";")]

            try:
                sample = {
                    "name": row[1],
                    "start_scan_num": int(row[2]),
                    "num_of_scans": int(row[3]),
                    "thickness": float(row[4]),
                    "exclude": exclude_runs,
                }
            except Exception as e:  # noqa E722
                sample = None
                logging.info(f"Error parsing sample {row}: {e}")
                # traceback.print_exc()

            if sample is not None:
                if row[0] == "b":
                    # Check if this sample is the empty sample
                    sample["is_background"] = True
                    background = sample
                else:
                    samples.append(sample)

    return {"background": background, "samples": samples}


def config_from_json(json_path: str) -> dict:
    """Reads the reduction configuration from a JSON file.

    Returns
    -------
    dict
        A dictionary with keys ``"background"`` (a dict or ``None``) and
        ``"samples"`` (a list of dicts), among other entries
    """
    with open(json_path, "r") as f:
        config = json.load(f)

    _validate_config(config)

    # Parse the background configuration
    bg = config.get("background")
    if bg:
        try:
            background = {
                "name": bg["name"],
                "start_scan_num": int(bg["start_scan_num"]),
                "num_of_scans": int(bg["num_of_scans"]),
                "thickness": float(bg["thickness"]),
                "is_background": True,
            }
        except KeyError as e:
            logging.info(f"Missing key in background configuration: {e}")
            background = None
        except (TypeError, ValueError) as e:
            logging.info(f"Error parsing background configuration: {e}")
            background = None
    else:
        logging.info("No background sample found in the configuration.")
        background = None
    config["background"] = background

    _samples: list[dict] = config.get("samples")
    if not _samples:
        logging.info("No samples found in the configuration.")
        config["samples"] = []
        return config

    # Parse the sample configurations
    samples = []
    for s in _samples:
        try:
            sample = {
                "name": s["name"],
                "start_scan_num": int(s["start_scan_num"]),
                "num_of_scans": int(s["num_of_scans"]),
                "thickness": float(s["thickness"]),
                "exclude": [int(x) for x in s.get("exclude", [])],
            }
            samples.append(sample)
        except Exception as e:  # noqa E722
            logging.info(f"Error parsing sample {s}: {e}")
            # traceback.print_exc()
    config["samples"] = samples
    return config


def read_config(file_path: str) -> dict:
    """Wrapper function to read configuration from different file formats.

    Parameters
    ----------
    file_path: str
        Path to the reduction configuration file.

    Returns
    -------
    dict
        A dictionary with a minumum set of keys ``"background"`` (a dict or ``None``) and
        ``"samples"`` (a list of dicts).
    """

    _, ext = os.path.splitext(file_path)
    if ext.lower() == ".csv":
        return config_from_csv(file_path)
    elif ext.lower() == ".json":
        return config_from_json(file_path)
    else:
        raise ValueError(f"Unsupported configuration file format: {ext}")
