import logging
import sys
from contextlib import contextmanager
from pathlib import Path
from typing import Iterator, Literal, Optional, Union

FORMAT = "%(asctime)s | %(levelname)-4s | %(name)-8s | %(message)s"
DATEFMT = "%Y-%m-%d %H:%M:%S"
LOG_FMT = logging.Formatter(fmt=FORMAT, datefmt=DATEFMT)


def set_log_config(level: Union[str, int] = logging.INFO):
    """Sets basic logging config and format for the root logger and all existing loggers,
    ensuring that all log messages are printed to stdout with the same format.

    Parameters
    ----------
    level : Union[str, int], optional
        The logging level, by default logging.INFO
    """

    logging.basicConfig(
        format=FORMAT,
        datefmt=DATEFMT,
        level=level,
        stream=sys.stdout,
        force=True,
    )

    for logger in [logging.getLogger(name) for name in logging.root.manager.loggerDict]:
        logger.setLevel(level)

    return


def get_logger(name: str, level: Union[str, int] = logging.INFO) -> logging.Logger:
    """Get the logger of a given name and level, adding a stream handler to stdout if none exists

    Parameters
    ----------
    name : str
        Name of the logger to retrieve.
    level : Union[str, int], optional
        Logging level for the logger, by default logging.INFO"""

    logger = logging.getLogger(name)
    logger.setLevel(level)
    logger.propagate = False

    # Shorten level names to 4 characters for better log formatting
    logging.addLevelName(logging.DEBUG, "DBUG")
    logging.addLevelName(logging.INFO, "INFO")
    logging.addLevelName(logging.WARNING, "WARN")
    logging.addLevelName(logging.ERROR, "ERR")
    logging.addLevelName(logging.CRITICAL, "CRIT")

    if not any(isinstance(h, logging.StreamHandler) for h in logger.handlers):
        stream_handler = logging.StreamHandler(sys.stdout)
        stream_handler.setFormatter(LOG_FMT)
        stream_handler.setLevel(level)
        logger.addHandler(stream_handler)

    return logger


def add_log_fh(logger: logging.Logger, logfile: str, mode: Literal["w", "a"] = "w") -> logging.Handler:
    """Add a file handler to a logger, replacing any existing one for the same file.

    Parameters
    ----------
    logger : logging.Logger
        The logger to which the file handler will be added.
    logfile : str
        The path to the log file. If the file or its parent directories do not exist, they will be created.
    mode : Literal["w", "a"], optional
        The file mode for the handler, either "w" for overwrite or "a" for append, by default "w".

    Returns
    -------
    logging.Handler
        The file handler that was added to the logger.
    """

    resolved_logfile = str(Path(logfile).resolve())
    Path(resolved_logfile).parent.mkdir(parents=True, exist_ok=True)

    # Keep only one open handler per logfile so callers can toggle logging on/off.
    remove_log_fh(logger, resolved_logfile)

    file_handler = logging.FileHandler(resolved_logfile, mode=mode)
    file_handler.setFormatter(LOG_FMT)
    file_handler.setLevel(logging.DEBUG)
    logger.addHandler(file_handler)
    logger.setLevel(logging.DEBUG)
    return file_handler


def remove_log_fh(logger: logging.Logger, logfile: str | None = None) -> int:
    """Close and remove file handlers from a logger.

    If logfile is provided, only handlers for that file are removed.
    Returns the number of handlers removed.
    """

    resolved_logfile = str(Path(logfile).resolve()) if logfile is not None else None
    removed = 0

    for handler in list(logger.handlers):
        if not isinstance(handler, logging.FileHandler):
            continue

        if (
            resolved_logfile is not None
            and Path(handler.baseFilename).resolve().as_posix() != Path(resolved_logfile).as_posix()
        ):
            continue

        logger.removeHandler(handler)
        handler.close()
        removed += 1

    return removed


@contextmanager
def log_to_file(logger: logging.Logger, logfile: str) -> Iterator[logging.Handler]:
    """Temporarily attach a file handler and close it on exit."""

    handler = add_log_fh(logger, logfile)
    try:
        yield handler
    finally:
        remove_log_fh(logger, logfile)
