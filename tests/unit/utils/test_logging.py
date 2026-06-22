import logging
import os
from pathlib import Path

from usansred.utils.logging import add_log_fh, get_logger, log_to_file, remove_log_fh, set_log_config


def test_logger():
    logger = get_logger("test_logger")
    assert logger is not None
    assert isinstance(logger, logging.Logger)
    assert logger.level == logging.INFO
    assert logger.name == "test_logger"
    assert len(logger.handlers) == 1
    assert isinstance(logger.handlers[0], logging.StreamHandler)

    set_log_config(logging.DEBUG)
    assert logger.level == logging.DEBUG

    logfile = "test.log"
    add_log_fh(logger, logfile)
    assert len(logger.handlers) == 2
    assert isinstance(logger.handlers[1], logging.FileHandler)
    logger.info("Test log message")
    assert logger.handlers[1].baseFilename == str(Path(logfile).absolute())

    removed = remove_log_fh(logger, logfile)
    assert removed == 1
    assert len(logger.handlers) == 1
    os.remove(logfile)


def test_log_to_file_context_manager():
    logger = get_logger("test_logger_context")
    logfile = "test_context.log"

    with log_to_file(logger, logfile) as handler:
        logger.info("test context message")
        assert handler in logger.handlers

    assert all(not isinstance(h, logging.FileHandler) for h in logger.handlers)
    assert os.path.exists(logfile)
    os.remove(logfile)


def test_add_log_fh_append_mode():
    logger = get_logger("test_logger_append")
    logfile = "test_append.log"

    add_log_fh(logger, logfile, mode="w")
    logger.info("first line")
    remove_log_fh(logger, logfile)

    add_log_fh(logger, logfile, mode="a")
    logger.info("second line")
    remove_log_fh(logger, logfile)

    with open(logfile, encoding="utf-8") as f:
        content = f.read()

    assert "first line" in content
    assert "second line" in content
    os.remove(logfile)
