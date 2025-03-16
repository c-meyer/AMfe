"""
This module provides convenience functions for AMfe's logging capabilities.
"""
import logging


def log_debug(name, message):
    """
    Sends a debug message to specified logger.

    Parameters
    ----------
    name: string
        Name of the logger.
    message: string
        Debug Message.

    Returns
    -------

    """
    logger = logging.getLogger(name)
    logger.debug(message)

def log_info(name, message):
    """
    Sends a info message to specified logger.

    Parameters
    ----------
    name: string
        Name of the logger.
    message: string
        Info Message.

    Returns
    -------

    """
    logger = logging.getLogger(name)
    logger.info(message)

def log_warning(name, message):
    """
    Sends a warning message to specified logger.

    Parameters
    ----------
    name: string
        Name of the logger.
    message: string
        Warning Message.

    Returns
    -------

    """
    logger = logging.getLogger(name)
    logger.warning(message)

def log_error(name, message):
    """
    Sends a error message to specified logger.

    Parameters
    ----------
    name: string
        Name of the logger.
    message: string
        Error Message.

    Returns
    -------

    """
    logger = logging.getLogger(name)
    logger.error(message)

def log_critical(name, message):
    """
    Sends a critical message to specified logger.

    Parameters
    ----------
    name: string
        Name of the logger.
    message: string
        Critical Message.

    Returns
    -------

    """
    logger = logging.getLogger(name)
    logger.critical(message)

def log_exception(name, e):
    """
    Sends an exception message to specified logger.

    Parameters
    ----------
    name: string
        Name of the logger.
    e: Exception
        Raised Exception whose information should be logged.

    Returns
    -------

    """
    logger = logging.getLogger(name)
    logger.exception(e)
