"""
Exceptions for dataset subcommand
"""

from aicrowd.exceptions import CLIException


class DatasetException(CLIException):
    """
    Base exception for the dataset subcommand
    """


class ChallengeNotFoundException(DatasetException):
    """
    The queried challenge was not found
    """


class DatasetNotFoundException(DatasetException):
    """
    The requested dataset could not be found
    """
