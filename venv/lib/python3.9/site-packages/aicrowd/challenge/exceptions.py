"""
Exceptions for challenge subcommand
"""

from aicrowd.exceptions import CLIException


class ChallengeException(CLIException):
    """
    Base exception for the challenge subcommand
    """


class ChallengeNotFoundException(ChallengeException):
    """
    The queried challenge was not found
    """
