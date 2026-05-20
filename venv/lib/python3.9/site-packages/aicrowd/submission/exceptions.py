"""
Exceptions for submission subcommand
"""

from aicrowd.exceptions import CLIException


class SubmissionException(CLIException):
    """
    Base exception for the challenge subcommand
    """


class ChallengeNotFoundException(SubmissionException):
    """
    The queried challenge was not found
    """


class InvalidChallengeDirException(SubmissionException):
    """
    The command was given an invalid challenge directory
    """


class SubmissionFileException(SubmissionException):
    """
    The file given for submission was invalid
    """


class SubmissionUploadException(SubmissionException):
    """
    Something went wrong while uploading the file
    """


class SubmissionGitException(SubmissionException):
    """
    Something went wrong while doing some git operation
    """
